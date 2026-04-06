"""
PINN Training Script for Soil Moisture Estimation
===================================================
Trains a Physics-Informed Neural Network that spatially discretizes Richards'
equation into 6 soil layers using a two-stage strategy:
  Stage 1: Derivative loss (3000 epochs)
  Stage 2: RK4 tracking loss (1000 epochs)

Hyperparameters obtained via Optuna (200+30 trials).
Output: ../weights/pinn_weights_H56_lr7.97e-05.weights.h5

Requirements:
  tensorflow >= 2.10 (float64), tfsolverMIconstrb_float64.py,
  training data in ../data/
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import time
import random
import csv
import sys
import os


# Add current folder to sys.path so we can import our local module
sys.path.append(os.getcwd())


# --- Set Random Seed for Reproducibility ---
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# Import the RK4 integrator from your module
from tfsolverMIconstrb_float64 import rk4_integration

# Resolve data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "."  # fallback: data files in current directory

# Load data
campbell = pd.read_json(os.path.join(DATA_DIR, "campbell_1h.json"))
davis = pd.read_json(os.path.join(DATA_DIR, "davis_1h.json"))
irrimeter = pd.read_json(os.path.join(DATA_DIR, "irrimeter_1h.json"))

# Parse timestamps
campbell['timestamp'] = pd.to_datetime(campbell['timestamp'])
davis['timestamp'] = pd.to_datetime(davis['timestamp'])
irrimeter['timestamp'] = pd.to_datetime(irrimeter['timestamp'])

# Rename and select required fields
campbell = campbell.rename(columns={"measurement": "irrigation"})
davis = davis.rename(columns={
    "measured_rain_mm_h": "rain",
    "potential_evapotranspiration_mm_h": "et0"
})
irrimeter = irrimeter[["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]]

# Merge on timestamp
df = pd.merge(davis[["timestamp", "rain", "et0"]], campbell[["timestamp", "irrigation"]], on="timestamp", how="inner")
df = pd.merge(df, irrimeter, on="timestamp", how="inner")

# Preprocess ET0 to ensure non-negative values
df["et0"] = df["et0"].clip(lower=0.0)

# Calculate time in hours since start
df["time_hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0

# ----------------------------
# Optional Normalization by Constants
# ----------------------------
normalize = {
    "time_hours": 1,     # scale time to ~[0, 1]
    "rain": 30,             # scale rain to ~[0, 1]
    "et0": 0.8,
    "irrigation": 1,
    "moisture": 40         # assuming moisture in % format
}

# Apply normalization
df["time_scaled"] = df["time_hours"] / normalize["time_hours"]
df["rain_scaled"] = df["rain"] / normalize["rain"]
df["et0_scaled"] = df["et0"] / normalize["et0"]
df["irrigation_scaled"] = df["irrigation"] / normalize["irrigation"]
df["moisture_1_scaled"] = df["moisture_1"] / normalize["moisture"]
df["moisture_2_scaled"] = df["moisture_2"] / normalize["moisture"]
df["moisture_3_scaled"] = df["moisture_3"] / normalize["moisture"]
df["moisture_4_scaled"] = df["moisture_4"] / normalize["moisture"]
df["moisture_5_scaled"] = df["moisture_5"] / normalize["moisture"]
df["moisture_6_scaled"] = df["moisture_6"] / normalize["moisture"]

# Sort and clean
df = df.sort_values("time_hours").dropna().reset_index(drop=True)

# Prepare tensors
t_tf = tf.constant(df["time_scaled"].values.astype(np.float64), dtype=tf.float64)  # Scaled time
inputs_tf = tf.constant(df[["irrigation_scaled", "rain_scaled", "et0_scaled"]].values.astype(np.float64), dtype=tf.float64)  # Inputs scaled
states_tf = tf.constant(df[["moisture_1_scaled", "moisture_2_scaled",
                            "moisture_3_scaled", "moisture_4_scaled",
                            "moisture_5_scaled", "moisture_6_scaled"]].values.astype(np.float64), dtype=tf.float64)  # Moisture scaled

# Optional: if you want NumPy arrays
time_np = t_tf.numpy()
inputs_np = inputs_tf.numpy()
states_np = states_tf.numpy()

# Shapes for verification
print("Time shape:", t_tf.shape)
print("Inputs shape:", inputs_tf.shape)
print("States shape:", states_tf.shape)

# RK4 solver parameters
dt = 1.0  # hourly step
# Mid-step values for loss calculation
dm_mid= (states_tf[1:] - states_tf[:-1]) / dt
m_mid = states_tf[:-1]        # state at time t
in_mid = inputs_tf[:-1]   # input at time t
t_mid = t_tf[:-1]        # time t

m0_np = states_tf[0].numpy()

dm_mid_np = dm_mid.numpy()
m_mid_np = m_mid.numpy()
in_mid_np = in_mid.numpy()
t_mid_np = t_mid.numpy()

class TwoTankPINN(tf.keras.Model):
    def __init__(self, neurons=56):
        super(TwoTankPINN, self).__init__()
        glorot_init = tf.keras.initializers.GlorotNormal()

        def scaled_he_init(shape, dtype=None):
            return 0.5 * tf.keras.initializers.HeNormal()(shape, dtype=dtype)

        # layer 1 outflow branch
        self.f1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 2 outflow branch
        self.f2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 3 outflow branch
        self.f3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 4 outflow branch
        self.f4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # layer 5 outflow branch
        self.f5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.f5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.f6_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=True)
        self.f6_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=scaled_he_init, use_bias=True)
        #################
        # Inter-tank flow branch
        self.f12_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f12_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f23_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f23_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f34_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f34_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f45_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f45_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)

        self.f56_dense1 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_dense2 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_dense3 = tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer=glorot_init, use_bias=False)
        self.f56_out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=glorot_init, use_bias=False)


        # Trainable inflow scaler
        # Irrigation branch
        self.inflow_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.inflow_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # Trainable inflow scaler
        # rain branch
        self.rain1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.rain6_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.rain6_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        # plant_coef_1
        self.pc1_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        # plant_coef_2
        self.pc2_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc3_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc4_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc5_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)

        self.pc6_dense1 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense2 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense3 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_dense4 = tf.keras.layers.Dense(2*neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.pc6_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=glorot_init, use_bias=False)


        # Lateral outflow branches
        self.fl1_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl1_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl2_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl2_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl3_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl3_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl4_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl4_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

        self.fl5_dense1 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_dense2 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_dense3 = tf.keras.layers.Dense(neurons, activation='mish', kernel_initializer=glorot_init, use_bias=False)
        self.fl5_out = tf.keras.layers.Dense(1, activation='mish', kernel_initializer=scaled_he_init, use_bias=False)

       

    def call(self, m, input_vec, t):

        # Unpack states
        m1 = tf.reshape(m[:, 0], [-1, 1])
        m2 = tf.reshape(m[:, 1], [-1, 1])
        m3 = tf.reshape(m[:, 2], [-1, 1])
        m4 = tf.reshape(m[:, 3], [-1, 1])
        m5 = tf.reshape(m[:, 4], [-1, 1])
        m6 = tf.reshape(m[:, 5], [-1, 1])

        # Compute differences between layers
        dm12 = m1 - m2
        dm23 = m2 - m3
        dm34 = m3 - m4
        dm45 = m4 - m5
        dm56 = m5 - m6

        # Input unpack
        qi   = tf.reshape(input_vec[:, 0], [-1, 1])  # irrigation
        rain = tf.reshape(input_vec[:, 1], [-1, 1])  # rain
        et0  = tf.reshape(input_vec[:, 2], [-1, 1])  # ET0
        t = tf.reshape(t, [-1, 1])  # time

        m1 = tf.identity(m1, name='m1')
        m2 = tf.identity(m2, name='m2')
        m3 = tf.identity(m3, name='m3')
        m4 = tf.identity(m4, name='m4')
        m5 = tf.identity(m5, name='m5')
        m6 = tf.identity(m6, name='m6')
        dm12 = tf.identity(dm12, name='dm12')
        dm23 = tf.identity(dm23, name='dm23')
        dm34 = tf.identity(dm34, name='dm34')
        dm45 = tf.identity(dm45, name='dm45')
        dm56 = tf.identity(dm56, name='dm56')
        qi = tf.identity(qi, name='qi')
        rain = tf.identity(rain, name='rain')
        et0 = tf.identity(et0, name='et0')
        t = tf.identity(t, name='t')



        # Outflow layer 1
        with tf.GradientTape() as tape:
            tape.watch(m1)
            x1 = self.f1_dense1(m1)
            x1 = self.f1_dense2(x1)
            x1 = self.f1_dense3(x1)
            f1 = self.f1_out(x1)
        df1_dm1 = tape.gradient(f1, m1)

        # Outflow layer 2
        with tf.GradientTape() as tape:
            tape.watch(m2)
            x2 = self.f2_dense1(m2)
            x2 = self.f2_dense2(x2)
            x2 = self.f2_dense3(x2)
            f2 = self.f2_out(x2)
        df2_dm2 = tape.gradient(f2, m2)

        # Outflow layer 3
        with tf.GradientTape() as tape:
            tape.watch(m3)
            x3 = self.f3_dense1(m3)
            x3 = self.f3_dense2(x3)
            x3 = self.f3_dense3(x3)
            f3 = self.f3_out(x3)
        df3_dm3 = tape.gradient(f3, m3)

        # Outflow layer 4
        with tf.GradientTape() as tape:
            tape.watch(m4)
            x4 = self.f4_dense1(m4)
            x4 = self.f4_dense2(x4)
            x4 = self.f4_dense3(x4)
            f4 = self.f4_out(x4)
        df4_dm4 = tape.gradient(f4, m4)

        # Outflow layer 5
        with tf.GradientTape() as tape:
            tape.watch(m5)
            x5 = self.f5_dense1(m5)
            x5 = self.f5_dense2(x5)
            x5 = self.f5_dense3(x5)
            f5 = self.f5_out(x5)
        df5_dm5 = tape.gradient(f5, m5)

        # Outflow layer 6
        with tf.GradientTape() as tape:
            tape.watch(m6)
            x6 = self.f6_dense1(m6)
            x6 = self.f6_dense2(x6)
            x6 = self.f6_dense3(x6)
            f6 = self.f6_out(x6)
        df6_dm6 = tape.gradient(f6, m6)

        # Interlayer f12
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m1)
            tape.watch(m2)
            m12_input = tf.concat([m1, m2], axis=1)
            x = self.f12_dense1(m12_input)
            x = self.f12_dense2(x)
            x = self.f12_dense3(x)
            f12 = self.f12_out(x)
        df12_dm1 = tape.gradient(f12, m1)
        df12_dm2 = tape.gradient(f12, m2)
        df12_ddm12 = df12_dm1 - df12_dm2

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m2)
            tape.watch(m3)
            m23_input = tf.concat([m2, m3], axis=1)
            x = self.f23_dense1(m23_input)
            x = self.f23_dense2(x)
            x = self.f23_dense3(x)
            f23 = self.f23_out(x)
        df23_dm2 = tape.gradient(f23, m2)
        df23_dm3 = tape.gradient(f23, m3)
        df23_ddm23 = df23_dm2 - df23_dm3

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m3)
            tape.watch(m4)
            m34_input = tf.concat([m3, m4], axis=1)
            x = self.f34_dense1(m34_input)
            x = self.f34_dense2(x)
            x = self.f34_dense3(x)
            f34 = self.f34_out(x)
        df34_dm3 = tape.gradient(f34, m3)
        df34_dm4 = tape.gradient(f34, m4)
        df34_ddm34 = df34_dm3 - df34_dm4

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m4)
            tape.watch(m5)
            m45_input = tf.concat([m4, m5], axis=1)
            x = self.f45_dense1(m45_input)
            x = self.f45_dense2(x)
            x = self.f45_dense3(x)
            f45 = self.f45_out(x)
        df45_dm4 = tape.gradient(f45, m4)
        df45_dm5 = tape.gradient(f45, m5)
        df45_ddm45 = df45_dm4 - df45_dm5

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(m5)
            tape.watch(m6)
            m56_input = tf.concat([m5, m6], axis=1)
            x = self.f56_dense1(m56_input)
            x = self.f56_dense2(x)
            x = self.f56_dense3(x)
            f56 = self.f56_out(x)
        df56_dm5 = tape.gradient(f56, m5)
        df56_dm6 = tape.gradient(f56, m6)
        df56_ddm56 = df56_dm5 - df56_dm6

        # Inflow
        with tf.GradientTape() as tape:
            tape.watch(qi)
            x = self.inflow_dense1(qi)
            x = self.inflow_dense2(x)
            x = self.inflow_dense3(x)
            inflow = self.inflow_out(x)
        dinflow_dq = tape.gradient(inflow, qi)

        # Rain
        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain1_dense1(rain)
            x = self.rain1_dense2(x)
            x = self.rain1_dense3(x)
            in_rain1 = self.rain1_out(x)
        din_rain1_dq = tape.gradient(in_rain1, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain2_dense1(rain)
            x = self.rain2_dense2(x)
            x = self.rain2_dense3(x)
            in_rain2 = self.rain2_out(x)
        din_rain2_dq = tape.gradient(in_rain2, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain3_dense1(rain)
            x = self.rain3_dense2(x)
            x = self.rain3_dense3(x)
            in_rain3 = self.rain3_out(x)
        din_rain3_dq = tape.gradient(in_rain3, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain4_dense1(rain)
            x = self.rain4_dense2(x)
            x = self.rain4_dense3(x)
            in_rain4 = self.rain4_out(x)
        din_rain4_dq = tape.gradient(in_rain4, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain5_dense1(rain)
            x = self.rain5_dense2(x)
            x = self.rain5_dense3(x)
            in_rain5 = self.rain5_out(x)
        din_rain5_dq = tape.gradient(in_rain5, rain)

        with tf.GradientTape() as tape:
            tape.watch(rain)
            x = self.rain6_dense1(rain)
            x = self.rain6_dense2(x)
            x = self.rain6_dense3(x)
            in_rain6 = self.rain6_out(x)
        din_rain6_dq = tape.gradient(in_rain6, rain)




        # Plant coef 1
                # Plant coef 1
        inputs_pc1 = tf.concat([m1, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc1)
            x_pc1 = self.pc1_dense1(inputs_pc1)
            x_pc1 = self.pc1_dense2(x_pc1)
            x_pc1 = self.pc1_dense3(x_pc1)
            x_pc1 = self.pc1_dense4(x_pc1)
            pc1 = self.pc1_out(x_pc1)
        dpc1_dinputs = tape.gradient(pc1, inputs_pc1)
        dpc1_dm1 = dpc1_dinputs[:, 0:1]
        dpc1_det0 = dpc1_dinputs[:, 1:2]

        # Plant coef 2
        inputs_pc2 = tf.concat([m2, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc2)
            x_pc2 = self.pc2_dense1(inputs_pc2)
            x_pc2 = self.pc2_dense2(x_pc2)
            x_pc2 = self.pc2_dense3(x_pc2)
            x_pc2 = self.pc2_dense4(x_pc2)
            pc2 = self.pc2_out(x_pc2)
        dpc2_dinputs = tape.gradient(pc2, inputs_pc2)
        dpc2_dm2 = dpc2_dinputs[:, 0:1]
        dpc2_det0 = dpc2_dinputs[:, 1:2]

        # Plant coef 3
        inputs_pc3 = tf.concat([m3, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc3)
            x_pc3 = self.pc3_dense1(inputs_pc3)
            x_pc3 = self.pc3_dense2(x_pc3)
            x_pc3 = self.pc3_dense3(x_pc3)
            x_pc3 = self.pc3_dense4(x_pc3)
            pc3 = self.pc3_out(x_pc3)
        dpc3_dinputs = tape.gradient(pc3, inputs_pc3)
        dpc3_dm3 = dpc3_dinputs[:, 0:1]
        dpc3_det0 = dpc3_dinputs[:, 1:2]

        # Plant coef 4
        inputs_pc4 = tf.concat([m4, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc4)
            x_pc4 = self.pc4_dense1(inputs_pc4)
            x_pc4 = self.pc4_dense2(x_pc4)
            x_pc4 = self.pc4_dense3(x_pc4)
            x_pc4 = self.pc4_dense4(x_pc4)
            pc4 = self.pc4_out(x_pc4)
        dpc4_dinputs = tape.gradient(pc4, inputs_pc4)
        dpc4_dm4 = dpc4_dinputs[:, 0:1]
        dpc4_det0 = dpc4_dinputs[:, 1:2]

        # Plant coef 5
        inputs_pc5 = tf.concat([m5, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc5)
            x_pc5 = self.pc5_dense1(inputs_pc5)
            x_pc5 = self.pc5_dense2(x_pc5)
            x_pc5 = self.pc5_dense3(x_pc5)
            x_pc5 = self.pc5_dense4(x_pc5)
            pc5 = self.pc5_out(x_pc5)
        dpc5_dinputs = tape.gradient(pc5, inputs_pc5)
        dpc5_dm5 = dpc5_dinputs[:, 0:1]
        dpc5_det0 = dpc5_dinputs[:, 1:2]

        # Plant coef 6
        inputs_pc6 = tf.concat([m6, et0], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(inputs_pc6)
            x_pc6 = self.pc6_dense1(inputs_pc6)
            x_pc6 = self.pc6_dense2(x_pc6)
            x_pc6 = self.pc6_dense3(x_pc6)
            x_pc6 = self.pc6_dense4(x_pc6)
            pc6 = self.pc6_out(x_pc6)
        dpc6_dinputs = tape.gradient(pc6, inputs_pc6)
        dpc6_dm6 = dpc6_dinputs[:, 0:1]
        dpc6_det0 = dpc6_dinputs[:, 1:2]



        # Lateral outflows
        with tf.GradientTape() as tape:
            tape.watch(m1)
            x_fl1 = self.fl1_dense1(m1)
            x_fl1 = self.fl1_dense2(x_fl1)
            x_fl1 = self.fl1_dense3(x_fl1)
            fl1 = self.fl1_out(x_fl1)
        dfl1_dm1 = tape.gradient(fl1, m1)

        with tf.GradientTape() as tape:
            tape.watch(m2)
            x_fl2 = self.fl2_dense1(m2)
            x_fl2 = self.fl2_dense2(x_fl2)
            x_fl2 = self.fl2_dense3(x_fl2)
            fl2 = self.fl2_out(x_fl2)
        dfl2_dm2 = tape.gradient(fl2, m2)

        with tf.GradientTape() as tape:
            tape.watch(m3)
            x_fl3 = self.fl3_dense1(m3)
            x_fl3 = self.fl3_dense2(x_fl3)
            x_fl3 = self.fl3_dense3(x_fl3)
            fl3 = self.fl3_out(x_fl3)
        dfl3_dm3 = tape.gradient(fl3, m3)

        with tf.GradientTape() as tape:
            tape.watch(m4)
            x_fl4 = self.fl4_dense1(m4)
            x_fl4 = self.fl4_dense2(x_fl4)
            x_fl4 = self.fl4_dense3(x_fl4)
            fl4 = self.fl4_out(x_fl4)
        dfl4_dm4 = tape.gradient(fl4, m4)

        with tf.GradientTape() as tape:
            tape.watch(m5)
            x_fl5 = self.fl5_dense1(m5)
            x_fl5 = self.fl5_dense2(x_fl5)
            x_fl5 = self.fl5_dense3(x_fl5)
            fl5 = self.fl5_out(x_fl5)
        dfl5_dm5 = tape.gradient(fl5, m5)

        

        return (
            f1, f2, f3, f4, f5, f6,
            f12, f23, f34, f45, f56,
            inflow,
            in_rain1, in_rain2, in_rain3, in_rain4, in_rain5, in_rain6,
            pc1, pc2, pc3, pc4, pc5, pc6,
            fl1, fl2, fl3, fl4, fl5,
            df1_dm1, df2_dm2, df3_dm3, df4_dm4, df5_dm5, df6_dm6,
            df12_ddm12, df23_ddm23, df34_ddm34, df45_ddm45, df56_ddm56,
            dinflow_dq,
            din_rain1_dq, din_rain2_dq, din_rain3_dq, din_rain4_dq, din_rain5_dq, din_rain6_dq,
            dpc1_dm1, dpc2_dm2, dpc3_dm3, dpc4_dm4, dpc5_dm5, dpc6_dm6,
            dpc1_det0, dpc2_det0, dpc3_det0, dpc4_det0, dpc5_det0, dpc6_det0,
            dfl1_dm1, dfl2_dm2, dfl3_dm3, dfl4_dm4, dfl5_dm5
        )

# Create an instance of the PINN model
pinn_model = TwoTankPINN(neurons=56)

@tf.function
def system_func_pinn(t, state, input_vec):
    m_batch = tf.reshape(state, [1, 6])
    in_batch = tf.reshape(input_vec, [1, 3])
    t_batch = tf.reshape(t, [1, 1])

    # Updated unpacking to include six rain inputs
    (f1, f2, f3, f4, f5, f6,
     f12, f23, f34, f45, f56,
     inflow,
     in_rain1, in_rain2, in_rain3, in_rain4, in_rain5, in_rain6,
     pc1, pc2, pc3, pc4, pc5, pc6,
     fl1, fl2, fl3, fl4, fl5,
     *_) = pinn_model(m_batch, in_batch, t_batch)

    # Drop batch dimension
    f1, f2, f3, f4, f5, f6 = f1[0, 0], f2[0, 0], f3[0, 0], f4[0, 0], f5[0, 0], f6[0, 0]
    f12, f23, f34, f45, f56 = f12[0, 0], f23[0, 0], f34[0, 0], f45[0, 0], f56[0, 0]
    inflow = inflow[0, 0]
    in_rain1, in_rain2, in_rain3 = in_rain1[0, 0], in_rain2[0, 0], in_rain3[0, 0]
    in_rain4, in_rain5, in_rain6 = in_rain4[0, 0], in_rain5[0, 0], in_rain6[0, 0]
    pc1, pc2, pc3 = pc1[0, 0], pc2[0, 0], pc3[0, 0]
    pc4, pc5, pc6 = pc4[0, 0], pc5[0, 0], pc6[0, 0]
    fl1, fl2, fl3 = fl1[0, 0], fl2[0, 0], fl3[0, 0]
    fl4, fl5 = fl4[0, 0], fl5[0, 0]

    # Updated mass balances with rain per layer
    dh1dt = inflow - f1 - f12 - pc1 + in_rain1 - fl1
    dh2dt = f1 + f12 - f2 - f23 - pc2 + in_rain2 - fl2
    dh3dt = f2 + f23 - f3 - f34 - pc3 + in_rain3 - fl3
    dh4dt = f3 + f34 - f4 - f45 - pc4 + in_rain4 - fl4
    dh5dt = f4 + f45 - f5 - f56 - pc5 + in_rain5 - fl5
    dh6dt = f5 + f56 - f6 - pc6 + in_rain6 

    return tf.stack([dh1dt, dh2dt, dh3dt, dh4dt, dh5dt, dh6dt])

@tf.function
def compute_loss(stage):
    zero64 = tf.constant(0.0, dtype=tf.float64)
    eps   = tf.constant(1e-6, dtype=tf.float64)

    # Define loss weights based on training stage
    if stage == 1:
        # Stage 1: Focus on derivative loss (j1)
        lambda_1 = 1.0      # Derivative loss
        lambda_2 = 0.0      # Trajectory loss (disabled)
        lambda_3 = 113982766.31047939      # Positivity
        lambda_4 = 1773295.4683835912    # Monotonicity
        lambda_5 = 126704655.89679742     # Zero-flow
    else:
        # Stage 2: Focus on trajectory loss (j2)
        lambda_1 = 0.0      # Derivative loss (reduced weight)
        lambda_2 = 1.0      # Trajectory loss
        lambda_3 = 113982766.31047939      # Positivity
        lambda_4 = 1773295.4683835912    # Monotonicity
        lambda_5 = 126704655.89679742     # Zero-flow


    # ── unpack mid-step data ────────────────────────────────────────────────
    m1 = m_mid[:, 0:1]
    m2 = m_mid[:, 1:2]
    m3 = m_mid[:, 2:3]
    m4 = m_mid[:, 3:4]
    m5 = m_mid[:, 4:5]
    m6 = m_mid[:, 5:6]
    dm12 = m1 - m2
    dm23 = m2 - m3
    dm34 = m3 - m4
    dm45 = m4 - m5
    dm56 = m5 - m6

    qi  = in_mid[:, 0:1]
    rain = in_mid[:, 1:2]
    et0 = in_mid[:, 2:3]

    dm1_true = dm_mid[:, 0:1]
    dm2_true = dm_mid[:, 1:2]
    dm3_true = dm_mid[:, 2:3]
    dm4_true = dm_mid[:, 3:4]
    dm5_true = dm_mid[:, 4:5]
    dm6_true = dm_mid[:, 5:6]

    # ── forward pass ────────────────────────────────────────────────────────
    (f1, f2, f3, f4, f5, f6,
     f12, f23, f34, f45, f56,
     inflow,
     in_rain1, in_rain2, in_rain3, in_rain4, in_rain5, in_rain6,
     pc1, pc2, pc3, pc4, pc5, pc6,
     fl1, fl2, fl3, fl4, fl5,
     df1_dm1, df2_dm2, df3_dm3, df4_dm4, df5_dm5, df6_dm6,
     df12_ddm12, df23_ddm23, df34_ddm34, df45_ddm45, df56_ddm56,
     dinflow_dq,
     din_rain1_dq, din_rain2_dq, din_rain3_dq, din_rain4_dq, din_rain5_dq, din_rain6_dq,
     dpc1_dm1, dpc2_dm2, dpc3_dm3, dpc4_dm4, dpc5_dm5, dpc6_dm6,
     dpc1_det0, dpc2_det0, dpc3_det0, dpc4_det0, dpc5_det0, dpc6_det0,
     dfl1_dm1, dfl2_dm2, dfl3_dm3, dfl4_dm4, dfl5_dm5) = pinn_model(m_mid, in_mid, t_mid)

    # ── predicted derivatives via mass balances ─────────────────────────────
    # Updated mass balances
    dm1 = inflow + in_rain1 - f1 - f12 - pc1 - fl1
    dm2 = f1 + f12 - f2 - f23 - pc2 + in_rain2 - fl2
    dm3 = f2 + f23 - f3 - f34 - pc3 + in_rain3 - fl3
    dm4 = f3 + f34 - f4 - f45 - pc4 + in_rain4 - fl4
    dm5 = f4 + f45 - f5 - f56 - pc5 + in_rain5 - fl5
    dm6 = f5 + f56 - f6 - pc6 + in_rain6
    dm_pred = tf.concat([dm1, dm2, dm3, dm4, dm5, dm6], axis=1)

    # ----evaluation across the entire posible input space

    m_rand = tf.random.uniform([1000, 6], minval=0.0, maxval=1.0, dtype=tf.float64)
    input_rand = tf.random.uniform([1000, 3], minval=0.0, maxval=1.0, dtype=tf.float64)
    t_rand = tf.random.uniform([1000, 1], minval=0.0, maxval=1.0, dtype=tf.float64)


    (f1_rand, f2_rand, f3_rand, f4_rand, f5_rand, f6_rand,
     f12_rand, f23_rand, f34_rand, f45_rand, f56_rand,
     inflow_rand,
     in_rain1_rand, in_rain2_rand, in_rain3_rand, in_rain4_rand, in_rain5_rand, in_rain6_rand,
     pc1_rand, pc2_rand, pc3_rand, pc4_rand, pc5_rand, pc6_rand,
     fl1_rand, fl2_rand, fl3_rand, fl4_rand, fl5_rand,
     df1_dm1_rand, df2_dm2_rand, df3_dm3_rand, df4_dm4_rand, df5_dm5_rand, df6_dm6_rand,
     df12_ddm12_rand, df23_ddm23_rand, df34_ddm34_rand, df45_ddm45_rand, df56_ddm56_rand,
     dinflow_dq_rand,
     din_rain1_dq_rand, din_rain2_dq_rand, din_rain3_dq_rand, din_rain4_dq_rand, din_rain5_dq_rand, din_rain6_dq_rand,
     dpc1_dm1_rand, dpc2_dm2_rand, dpc3_dm3_rand, dpc4_dm4_rand, dpc5_dm5_rand, dpc6_dm6_rand,
     dpc1_det0_rand, dpc2_det0_rand, dpc3_det0_rand, dpc4_det0_rand, dpc5_det0_rand, dpc6_det0_rand,
     dfl1_dm1_rand, dfl2_dm2_rand, dfl3_dm3_rand, dfl4_dm4_rand, dfl5_dm5_rand) = pinn_model(m_rand, input_rand, t_rand)

     # j1: Derivative match with diagonal-weighted L2 norm
    derrors = dm_pred - dm_mid  # Shape: [batch_size, 6]
    # Center dm_mid to compute variances
    dm_mid_centered = dm_mid - tf.reduce_mean(dm_mid, axis=0, keepdims=True)
    # Compute variances (diagonal of covariance matrix)
    dvariances = tf.reduce_mean(tf.square(dm_mid_centered), axis=0)  # Shape: [6]
    # Compute inverse standard deviations (1/sqrt(variance))
    dinv_std = tf.math.rsqrt(dvariances)  # Shape: [6]
    # Weight errors by inverse standard deviations
    dweighted_errors = derrors * dinv_std  # Shape: [batch_size, 6]

    j1 = tf.reduce_mean(tf.reduce_sum(tf.square(dweighted_errors), axis=1))

    if stage == 1:
        j2 = tf.constant(0.0, dtype=tf.float64)  # Skip RK4 in Stage 1
    else:
        m_pred = rk4_integration(system_func_pinn, t_tf, m0_np, dt, forced_data=in_mid)
        errors = m_pred - states_tf  # Shape: [batch_size, 6]
        # Center states_tf to compute variances
        states_centered = states_tf - tf.reduce_mean(states_tf, axis=0, keepdims=True)
        # Compute variances (diagonal of covariance matrix)
        variances = tf.reduce_mean(tf.square(states_centered), axis=0)  # Shape: [6]
        # Compute inverse standard deviations (1/sqrt(variance))
        inv_std = tf.math.rsqrt(variances)  # Shape: [6]
        # Weight errors by inverse standard deviations
        weighted_errors = errors * inv_std  # Shape: [batch_size, 6]
        j2 = tf.reduce_mean(tf.reduce_sum(tf.abs(weighted_errors), axis=1))
    # ── j3: positivity constraint ───────────────────────────────────────────
    j3 = (tf.reduce_mean(tf.square(tf.minimum(zero64, f1_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, f2_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, f3_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, f4_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, f5_rand)))     +
          #tf.reduce_mean(tf.square(tf.minimum(zero64, f6_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, inflow_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain1_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain2_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain3_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain4_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain5_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, in_rain6_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc1_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc2_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc3_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc4_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc5_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, pc6_rand)))     +
          tf.reduce_mean(tf.square(tf.minimum(zero64, fl1_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, fl2_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, fl3_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, fl4_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, fl5_rand))) )
    # ── j4: monotonic derivatives must be ≥ 0  (optional) ───────────────────
    j4 = (tf.reduce_mean(tf.square(tf.minimum(zero64, dinflow_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain1_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain2_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain3_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain4_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain5_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, din_rain6_dq_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df1_dm1_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df2_dm2_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df3_dm3_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df4_dm4_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df5_dm5_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df6_dm6_rand)))   +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df12_ddm12_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df23_ddm23_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df34_ddm34_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df45_ddm45_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, df56_ddm56_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc1_dm1_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc2_dm2_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc3_dm3_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc4_dm4_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc5_dm5_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc6_dm6_rand)))  +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc1_det0_rand))) +  # New term
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc2_det0_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc3_det0_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc4_det0_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc5_det0_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dpc6_det0_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dfl1_dm1_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dfl2_dm2_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dfl3_dm3_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dfl4_dm4_rand))) +
          tf.reduce_mean(tf.square(tf.minimum(zero64, dfl5_dm5_rand))) 
        )    # ── j5: zero-flow when potential (or qi) ≈ 0 ────────────────────────────
    j5 = (tf.reduce_mean(tf.square(f1)     * tf.cast(tf.abs(m1) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f2)     * tf.cast(tf.abs(m2) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f3)     * tf.cast(tf.abs(m3) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f4)     * tf.cast(tf.abs(m4) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f5)     * tf.cast(tf.abs(m5) <= eps, tf.float64)) +
          #tf.reduce_mean(tf.square(f6)     * tf.cast(tf.abs(m6) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f12)    * tf.cast(tf.abs(dm12) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f23)    * tf.cast(tf.abs(dm23) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f34)    * tf.cast(tf.abs(dm34) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f45)    * tf.cast(tf.abs(dm45) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(f56)    * tf.cast(tf.abs(dm56) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(inflow) * tf.cast(tf.abs(qi) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain1) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain2) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain3) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain4) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain5) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(in_rain6) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(pc1)    * tf.cast((tf.abs(m1) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc2)    * tf.cast((tf.abs(m2) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc3)    * tf.cast((tf.abs(m3) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc4)    * tf.cast((tf.abs(m4) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc5)    * tf.cast((tf.abs(m5) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc6)    * tf.cast((tf.abs(m6) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc1)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc2)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc3)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc4)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc5)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(pc6)    * tf.cast((tf.abs(et0) <= eps), tf.float64)) +
          tf.reduce_mean(tf.square(fl1)     * tf.cast(tf.abs(m1) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(fl2)     * tf.cast(tf.abs(m2) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(fl3)     * tf.cast(tf.abs(m3) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(fl4)     * tf.cast(tf.abs(m4) <= eps, tf.float64)) +
          tf.reduce_mean(tf.square(fl5)     * tf.cast(tf.abs(m5) <= eps, tf.float64)) 
    )

    # ── weighted total loss ─────────────────────────────────────────────────
    total_loss = (lambda_1 * j1 + lambda_2 * j2 + lambda_3 * j3 +
                  lambda_4 * j4 + lambda_5 * j5)

    return (total_loss,
            lambda_1 * j1, lambda_2 * j2, lambda_3 * j3,
            lambda_4 * j4, lambda_5 * j5)

class TwoStageExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr_stage1, initial_lr_stage2, decay_steps, decay_rate, epochs_stage1):
        super(TwoStageExponentialDecay, self).__init__()
        self.initial_lr_stage1 = tf.cast(initial_lr_stage1, tf.float64)
        self.initial_lr_stage2 = tf.cast(initial_lr_stage2, tf.float64)
        self.decay_steps = tf.cast(decay_steps, tf.int64)  # Changed to int64
        self.decay_rate = tf.cast(decay_rate, tf.float64)
        self.epochs_stage1 = tf.cast(epochs_stage1, tf.int64)

    def __call__(self, step):
        step = tf.cast(step, tf.int64)
        is_stage1 = tf.less(step, self.epochs_stage1)
        exponent_stage1 = tf.cast(step // self.decay_steps, tf.float64)
        lr_stage1 = self.initial_lr_stage1 * tf.pow(self.decay_rate, exponent_stage1)
        stage2_step = step - self.epochs_stage1
        exponent_stage2 = tf.cast(stage2_step // self.decay_steps, tf.float64)
        lr_stage2 = self.initial_lr_stage2 * tf.pow(self.decay_rate, exponent_stage2)
        return tf.where(is_stage1, lr_stage1, lr_stage2)

    def get_config(self):
        return {
            "initial_lr_stage1": self.initial_lr_stage1.numpy(),
            "initial_lr_stage2": self.initial_lr_stage2.numpy(),
            "decay_steps": self.decay_steps.numpy(),
            "decay_rate": self.decay_rate.numpy(),
            "epochs_stage1": self.epochs_stage1.numpy()
        }

# Training parameters
epochs_stage1 = 3000  # Number of epochs for Stage 1
epochs_stage2 = 1000   # Number of epochs for Stage 2
total_epochs = epochs_stage1 + epochs_stage2

# Initialize the learning rate schedule
lr_schedule = TwoStageExponentialDecay(
    initial_lr_stage1=7.970521877412297e-05,  # Starting learning rate for Stage 1
    initial_lr_stage2=5e-6,  # Starting learning rate for Stage 2 # 1e-6 total loss 2.6755
    decay_steps=1000,            # Decay every 50 epochs
    decay_rate=0.9,            # Decay factor
    epochs_stage1=epochs_stage1
)

# Define the optimizer with the custom schedule
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    amsgrad=True,
    clipvalue=0.5
)

# Training step function (assumes compute_loss is defined elsewhere in your code)
@tf.function
def train_step(stage):
    with tf.GradientTape() as tape:
        total_loss, j1, j2, j3, j4, j5 = compute_loss(stage)
    grads = tape.gradient(total_loss, pinn_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
    return total_loss, j1, j2, j3, j4, j5

# Warm-up step to compile the function
_ = train_step(1)
print("Warm-up step OK.")

# Loss history tracking
loss_history = []
j1_history = []
j2_history = []
j3_history = []
j4_history = []
j5_history = []

# Timing lists for each stage
stage1_durations = []
stage2_durations = []

# Training loop
for epoch in range(total_epochs):
    if epoch < epochs_stage1:
        stage = 1  # Stage 1: Focus on derivative loss
    else:
        stage = 2  # Stage 2: Focus on trajectory loss

    start_time = time.perf_counter()
    loss_val, j1_val, j2_val, j3_val, j4_val, j5_val = train_step(stage)
    duration = time.perf_counter() - start_time

    # Store duration in the appropriate list
    if stage == 1:
        stage1_durations.append(duration)
    else:
        stage2_durations.append(duration)

    # Store loss values
    loss_history.append(loss_val.numpy())
    j1_history.append(j1_val.numpy())
    j2_history.append(j2_val.numpy())
    j3_history.append(j3_val.numpy())
    j4_history.append(j4_val.numpy())
    j5_history.append(j5_val.numpy())

    # Log progress every 2 epochs
    if epoch % 2 == 0:
        print(f"Epoch {epoch} (Stage {stage}): Total Loss = {loss_val.numpy():.4f}, "
              f"Derivative Loss = {j1_val.numpy():.4f}, Trajectory Loss = {j2_val.numpy():.4f}, "
              f"Positivity = {j3_val.numpy():.4f}, Monotonicity = {j4_val.numpy():.4f}, "
              f"Zero-flow = {j5_val.numpy():.4f}")

# Save timing info to CSV
import csv
with open("train_step_durations.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "stage", "duration_seconds"])
    for i, d in enumerate(stage1_durations):
        writer.writerow([i, 1, d])
    for i, d in enumerate(stage2_durations):
        writer.writerow([i + epochs_stage1, 2, d])

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS_FILE = f"pinn_weights_H{pinn_model.f1_dense1.units}_lr{lr_schedule.initial_lr_stage1.numpy():.2e}.weights.h5"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)
pinn_model.save_weights(WEIGHTS_PATH)
print(f"Weights saved to {WEIGHTS_PATH}")
print("Training step durations saved to train_step_durations.csv")

# After training loop, save all training info for later plotting
import numpy as np
np.savez(
    "training_history_and_timing.npz",
    epoch=np.arange(total_epochs),
    stage=np.array([1 if i < epochs_stage1 else 2 for i in range(total_epochs)]),
    total_loss=np.array(loss_history),
    j1=np.array(j1_history),
    j2=np.array(j2_history),
    j3=np.array(j3_history),
    j4=np.array(j4_history),
    j5=np.array(j5_history),
    duration_seconds=np.array(stage1_durations + stage2_durations)
)
print("Training history and timing saved to training_history_and_timing.npz")

# Print the first and last date of data used for training
print("First training data timestamp:", df['timestamp'].min())
print("Last training data timestamp:", df['timestamp'].max())

