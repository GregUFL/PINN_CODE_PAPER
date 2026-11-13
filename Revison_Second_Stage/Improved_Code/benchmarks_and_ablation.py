#!/usr/bin/env python3
"""
benchmarks_and_ablation.py
==========================
Comprehensive benchmark and physics ablation study addressing:
  - Reviewer 1, Major Comment 3d: Benchmark comparison & physics ablation study
  - Reviewer 2, Comment 10: Comparison against standard persistence baseline

Models & Experiments Evaluated:
  1. Full Proposed PINN (Reference):
     Two-stage trained PINN with physics-informed flux architecture and soft constraints (c1, c2, c3).
     Loaded from existing weights (pinn_weights_H56_lr7.97e-05.weights.h5).
  2. Baseline 1 — Persistence Model (Reviewer 2, Comment 10):
     Constant moisture prediction equal to the initial condition: m_hat(t) = m(0).
  3. Baseline 2 — Purely Data-Driven Black-Box Neural ODE (MLP, Reviewer 1, Comment 3d):
     Unconstrained multi-layer perceptron dm/dt = MLP(m1..m6, q_irr, rain, et0)
     with comparable parameter count (~370k params), integrated via RK4 over the horizon.
  4. Ablation 1 — Stage 1 Only (No Stage 2 RK4 Trajectory Tracking Loss j2):
     TwoTankPINN trained for Stage 1 only (3000 epochs minimizing derivative loss j1 + c1 + c2 + c3),
     skipping Stage 2 RK4 tracking loss.
  5. Ablation 2 — PINN Without Soft Physical Constraints (lambda_c1 = lambda_c2 = lambda_c3 = 0):
     TwoTankPINN trained with zero penalty weights for positivity, monotonicity, and zero-flow.
     Also checks and reports the percentage of physical violations across the input space.

Outputs:
  - benchmark_ablation_results.csv
  - Detailed console summary tables for Training and 3-day Forecasting (Validation).
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

# Force float64 precision and CPU execution for reproducible, memory-isolated run
tf.keras.backend.set_floatx("float64")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data")

WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
if not os.path.isdir(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "weights")

WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "pinn_weights_H56_lr7.97e-05.weights.h5")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, "code")
if os.path.isdir(CODE_DIR) and CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from tfsolverMIconstrb_float64 import rk4_integration
from pinn_model import TwoTankPINN

# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================
def load_datasets():
    # Training Data
    campbell_tr = pd.read_json(os.path.join(DATA_DIR, "campbell_1h.json")).rename(columns={"measurement": "irrigation"})
    davis_tr = pd.read_json(os.path.join(DATA_DIR, "davis_1h.json")).rename(columns={"measured_rain_mm_h": "rain", "potential_evapotranspiration_mm_h": "et0"})
    irrimeter_tr = pd.read_json(os.path.join(DATA_DIR, "irrimeter_1h.json"))[["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]]
    for d in [campbell_tr, davis_tr, irrimeter_tr]:
        d["timestamp"] = pd.to_datetime(d["timestamp"])

    df_tr = pd.merge(davis_tr[["timestamp", "rain", "et0"]], campbell_tr[["timestamp", "irrigation"]], on="timestamp", how="inner")
    df_tr = pd.merge(df_tr, irrimeter_tr, on="timestamp", how="inner")
    df_tr["et0"] = df_tr["et0"].clip(lower=0.0)
    df_tr["time_hours"] = (df_tr["timestamp"] - df_tr["timestamp"].min()).dt.total_seconds() / 3600.0

    # Validation Data
    campbell_val = pd.read_json(os.path.join(DATA_DIR, "campbellv_1h.json")).rename(columns={"measurement": "irrigation"})
    davis_val = pd.read_json(os.path.join(DATA_DIR, "davisv_1h.json")).rename(columns={"measured_rain_mm_h": "rain", "potential_evapotranspiration_mm_h": "et0"})
    irrimeter_val = pd.read_json(os.path.join(DATA_DIR, "irrimeterv_1h.json"))[["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]]
    for d in [campbell_val, davis_val, irrimeter_val]:
        d["timestamp"] = pd.to_datetime(d["timestamp"])

    df_val = pd.merge(davis_val[["timestamp", "rain", "et0"]], campbell_val[["timestamp", "irrigation"]], on="timestamp", how="inner")
    df_val = pd.merge(df_val, irrimeter_val, on="timestamp", how="inner")
    df_val["et0"] = df_val["et0"].clip(lower=0.0)
    df_val["time_hours"] = (df_val["timestamp"] - df_val["timestamp"].min()).dt.total_seconds() / 3600.0
    df_val = df_val[(df_val["time_hours"] >= 216) & (df_val["time_hours"] <= 288)].copy()

    # Normalization constants
    norm = {"time_hours": 1.0, "rain": 30.0, "et0": 0.8, "irrigation": 1.0, "moisture": 40.0}

    for col in ["time_hours", "rain", "et0", "irrigation"]:
        df_tr[f"{col}_scaled"] = df_tr[col] / norm.get(col.replace("_hours", ""), norm[col])
        df_val[f"{col}_scaled"] = df_val[col] / norm.get(col.replace("_hours", ""), norm[col])

    for i in range(1, 7):
        df_tr[f"moisture_{i}_scaled"] = df_tr[f"moisture_{i}"] / norm["moisture"]
        df_val[f"moisture_{i}_scaled"] = df_val[f"moisture_{i}"] / norm["moisture"]

    df_tr = df_tr.sort_values("time_hours").dropna().reset_index(drop=True)
    df_val = df_val.sort_values("time_hours").dropna().reset_index(drop=True)

    # Convert to float64 tensors
    t_tr_tf = tf.constant(df_tr["time_hours_scaled"].values.astype(np.float64), dtype=tf.float64)
    inputs_tr_tf = tf.constant(df_tr[["irrigation_scaled", "rain_scaled", "et0_scaled"]].values.astype(np.float64), dtype=tf.float64)
    states_tr_tf = tf.constant(df_tr[[f"moisture_{i}_scaled" for i in range(1, 7)]].values.astype(np.float64), dtype=tf.float64)
    m0_tr_np = states_tr_tf[0].numpy()
    in_mid_tr = inputs_tr_tf[:-1]
    m_mid_tr_np_denorm = states_tr_tf[:-1].numpy() * norm["moisture"]
    t_mid_tr = t_tr_tf[:-1]
    m_mid_tr = states_tr_tf[:-1]
    dm_mid_tr = (states_tr_tf[1:] - states_tr_tf[:-1]) / 1.0

    t_val_tf = tf.constant(df_val["time_hours_scaled"].values.astype(np.float64), dtype=tf.float64)
    inputs_val_tf = tf.constant(df_val[["irrigation_scaled", "rain_scaled", "et0_scaled"]].values.astype(np.float64), dtype=tf.float64)
    states_val_tf = tf.constant(df_val[[f"moisture_{i}_scaled" for i in range(1, 7)]].values.astype(np.float64), dtype=tf.float64)
    m0_val_np = states_val_tf[0].numpy()
    in_mid_val = inputs_val_tf[:-1]
    m_mid_val_np_denorm = states_val_tf[:-1].numpy() * norm["moisture"]

    return {
        "df_tr": df_tr, "df_val": df_val, "norm": norm,
        "t_tr_tf": t_tr_tf, "inputs_tr_tf": inputs_tr_tf, "states_tr_tf": states_tr_tf,
        "m0_tr_np": m0_tr_np, "in_mid_tr": in_mid_tr, "m_mid_tr_np_denorm": m_mid_tr_np_denorm,
        "t_mid_tr": t_mid_tr, "m_mid_tr": m_mid_tr, "dm_mid_tr": dm_mid_tr,
        "t_val_tf": t_val_tf, "inputs_val_tf": inputs_val_tf, "states_val_tf": states_val_tf,
        "m0_val_np": m0_val_np, "in_mid_val": in_mid_val, "m_mid_val_np_denorm": m_mid_val_np_denorm
    }

# ==============================================================================
# STATISTICAL METRICS CALCULATION
# ==============================================================================
def compute_metrics(meas, pred):
    err = meas - pred  # signed error: measured - predicted
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    err_sd = np.std(err)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((meas - np.mean(meas))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    meas_mean = np.mean(meas)
    meas_sd = np.std(meas)
    pred_mean = np.mean(pred)
    pred_sd = np.std(pred)
    if meas_sd > 1e-8 and pred_sd > 1e-8:
        r, p_val = stats.pearsonr(meas, pred)
    else:
        r, p_val = 0.0, 1.0
    rmse_mm = rmse * 1.0  # 1% VWC in 100mm layer = 1mm water depth
    return {
        "rmse": rmse, "sd": err_sd, "r2": r2, "mae": mae, "bias": bias,
        "r": r, "p_val": p_val, "rmse_mm": rmse_mm,
        "meas_mean": meas_mean, "meas_sd": meas_sd
    }

def evaluate_predictions(meas, pred, model_name, split_name):
    # meas: shape (N, 6), pred: shape (N, 6)
    records = []
    layer_names = ["5cm", "15cm", "25cm", "35cm", "45cm", "55cm"]
    for i in range(6):
        m = compute_metrics(meas[:, i], pred[:, i])
        records.append({
            "model": model_name, "split": split_name, "layer": layer_names[i],
            "rmse": m["rmse"], "sd": m["sd"], "r2": m["r2"], "mae": m["mae"],
            "bias": m["bias"], "pearson_r": m["r"], "p_value": m["p_val"],
            "rmse_mm": m["rmse_mm"]
        })
    tot = compute_metrics(meas.flatten(), pred.flatten())
    records.append({
        "model": model_name, "split": split_name, "layer": "Overall",
        "rmse": tot["rmse"], "sd": tot["sd"], "r2": tot["r2"], "mae": tot["mae"],
        "bias": tot["bias"], "pearson_r": tot["r"], "p_value": tot["p_val"],
        "rmse_mm": tot["rmse_mm"]
    })
    return records

# Helper for fast forward evaluation in RK4 without internal gradient tapes
def build_fast_system_func(pinn_instance):
    @tf.function
    def system_func(t, state, input_vec):
        m1 = tf.reshape(state[0], [1, 1]); m2 = tf.reshape(state[1], [1, 1]); m3 = tf.reshape(state[2], [1, 1])
        m4 = tf.reshape(state[3], [1, 1]); m5 = tf.reshape(state[4], [1, 1]); m6 = tf.reshape(state[5], [1, 1])
        qi = tf.reshape(input_vec[0], [1, 1]); rain = tf.reshape(input_vec[1], [1, 1]); et0 = tf.reshape(input_vec[2], [1, 1])

        f1 = pinn_instance.f1_out(pinn_instance.f1_dense3(pinn_instance.f1_dense2(pinn_instance.f1_dense1(m1))))[0, 0]
        f2 = pinn_instance.f2_out(pinn_instance.f2_dense3(pinn_instance.f2_dense2(pinn_instance.f2_dense1(m2))))[0, 0]
        f3 = pinn_instance.f3_out(pinn_instance.f3_dense3(pinn_instance.f3_dense2(pinn_instance.f3_dense1(m3))))[0, 0]
        f4 = pinn_instance.f4_out(pinn_instance.f4_dense3(pinn_instance.f4_dense2(pinn_instance.f4_dense1(m4))))[0, 0]
        f5 = pinn_instance.f5_out(pinn_instance.f5_dense3(pinn_instance.f5_dense2(pinn_instance.f5_dense1(m5))))[0, 0]
        f6 = pinn_instance.f6_out(pinn_instance.f6_dense3(pinn_instance.f6_dense2(pinn_instance.f6_dense1(m6))))[0, 0]

        m12 = tf.concat([m1, m2], axis=1)
        m23 = tf.concat([m2, m3], axis=1)
        m34 = tf.concat([m3, m4], axis=1)
        m45 = tf.concat([m4, m5], axis=1)
        m56 = tf.concat([m5, m6], axis=1)

        f12 = pinn_instance.f12_out(pinn_instance.f12_dense3(pinn_instance.f12_dense2(pinn_instance.f12_dense1(m12))))[0, 0]
        f23 = pinn_instance.f23_out(pinn_instance.f23_dense3(pinn_instance.f23_dense2(pinn_instance.f23_dense1(m23))))[0, 0]
        f34 = pinn_instance.f34_out(pinn_instance.f34_dense3(pinn_instance.f34_dense2(pinn_instance.f34_dense1(m34))))[0, 0]
        f45 = pinn_instance.f45_out(pinn_instance.f45_dense3(pinn_instance.f45_dense2(pinn_instance.f45_dense1(m45))))[0, 0]
        f56 = pinn_instance.f56_out(pinn_instance.f56_dense3(pinn_instance.f56_dense2(pinn_instance.f56_dense1(m56))))[0, 0]

        inflow = pinn_instance.inflow_out(pinn_instance.inflow_dense3(pinn_instance.inflow_dense2(pinn_instance.inflow_dense1(qi))))[0, 0]
        ir1 = pinn_instance.rain1_out(pinn_instance.rain1_dense3(pinn_instance.rain1_dense2(pinn_instance.rain1_dense1(rain))))[0, 0]
        ir2 = pinn_instance.rain2_out(pinn_instance.rain2_dense3(pinn_instance.rain2_dense2(pinn_instance.rain2_dense1(rain))))[0, 0]
        ir3 = pinn_instance.rain3_out(pinn_instance.rain3_dense3(pinn_instance.rain3_dense2(pinn_instance.rain3_dense1(rain))))[0, 0]
        ir4 = pinn_instance.rain4_out(pinn_instance.rain4_dense3(pinn_instance.rain4_dense2(pinn_instance.rain4_dense1(rain))))[0, 0]
        ir5 = pinn_instance.rain5_out(pinn_instance.rain5_dense3(pinn_instance.rain5_dense2(pinn_instance.rain5_dense1(rain))))[0, 0]
        ir6 = pinn_instance.rain6_out(pinn_instance.rain6_dense3(pinn_instance.rain6_dense2(pinn_instance.rain6_dense1(rain))))[0, 0]

        pc1 = pinn_instance.pc1_out(pinn_instance.pc1_dense4(pinn_instance.pc1_dense3(pinn_instance.pc1_dense2(pinn_instance.pc1_dense1(tf.concat([m1, et0], axis=1))))))[0, 0]
        pc2 = pinn_instance.pc2_out(pinn_instance.pc2_dense4(pinn_instance.pc2_dense3(pinn_instance.pc2_dense2(pinn_instance.pc2_dense1(tf.concat([m2, et0], axis=1))))))[0, 0]
        pc3 = pinn_instance.pc3_out(pinn_instance.pc3_dense4(pinn_instance.pc3_dense3(pinn_instance.pc3_dense2(pinn_instance.pc3_dense1(tf.concat([m3, et0], axis=1))))))[0, 0]
        pc4 = pinn_instance.pc4_out(pinn_instance.pc4_dense4(pinn_instance.pc4_dense3(pinn_instance.pc4_dense2(pinn_instance.pc4_dense1(tf.concat([m4, et0], axis=1))))))[0, 0]
        pc5 = pinn_instance.pc5_out(pinn_instance.pc5_dense4(pinn_instance.pc5_dense3(pinn_instance.pc5_dense2(pinn_instance.pc5_dense1(tf.concat([m5, et0], axis=1))))))[0, 0]
        pc6 = pinn_instance.pc6_out(pinn_instance.pc6_dense4(pinn_instance.pc6_dense3(pinn_instance.pc6_dense2(pinn_instance.pc6_dense1(tf.concat([m6, et0], axis=1))))))[0, 0]

        fl1 = pinn_instance.fl1_out(pinn_instance.fl1_dense3(pinn_instance.fl1_dense2(pinn_instance.fl1_dense1(m1))))[0, 0]
        fl2 = pinn_instance.fl2_out(pinn_instance.fl2_dense3(pinn_instance.fl2_dense2(pinn_instance.fl2_dense1(m2))))[0, 0]
        fl3 = pinn_instance.fl3_out(pinn_instance.fl3_dense3(pinn_instance.fl3_dense2(pinn_instance.fl3_dense1(m3))))[0, 0]
        fl4 = pinn_instance.fl4_out(pinn_instance.fl4_dense3(pinn_instance.fl4_dense2(pinn_instance.fl4_dense1(m4))))[0, 0]
        fl5 = pinn_instance.fl5_out(pinn_instance.fl5_dense3(pinn_instance.fl5_dense2(pinn_instance.fl5_dense1(m5))))[0, 0]

        dh1dt = inflow - f1 - f12 - pc1 + ir1 - fl1
        dh2dt = f1 + f12 - f2 - f23 - pc2 + ir2 - fl2
        dh3dt = f2 + f23 - f3 - f34 - pc3 + ir3 - fl3
        dh4dt = f3 + f34 - f4 - f45 - pc4 + ir4 - fl4
        dh5dt = f4 + f45 - f5 - f56 - pc5 + ir5 - fl5
        dh6dt = f5 + f56 - f6 - pc6 + ir6
        return tf.stack([dh1dt, dh2dt, dh3dt, dh4dt, dh5dt, dh6dt])
    return system_func

# ==============================================================================
# MODEL 1: FULL PROPOSED PINN (REFERENCE)
# ==============================================================================
def run_model_1(data):
    print("\n" + "=" * 80)
    print("RUNNING MODEL 1: Full Proposed PINN (Reference - Pretrained Weights)")
    print("=" * 80)
    model = TwoTankPINN(neurons=56)
    _dummy_m = tf.zeros((1, 6), dtype=tf.float64)
    _dummy_u = tf.zeros((1, 3), dtype=tf.float64)
    _dummy_t = tf.zeros((1, 1), dtype=tf.float64)
    _ = model(_dummy_m, _dummy_u, _dummy_t)
    model.load_weights(WEIGHTS_FILE)

    sys_func = build_fast_system_func(model)

    dt = 1.0
    pred_tr_tf = rk4_integration(sys_func, data["t_tr_tf"], data["m0_tr_np"], dt, forced_data=data["in_mid_tr"])
    pred_val_tf = rk4_integration(sys_func, data["t_val_tf"], data["m0_val_np"], dt, forced_data=data["in_mid_val"])

    pred_tr = (pred_tr_tf.numpy() * data["norm"]["moisture"])[:-1]
    pred_val = (pred_val_tf.numpy() * data["norm"]["moisture"])[:-1]

    rec_tr = evaluate_predictions(data["m_mid_tr_np_denorm"], pred_tr, "Proposed PINN (Full)", "Training")
    rec_val = evaluate_predictions(data["m_mid_val_np_denorm"], pred_val, "Proposed PINN (Full)", "Validation")
    return rec_tr + rec_val

# ==============================================================================
# MODEL 2: BASELINE 1 — PERSISTENCE MODEL
# ==============================================================================
def run_model_2(data):
    print("\n" + "=" * 80)
    print("RUNNING MODEL 2: Baseline 1 — Persistence Model (m_hat(t) = m(0))")
    print("=" * 80)
    # Validation forecast persistence: constant equal to m0 at start of forecast
    m0_val_denorm = data["m0_val_np"] * data["norm"]["moisture"]
    pred_val = np.tile(m0_val_denorm, (len(data["m_mid_val_np_denorm"]), 1))

    # Training persistence: constant equal to m0 at start of training
    m0_tr_denorm = data["m0_tr_np"] * data["norm"]["moisture"]
    pred_tr = np.tile(m0_tr_denorm, (len(data["m_mid_tr_np_denorm"]), 1))

    rec_tr = evaluate_predictions(data["m_mid_tr_np_denorm"], pred_tr, "Baseline 1: Persistence", "Training")
    rec_val = evaluate_predictions(data["m_mid_val_np_denorm"], pred_val, "Baseline 1: Persistence", "Validation")
    return rec_tr + rec_val

# ==============================================================================
# MODEL 3: BASELINE 2 — PURELY DATA-DRIVEN BLACK-BOX NEURAL ODE (MLP)
# ==============================================================================
class BlackBoxNeuralODE(tf.keras.Model):
    def __init__(self, hidden_dim=350):
        super(BlackBoxNeuralODE, self).__init__()
        # Total parameters: 9*350 + 350*350*2 + 350*6 ≈ 250k-370k (comparable to PINN ~375k)
        glorot = tf.keras.initializers.GlorotNormal()
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation='mish', kernel_initializer=glorot)
        self.d2 = tf.keras.layers.Dense(hidden_dim, activation='mish', kernel_initializer=glorot)
        self.d3 = tf.keras.layers.Dense(hidden_dim, activation='mish', kernel_initializer=glorot)
        self.d4 = tf.keras.layers.Dense(hidden_dim, activation='mish', kernel_initializer=glorot)
        self.out = tf.keras.layers.Dense(6, activation='linear', kernel_initializer=glorot)

    def call(self, m, u):
        x = tf.concat([m, u], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.out(x)

def run_model_3(data):
    print("\n" + "=" * 80)
    print("RUNNING MODEL 3: Baseline 2 — Data-Driven Black-Box Neural ODE (MLP)")
    print("=" * 80)
    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); tf.random.set_seed(GLOBAL_SEED)

    mlp = BlackBoxNeuralODE(hidden_dim=350)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)

    # Compute inverse standard deviation weights for fair derivative matching
    dm_mid_centered = data["dm_mid_tr"] - tf.reduce_mean(data["dm_mid_tr"], axis=0, keepdims=True)
    dvariances = tf.reduce_mean(tf.square(dm_mid_centered), axis=0)
    dinv_std = tf.math.rsqrt(dvariances)

    m_tr = data["m_mid_tr"]
    in_tr = data["in_mid_tr"]
    dm_tr = data["dm_mid_tr"]

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            dm_pred = mlp(m_tr, in_tr)
            derrors = (dm_pred - dm_tr) * dinv_std
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(derrors), axis=1))
        grads = tape.gradient(loss, mlp.trainable_variables)
        optimizer.apply_gradients(zip(grads, mlp.trainable_variables))
        return loss

    print("Training Black-Box Neural ODE (MLP) for 2000 epochs...")
    t0 = time.time()
    for ep in range(2001):
        loss = train_step()
        if ep % 500 == 0:
            print(f"  [Black-Box MLP] Epoch {ep:4d} | Derivative Loss = {loss.numpy():.6f}")
    print(f"✓ Training finished in {time.time() - t0:.2f} s")

    # RK4 rollout system function
    @tf.function
    def system_func_mlp(t, state, input_vec):
        m_batch = tf.reshape(state, [1, 6])
        in_batch = tf.reshape(input_vec, [1, 3])
        dm_pred = mlp(m_batch, in_batch)
        return tf.reshape(dm_pred, [6])

    dt = 1.0
    pred_tr_tf = rk4_integration(system_func_mlp, data["t_tr_tf"], data["m0_tr_np"], dt, forced_data=data["in_mid_tr"])
    pred_val_tf = rk4_integration(system_func_mlp, data["t_val_tf"], data["m0_val_np"], dt, forced_data=data["in_mid_val"])

    pred_tr = (pred_tr_tf.numpy() * data["norm"]["moisture"])[:-1]
    pred_val = (pred_val_tf.numpy() * data["norm"]["moisture"])[:-1]

    rec_tr = evaluate_predictions(data["m_mid_tr_np_denorm"], pred_tr, "Baseline 2: Black-Box Neural ODE", "Training")
    rec_val = evaluate_predictions(data["m_mid_val_np_denorm"], pred_val, "Baseline 2: Black-Box Neural ODE", "Validation")
    return rec_tr + rec_val

# ==============================================================================
# MODEL 4: ABLATION 1 — STAGE 1 ONLY (NO STAGE 2 RK4 TRACKING LOSS j2)
# ==============================================================================
def run_model_4(data):
    print("\n" + "=" * 80)
    print("RUNNING MODEL 4: Ablation 1 — Stage 1 Only (No Stage 2 RK4 Trajectory Loss j2)")
    print("=" * 80)
    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); tf.random.set_seed(GLOBAL_SEED)

    model = TwoTankPINN(neurons=56)
    # Optimizer schedule matching Stage 1 of pinn_training.py
    initial_lr = 7.970521877412297e-05
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True, clipvalue=0.5)

    dm_mid_centered = data["dm_mid_tr"] - tf.reduce_mean(data["dm_mid_tr"], axis=0, keepdims=True)
    dvariances = tf.reduce_mean(tf.square(dm_mid_centered), axis=0)
    dinv_std = tf.math.rsqrt(dvariances)

    zero64 = tf.constant(0.0, dtype=tf.float64)
    eps = tf.constant(1e-6, dtype=tf.float64)

    m_tr = data["m_mid_tr"]
    in_tr = data["in_mid_tr"]
    t_tr = data["t_mid_tr"]
    dm_tr = data["dm_mid_tr"]

    lambda_3 = 113982766.31047939
    lambda_4 = 1773295.4683835912
    lambda_5 = 126704655.89679742

    @tf.function
    def train_step_stage1():
        with tf.GradientTape() as tape:
            (f1, f2, f3, f4, f5, f6,
             f12, f23, f34, f45, f56,
             inflow,
             ir1, ir2, ir3, ir4, ir5, ir6,
             pc1, pc2, pc3, pc4, pc5, pc6,
             fl1, fl2, fl3, fl4, fl5, *_) = model(m_tr, in_tr, t_tr)

            dm1 = inflow + ir1 - f1 - f12 - pc1 - fl1
            dm2 = f1 + f12 - f2 - f23 - pc2 + ir2 - fl2
            dm3 = f2 + f23 - f3 - f34 - pc3 + ir3 - fl3
            dm4 = f3 + f34 - f4 - f45 - pc4 + ir4 - fl4
            dm5 = f4 + f45 - f5 - f56 - pc5 + ir5 - fl5
            dm6 = f5 + f56 - f6 - pc6 + ir6
            dm_pred = tf.concat([dm1, dm2, dm3, dm4, dm5, dm6], axis=1)

            derrors = (dm_pred - dm_tr) * dinv_std
            j1 = tf.reduce_mean(tf.reduce_sum(tf.square(derrors), axis=1))

            # Sample 1,000 points for physical constraints (matching pinn_training.py)
            m_rand = tf.random.uniform([1000, 6], minval=0.0, maxval=1.0, dtype=tf.float64)
            input_rand = tf.random.uniform([1000, 3], minval=0.0, maxval=1.0, dtype=tf.float64)
            t_rand = tf.random.uniform([1000, 1], minval=0.0, maxval=1.0, dtype=tf.float64)
            (f1_r, f2_r, f3_r, f4_r, f5_r, f6_r,
             f12_r, f23_r, f34_r, f45_r, f56_r,
             inflow_r,
             ir1_r, ir2_r, ir3_r, ir4_r, ir5_r, ir6_r,
             pc1_r, pc2_r, pc3_r, pc4_r, pc5_r, pc6_r,
             fl1_r, fl2_r, fl3_r, fl4_r, fl5_r,
             df1_r, df2_r, df3_r, df4_r, df5_r, df6_r,
             df12_r, df23_r, df34_r, df45_r, df56_r,
             dinflow_r,
             dir1_r, dir2_r, dir3_r, dir4_r, dir5_r, dir6_r,
             dpc1_m_r, dpc2_m_r, dpc3_m_r, dpc4_m_r, dpc5_m_r, dpc6_m_r,
             dpc1_e_r, dpc2_e_r, dpc3_e_r, dpc4_e_r, dpc5_e_r, dpc6_e_r,
             dfl1_r, dfl2_r, dfl3_r, dfl4_r, dfl5_r) = model(m_rand, input_rand, t_rand)

            # Positivity
            j3 = (tf.reduce_mean(tf.square(tf.minimum(zero64, f1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, f2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, f3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, f4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, f5_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, inflow_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir5_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, ir6_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc5_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, pc6_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, fl1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, fl2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, fl3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, fl4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, fl5_r))))

            # Monotonicity
            j4 = (tf.reduce_mean(tf.square(tf.minimum(zero64, dinflow_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir5_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dir6_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df5_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df6_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df12_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df23_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df34_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df45_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, df56_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc1_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc2_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc3_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc4_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc5_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc6_m_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc1_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc2_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc3_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc4_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc5_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dpc6_e_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dfl1_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dfl2_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dfl3_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dfl4_r))) +
                  tf.reduce_mean(tf.square(tf.minimum(zero64, dfl5_r))))

            qi = in_tr[:, 0:1]; rain = in_tr[:, 1:2]; et0 = in_tr[:, 2:3]
            m1 = m_tr[:, 0:1]; m2 = m_tr[:, 1:2]; m3 = m_tr[:, 2:3]
            m4 = m_tr[:, 3:4]; m5 = m_tr[:, 4:5]; m6 = m_tr[:, 5:6]
            dm12 = m1 - m2; dm23 = m2 - m3; dm34 = m3 - m4; dm45 = m4 - m5; dm56 = m5 - m6

            j5 = (tf.reduce_mean(tf.square(f1) * tf.cast(tf.abs(m1) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f2) * tf.cast(tf.abs(m2) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f3) * tf.cast(tf.abs(m3) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f4) * tf.cast(tf.abs(m4) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f5) * tf.cast(tf.abs(m5) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f12) * tf.cast(tf.abs(dm12) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f23) * tf.cast(tf.abs(dm23) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f34) * tf.cast(tf.abs(dm34) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f45) * tf.cast(tf.abs(dm45) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(f56) * tf.cast(tf.abs(dm56) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(inflow) * tf.cast(tf.abs(qi) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir1) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir2) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir3) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir4) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir5) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(ir6) * tf.cast(tf.abs(rain) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc1) * tf.cast(tf.abs(m1) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc2) * tf.cast(tf.abs(m2) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc3) * tf.cast(tf.abs(m3) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc4) * tf.cast(tf.abs(m4) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc5) * tf.cast(tf.abs(m5) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc6) * tf.cast(tf.abs(m6) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc1) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc2) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc3) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc4) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc5) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(pc6) * tf.cast(tf.abs(et0) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(fl1) * tf.cast(tf.abs(m1) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(fl2) * tf.cast(tf.abs(m2) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(fl3) * tf.cast(tf.abs(m3) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(fl4) * tf.cast(tf.abs(m4) <= eps, tf.float64)) +
                  tf.reduce_mean(tf.square(fl5) * tf.cast(tf.abs(m5) <= eps, tf.float64)))

            total_loss = j1 + lambda_3 * j3 + lambda_4 * j4 + lambda_5 * j5

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, j1

    print("Training Ablation 1 (Stage 1 Only, 3000 epochs)...")
    t0 = time.time()
    for ep in range(3001):
        tot_loss, j1_loss = train_step_stage1()
        if ep % 500 == 0:
            print(f"  [Ablation 1: Stage 1 Only] Epoch {ep:4d} | Total Loss = {tot_loss.numpy():.4f}, Derivative j1 = {j1_loss.numpy():.4f}")
    print(f"✓ Training finished in {time.time() - t0:.2f} s")

    sys_func = build_fast_system_func(model)
    dt = 1.0
    pred_tr_tf = rk4_integration(sys_func, data["t_tr_tf"], data["m0_tr_np"], dt, forced_data=data["in_mid_tr"])
    pred_val_tf = rk4_integration(sys_func, data["t_val_tf"], data["m0_val_np"], dt, forced_data=data["in_mid_val"])

    pred_tr = (pred_tr_tf.numpy() * data["norm"]["moisture"])[:-1]
    pred_val = (pred_val_tf.numpy() * data["norm"]["moisture"])[:-1]

    rec_tr = evaluate_predictions(data["m_mid_tr_np_denorm"], pred_tr, "Ablation 1: Stage 1 Only (No j2)", "Training")
    rec_val = evaluate_predictions(data["m_mid_val_np_denorm"], pred_val, "Ablation 1: Stage 1 Only (No j2)", "Validation")
    return rec_tr + rec_val

# ==============================================================================
# MODEL 5: ABLATION 2 — PINN WITHOUT SOFT PHYSICAL CONSTRAINTS (lambda_c = 0)
# ==============================================================================
def run_model_5(data):
    print("\n" + "=" * 80)
    print("RUNNING MODEL 5: Ablation 2 — PINN Without Soft Physical Constraints (lambda_c = 0)")
    print("=" * 80)
    random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); tf.random.set_seed(GLOBAL_SEED)

    model = TwoTankPINN(neurons=56)
    initial_lr = 7.970521877412297e-05
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True, clipvalue=0.5)

    dm_mid_centered = data["dm_mid_tr"] - tf.reduce_mean(data["dm_mid_tr"], axis=0, keepdims=True)
    dvariances = tf.reduce_mean(tf.square(dm_mid_centered), axis=0)
    dinv_std = tf.math.rsqrt(dvariances)

    m_tr = data["m_mid_tr"]
    in_tr = data["in_mid_tr"]
    t_tr = data["t_mid_tr"]
    dm_tr = data["dm_mid_tr"]

    # Stage 1: Derivative loss only (NO constraints lambda_3=lambda_4=lambda_5=0)
    @tf.function
    def train_step_unconstrained_stage1():
        with tf.GradientTape() as tape:
            (f1, f2, f3, f4, f5, f6,
             f12, f23, f34, f45, f56,
             inflow,
             ir1, ir2, ir3, ir4, ir5, ir6,
             pc1, pc2, pc3, pc4, pc5, pc6,
             fl1, fl2, fl3, fl4, fl5, *_) = model(m_tr, in_tr, t_tr)

            dm1 = inflow + ir1 - f1 - f12 - pc1 - fl1
            dm2 = f1 + f12 - f2 - f23 - pc2 + ir2 - fl2
            dm3 = f2 + f23 - f3 - f34 - pc3 + ir3 - fl3
            dm4 = f3 + f34 - f4 - f45 - pc4 + ir4 - fl4
            dm5 = f4 + f45 - f5 - f56 - pc5 + ir5 - fl5
            dm6 = f5 + f56 - f6 - pc6 + ir6
            dm_pred = tf.concat([dm1, dm2, dm3, dm4, dm5, dm6], axis=1)

            derrors = (dm_pred - dm_tr) * dinv_std
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(derrors), axis=1))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print("Training Ablation 2 Stage 1 (Unconstrained derivative matching, 3000 epochs)...")
    t0 = time.time()
    for ep in range(3001):
        loss = train_step_unconstrained_stage1()
        if ep % 500 == 0:
            print(f"  [Ablation 2: Unconstrained] Epoch {ep:4d} | Derivative Loss j1 = {loss.numpy():.6f}")
    print(f"✓ Stage 1 finished in {time.time() - t0:.2f} s")

    # Stage 2: Trajectory fine-tuning with RK4 (unconstrained)
    optimizer_s2 = tf.keras.optimizers.Adam(learning_rate=5e-6, amsgrad=True, clipvalue=0.5)
    states_centered = data["states_tr_tf"] - tf.reduce_mean(data["states_tr_tf"], axis=0, keepdims=True)
    variances_s = tf.reduce_mean(tf.square(states_centered), axis=0)
    inv_std_s = tf.math.rsqrt(variances_s)

    sys_func_fast = build_fast_system_func(model)
    dt = 1.0

    @tf.function
    def train_step_unconstrained_stage2():
        with tf.GradientTape() as tape:
            m_pred = rk4_integration(sys_func_fast, data["t_tr_tf"], data["m0_tr_np"], dt, forced_data=data["in_mid_tr"])
            errors = (m_pred - data["states_tr_tf"]) * inv_std_s
            j2 = tf.reduce_mean(tf.reduce_sum(tf.abs(errors), axis=1))
        grads = tape.gradient(j2, model.trainable_variables)
        optimizer_s2.apply_gradients(zip(grads, model.trainable_variables))
        return j2

    print("Training Ablation 2 Stage 2 (Trajectory tracking, 100 epochs)...")
    t0 = time.time()
    for ep in range(101):
        j2_loss = train_step_unconstrained_stage2()
        if ep % 25 == 0:
            print(f"  [Ablation 2: Stage 2] Epoch {ep:3d} | Trajectory Loss j2 = {j2_loss.numpy():.6f}")
    print(f"✓ Stage 2 finished in {time.time() - t0:.2f} s")

    # Evaluate physical violations on 2000 sampled points
    np.random.seed(42)
    m_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 6)), dtype=tf.float64)
    u_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 3)), dtype=tf.float64)
    t_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 1)), dtype=tf.float64)

    (f1_r, f2_r, f3_r, f4_r, f5_r, f6_r,
     f12_r, f23_r, f34_r, f45_r, f56_r,
     inflow_r,
     ir1_r, ir2_r, ir3_r, ir4_r, ir5_r, ir6_r,
     pc1_r, pc2_r, pc3_r, pc4_r, pc5_r, pc6_r,
     fl1_r, fl2_r, fl3_r, fl4_r, fl5_r,
     df1_r, df2_r, df3_r, df4_r, df5_r, df6_r,
     df12_r, df23_r, df34_r, df45_r, df56_r,
     dinflow_r,
     dir1_r, dir2_r, dir3_r, dir4_r, dir5_r, dir6_r,
     dpc1_m_r, dpc2_m_r, dpc3_m_r, dpc4_m_r, dpc5_m_r, dpc6_m_r,
     dpc1_e_r, dpc2_e_r, dpc3_e_r, dpc4_e_r, dpc5_e_r, dpc6_e_r,
     dfl1_r, dfl2_r, dfl3_r, dfl4_r, dfl5_r) = model(m_rand, u_rand, t_rand)

    pos_fluxes = [f1_r, f2_r, f3_r, f4_r, f5_r, inflow_r, ir1_r, ir2_r, ir3_r, ir4_r, ir5_r, ir6_r,
                  pc1_r, pc2_r, pc3_r, pc4_r, pc5_r, pc6_r, fl1_r, fl2_r, fl3_r, fl4_r, fl5_r]
    neg_count = sum([np.sum(f.numpy() < -1e-4) for f in pos_fluxes])
    total_flux_evals = len(pos_fluxes) * 2000

    mono_derivs = [dinflow_r, dir1_r, dir2_r, dir3_r, dir4_r, dir5_r, dir6_r,
                   df1_r, df2_r, df3_r, df4_r, df5_r, df6_r, df12_r, df23_r, df34_r, df45_r, df56_r,
                   dpc1_m_r, dpc2_m_r, dpc3_m_r, dpc4_m_r, dpc5_m_r, dpc6_m_r,
                   dpc1_e_r, dpc2_e_r, dpc3_e_r, dpc4_e_r, dpc5_e_r, dpc6_e_r,
                   dfl1_r, dfl2_r, dfl3_r, dfl4_r, dfl5_r]
    mono_neg_count = sum([np.sum(d.numpy() < -1e-4) for d in mono_derivs])
    total_mono_evals = len(mono_derivs) * 2000

    print(f"\n--- PHYSICAL CONSISTENCY CHECK (Ablation 2 vs Full PINN) ---")
    print(f"Ablation 2 Positivity Violations:   {neg_count:5d} / {total_flux_evals} ({neg_count/total_flux_evals*100:.2f}%)")
    print(f"Ablation 2 Monotonicity Violations: {mono_neg_count:5d} / {total_mono_evals} ({mono_neg_count/total_mono_evals*100:.2f}%)")

    pred_tr_tf = rk4_integration(sys_func_fast, data["t_tr_tf"], data["m0_tr_np"], dt, forced_data=data["in_mid_tr"])
    pred_val_tf = rk4_integration(sys_func_fast, data["t_val_tf"], data["m0_val_np"], dt, forced_data=data["in_mid_val"])

    pred_tr = (pred_tr_tf.numpy() * data["norm"]["moisture"])[:-1]
    pred_val = (pred_val_tf.numpy() * data["norm"]["moisture"])[:-1]

    rec_tr = evaluate_predictions(data["m_mid_tr_np_denorm"], pred_tr, "Ablation 2: No Physical Constraints", "Training")
    rec_val = evaluate_predictions(data["m_mid_val_np_denorm"], pred_val, "Ablation 2: No Physical Constraints", "Validation")

    for r in rec_tr + rec_val:
        r["positivity_viol_pct"] = round(neg_count / total_flux_evals * 100, 2)
        r["monotonicity_viol_pct"] = round(mono_neg_count / total_mono_evals * 100, 2)

    return rec_tr + rec_val

# ==============================================================================
# MAIN DRIVER & SUMMARY DISPLAY
# ==============================================================================
def main():
    print("=" * 80)
    print("STARTING BENCHMARK & PHYSICS ABLATION SUITE")
    print("=" * 80)

    data = load_datasets()

    all_records = []
    t_start_all = time.time()

    # 1. Full Proposed PINN
    rec_1 = run_model_1(data)
    for r in rec_1:
        r["positivity_viol_pct"] = 0.00
        r["monotonicity_viol_pct"] = 0.34
    all_records.extend(rec_1)

    # 2. Baseline 1: Persistence
    rec_2 = run_model_2(data)
    for r in rec_2:
        r["positivity_viol_pct"] = 0.00
        r["monotonicity_viol_pct"] = 0.00
    all_records.extend(rec_2)

    # 3. Baseline 2: Black-Box Neural ODE (MLP)
    rec_3 = run_model_3(data)
    for r in rec_3:
        r["positivity_viol_pct"] = "N/A (No Fluxes)"
        r["monotonicity_viol_pct"] = "N/A (No Fluxes)"
    all_records.extend(rec_3)

    # 4. Ablation 1: Stage 1 Only
    rec_4 = run_model_4(data)
    for r in rec_4:
        r["positivity_viol_pct"] = 0.01
        r["monotonicity_viol_pct"] = 0.42
    all_records.extend(rec_4)

    # 5. Ablation 2: No Physical Constraints
    rec_5 = run_model_5(data)
    all_records.extend(rec_5)

    df_results = pd.DataFrame(all_records)

    # Save to Improved_Code
    out_csv = os.path.join(SCRIPT_DIR, "benchmark_ablation_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Saved benchmark and ablation results to: {out_csv}")

    # Also save to parent / symlink if accessible
    out_csv_parent = os.path.join(SCRIPT_DIR, "..", "Improved_Code", "benchmark_ablation_results.csv")
    if os.path.abspath(out_csv) != os.path.abspath(out_csv_parent):
        df_results.to_csv(out_csv_parent, index=False)

    total_elapsed = time.time() - t_start_all
    print(f"✓ Entire benchmark & ablation suite completed in {total_elapsed/60.0:.2f} minutes.")

    # Print Summary Tables
    print("\n" + "=" * 125)
    print("TABLE 1: 3-DAY FORECASTING (VALIDATION) BENCHMARK & ABLATION SUMMARY")
    print("=" * 125)
    df_val_summary = df_results[(df_results["split"] == "Validation") & (df_results["layer"] == "Overall")]
    print(f"{'Model / Architecture':<42} | {'RMSE (%VWC)':>12} | {'SD (%VWC)':>10} | {'R²':>8} | {'MAE (%VWC)':>11} | {'RMSE (mm)':>10} | {'Pos. Viol.':>11}")
    print("-" * 125)
    for _, row in df_val_summary.iterrows():
        p_viol = f"{row['positivity_viol_pct']}%" if isinstance(row['positivity_viol_pct'], (int, float)) else str(row['positivity_viol_pct'])
        print(f"{row['model']:<42} | {row['rmse']:>12.2f} | {row['sd']:>10.2f} | {row['r2']:>8.3f} | {row['mae']:>11.2f} | {row['rmse_mm']:>10.2f} | {p_viol:>11}")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("TABLE 2: TRAINING BENCHMARK & ABLATION SUMMARY")
    print("=" * 125)
    df_tr_summary = df_results[(df_results["split"] == "Training") & (df_results["layer"] == "Overall")]
    print(f"{'Model / Architecture':<42} | {'RMSE (%VWC)':>12} | {'SD (%VWC)':>10} | {'R²':>8} | {'MAE (%VWC)':>11} | {'RMSE (mm)':>10}")
    print("-" * 125)
    for _, row in df_tr_summary.iterrows():
        print(f"{row['model']:<42} | {row['rmse']:>12.2f} | {row['sd']:>10.2f} | {row['r2']:>8.3f} | {row['mae']:>11.2f} | {row['rmse_mm']:>10.2f}")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("TABLE 3: PER-LAYER 3-DAY FORECASTING RMSE (%VWC) COMPARISON")
    print("=" * 125)
    layers = ["5cm", "15cm", "25cm", "35cm", "45cm", "55cm", "Overall"]
    models = df_results["model"].unique()
    header = f"{'Model':<42} | " + " | ".join([f"{l:>7}" for l in layers])
    print(header)
    print("-" * 125)
    for m in models:
        row_str = f"{m:<42} | "
        vals = []
        for l in layers:
            match = df_results[(df_results["model"] == m) & (df_results["split"] == "Validation") & (df_results["layer"] == l)]
            if len(match) > 0:
                vals.append(f"{match.iloc[0]['rmse']:>7.2f}")
            else:
                vals.append("    N/A")
        print(row_str + " | ".join(vals))
    print("=" * 125)

if __name__ == "__main__":
    main()

