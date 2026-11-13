#!/usr/bin/env python3
"""
diagnostic_unseen_experiments.py
==================================
Three diagnostic experiments across the full 792-hour unseen post-training dataset
(May 3, 2025 21:00:00 to June 5, 2025 20:00:00) without modifying core code or main outputs:

  1. Rolling 3-Day Window Evaluation & Longer Closed-Loop Horizons (7d, 14d, 33d) across all 792 unseen hours:
     Models:
       - Proposed PINN (Full, 375,089 params, pretrained weights)
       - Persistence Baseline
       - Classical Linear State-Space ODE: dm/dt = A*m + B*u + c0 (60 params, OLS fit)
       - Black-Box Neural ODE (Capacity-Matched: h=350, 374,156 params) — Stage 1 Only
       - Black-Box Neural ODE (Capacity-Matched: h=350, 374,156 params) — Stage 1 + Stage 2 (RK4)
       - Black-Box Neural ODE (Low-Capacity: h=24, 2,190 params) — Stage 1 Only
       - Black-Box Neural ODE (Low-Capacity: h=24, 2,190 params) — Stage 1 + Stage 2 (RK4)

  2. Fair Parameter Count & Stage-2 Comparison:
     Explicitly compares parameter capacity (375k vs 374k vs 2.2k vs 60) and Stage 1 vs Stage 1+Stage 2 RK4 tuning.

  3. Control-Oriented Physical Admissibility Check:
     Computes exact input-output Jacobians d(dm/dt)/du across both the [0,1]^9 operating domain
     and the 792 unseen trajectory points:
       (a) Irrigation monotonicity violations: d(dm_1/dt)/dq_irr < 0 and d(dm_i/dt)/dq_irr < 0
       (b) Evapotranspiration sink violations: d(dm_i/dt)/dET0 > 0
       (c) Zero-state / zero-input equilibrium admissibility: m=0, q_irr=0, r=0 (spontaneous creation or <0 drift)
       (d) Bonus: Rain admissibility (d(dm_i/dt)/dr < 0) & unclipped RK4 negativity counts.

Outputs:
  - diagnostic_unseen_windows.csv
  - diagnostic_unseen_summary.txt
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

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


NORM = {"time_hours": 1.0, "rain": 30.0, "et0": 0.8, "irrigation": 1.0, "moisture": 40.0}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_train_and_full_unseen():
    """Load training (840 rows) and full unseen post-training data (t >= 216, 792 rows)."""
    c_tr = pd.read_json(os.path.join(DATA_DIR, "campbell_1h.json")).rename(columns={"measurement": "irrigation"})
    d_tr = pd.read_json(os.path.join(DATA_DIR, "davis_1h.json")).rename(
        columns={"measured_rain_mm_h": "rain", "potential_evapotranspiration_mm_h": "et0"}
    )
    i_tr = pd.read_json(os.path.join(DATA_DIR, "irrimeter_1h.json"))[
        ["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]
    ]
    for d in (c_tr, d_tr, i_tr):
        d["timestamp"] = pd.to_datetime(d["timestamp"])

    df_tr = pd.merge(d_tr[["timestamp", "rain", "et0"]], c_tr[["timestamp", "irrigation"]], on="timestamp", how="inner")
    df_tr = pd.merge(df_tr, i_tr, on="timestamp", how="inner")
    df_tr["et0"] = df_tr["et0"].clip(lower=0.0)
    df_tr["time_hours"] = (df_tr["timestamp"] - df_tr["timestamp"].min()).dt.total_seconds() / 3600.0
    df_tr = df_tr.sort_values("time_hours").dropna().reset_index(drop=True)

    c_val = pd.read_json(os.path.join(DATA_DIR, "campbellv_1h.json")).rename(columns={"measurement": "irrigation"})
    d_val = pd.read_json(os.path.join(DATA_DIR, "davisv_1h.json")).rename(
        columns={"measured_rain_mm_h": "rain", "potential_evapotranspiration_mm_h": "et0"}
    )
    i_val = pd.read_json(os.path.join(DATA_DIR, "irrimeterv_1h.json"))[
        ["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]
    ]
    for d in (c_val, d_val, i_val):
        d["timestamp"] = pd.to_datetime(d["timestamp"])

    df_val = pd.merge(d_val[["timestamp", "rain", "et0"]], c_val[["timestamp", "irrigation"]], on="timestamp", how="inner")
    df_val = pd.merge(df_val, i_val, on="timestamp", how="inner")
    df_val["et0"] = df_val["et0"].clip(lower=0.0)
    df_val["time_hours"] = (df_val["timestamp"] - df_val["timestamp"].min()).dt.total_seconds() / 3600.0
    # Full unseen dataset starting May 3 21:00:00 (25h after training end) through June 5 20:00:00
    df_unseen = df_val[df_val["time_hours"] >= 216].sort_values("time_hours").dropna().reset_index(drop=True)

    for df in (df_tr, df_unseen):
        df["time_hours_scaled"] = df["time_hours"] / NORM["time_hours"]
        df["rain_scaled"] = df["rain"] / NORM["rain"]
        df["et0_scaled"] = df["et0"] / NORM["et0"]
        df["irrigation_scaled"] = df["irrigation"] / NORM["irrigation"]
        for k in range(1, 7):
            df[f"moisture_{k}_scaled"] = df[f"moisture_{k}"] / NORM["moisture"]

    t_tr_tf = tf.constant(df_tr["time_hours_scaled"].values.astype(np.float64), dtype=tf.float64)
    in_tr_tf = tf.constant(df_tr[["irrigation_scaled", "rain_scaled", "et0_scaled"]].values.astype(np.float64), dtype=tf.float64)
    st_tr_tf = tf.constant(df_tr[[f"moisture_{k}_scaled" for k in range(1, 7)]].values.astype(np.float64), dtype=tf.float64)

    t_un_tf = tf.constant(df_unseen["time_hours_scaled"].values.astype(np.float64), dtype=tf.float64)
    in_un_tf = tf.constant(df_unseen[["irrigation_scaled", "rain_scaled", "et0_scaled"]].values.astype(np.float64), dtype=tf.float64)
    st_un_tf = tf.constant(df_unseen[[f"moisture_{k}_scaled" for k in range(1, 7)]].values.astype(np.float64), dtype=tf.float64)

    return {
        "df_tr": df_tr,
        "df_unseen": df_unseen,
        "t_tr_tf": t_tr_tf,
        "in_tr_tf": in_tr_tf,
        "st_tr_tf": st_tr_tf,
        "m0_tr_np": st_tr_tf[0].numpy(),
        "in_mid_tr": in_tr_tf[:-1],
        "m_mid_tr": st_tr_tf[:-1],
        "dm_mid_tr": st_tr_tf[1:] - st_tr_tf[:-1],
        "t_un_tf": t_un_tf,
        "in_un_tf": in_un_tf,
        "st_un_tf": st_un_tf,
        "max_tr_rain": float(df_tr["rain"].max()),
        "max_tr_et0": float(df_tr["et0"].max()),
    }


def compute_metrics_flat(meas, pred):
    """Compute overall RMSE, MAE, Bias, Error SD, R2 on denormalized arrays."""
    m_f = meas.flatten()
    p_f = pred.flatten()
    err = m_f - p_f
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    sd = float(np.std(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((m_f - np.mean(m_f)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    # Also compute layer-1 (5cm) RMSE
    l1_rmse = float(np.sqrt(np.mean((meas[:, 0] - pred[:, 0]) ** 2)))
    return {"rmse": rmse, "mae": mae, "bias": bias, "sd": sd, "r2": r2, "l1_rmse": l1_rmse}


# ==============================================================================
# FAST UNIFIED RK4 & UNCLIPPED RK4 HELPERS
# ==============================================================================
def build_fast_pinn_rhs(pinn_instance):
    """Build fast forward-only TensorFlow RHS callable (no internal GradientTapes) for TwoTankPINN."""
    def rhs_batch(m, u):
        m1, m2, m3 = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        m4, m5, m6 = m[:, 3:4], m[:, 4:5], m[:, 5:6]
        qi, rain, et0 = u[:, 0:1], u[:, 1:2], u[:, 2:3]

        f1 = pinn_instance.f1_out(pinn_instance.f1_dense3(pinn_instance.f1_dense2(pinn_instance.f1_dense1(m1))))
        f2 = pinn_instance.f2_out(pinn_instance.f2_dense3(pinn_instance.f2_dense2(pinn_instance.f2_dense1(m2))))
        f3 = pinn_instance.f3_out(pinn_instance.f3_dense3(pinn_instance.f3_dense2(pinn_instance.f3_dense1(m3))))
        f4 = pinn_instance.f4_out(pinn_instance.f4_dense3(pinn_instance.f4_dense2(pinn_instance.f4_dense1(m4))))
        f5 = pinn_instance.f5_out(pinn_instance.f5_dense3(pinn_instance.f5_dense2(pinn_instance.f5_dense1(m5))))
        f6 = pinn_instance.f6_out(pinn_instance.f6_dense3(pinn_instance.f6_dense2(pinn_instance.f6_dense1(m6))))

        f12 = pinn_instance.f12_out(pinn_instance.f12_dense3(pinn_instance.f12_dense2(pinn_instance.f12_dense1(tf.concat([m1, m2], axis=1)))))
        f23 = pinn_instance.f23_out(pinn_instance.f23_dense3(pinn_instance.f23_dense2(pinn_instance.f23_dense1(tf.concat([m2, m3], axis=1)))))
        f34 = pinn_instance.f34_out(pinn_instance.f34_dense3(pinn_instance.f34_dense2(pinn_instance.f34_dense1(tf.concat([m3, m4], axis=1)))))
        f45 = pinn_instance.f45_out(pinn_instance.f45_dense3(pinn_instance.f45_dense2(pinn_instance.f45_dense1(tf.concat([m4, m5], axis=1)))))
        f56 = pinn_instance.f56_out(pinn_instance.f56_dense3(pinn_instance.f56_dense2(pinn_instance.f56_dense1(tf.concat([m5, m6], axis=1)))))

        inflow = pinn_instance.inflow_out(pinn_instance.inflow_dense3(pinn_instance.inflow_dense2(pinn_instance.inflow_dense1(qi))))
        ir1 = pinn_instance.rain1_out(pinn_instance.rain1_dense3(pinn_instance.rain1_dense2(pinn_instance.rain1_dense1(rain))))
        ir2 = pinn_instance.rain2_out(pinn_instance.rain2_dense3(pinn_instance.rain2_dense2(pinn_instance.rain2_dense1(rain))))
        ir3 = pinn_instance.rain3_out(pinn_instance.rain3_dense3(pinn_instance.rain3_dense2(pinn_instance.rain3_dense1(rain))))
        ir4 = pinn_instance.rain4_out(pinn_instance.rain4_dense3(pinn_instance.rain4_dense2(pinn_instance.rain4_dense1(rain))))
        ir5 = pinn_instance.rain5_out(pinn_instance.rain5_dense3(pinn_instance.rain5_dense2(pinn_instance.rain5_dense1(rain))))
        ir6 = pinn_instance.rain6_out(pinn_instance.rain6_dense3(pinn_instance.rain6_dense2(pinn_instance.rain6_dense1(rain))))

        pc1 = pinn_instance.pc1_out(pinn_instance.pc1_dense4(pinn_instance.pc1_dense3(pinn_instance.pc1_dense2(pinn_instance.pc1_dense1(tf.concat([m1, et0], axis=1))))))
        pc2 = pinn_instance.pc2_out(pinn_instance.pc2_dense4(pinn_instance.pc2_dense3(pinn_instance.pc2_dense2(pinn_instance.pc2_dense1(tf.concat([m2, et0], axis=1))))))
        pc3 = pinn_instance.pc3_out(pinn_instance.pc3_dense4(pinn_instance.pc3_dense3(pinn_instance.pc3_dense2(pinn_instance.pc3_dense1(tf.concat([m3, et0], axis=1))))))
        pc4 = pinn_instance.pc4_out(pinn_instance.pc4_dense4(pinn_instance.pc4_dense3(pinn_instance.pc4_dense2(pinn_instance.pc4_dense1(tf.concat([m4, et0], axis=1))))))
        pc5 = pinn_instance.pc5_out(pinn_instance.pc5_dense4(pinn_instance.pc5_dense3(pinn_instance.pc5_dense2(pinn_instance.pc5_dense1(tf.concat([m5, et0], axis=1))))))
        pc6 = pinn_instance.pc6_out(pinn_instance.pc6_dense4(pinn_instance.pc6_dense3(pinn_instance.pc6_dense2(pinn_instance.pc6_dense1(tf.concat([m6, et0], axis=1))))))

        fl1 = pinn_instance.fl1_out(pinn_instance.fl1_dense3(pinn_instance.fl1_dense2(pinn_instance.fl1_dense1(m1))))
        fl2 = pinn_instance.fl2_out(pinn_instance.fl2_dense3(pinn_instance.fl2_dense2(pinn_instance.fl2_dense1(m2))))
        fl3 = pinn_instance.fl3_out(pinn_instance.fl3_dense3(pinn_instance.fl3_dense2(pinn_instance.fl3_dense1(m3))))
        fl4 = pinn_instance.fl4_out(pinn_instance.fl4_dense3(pinn_instance.fl4_dense2(pinn_instance.fl4_dense1(m4))))
        fl5 = pinn_instance.fl5_out(pinn_instance.fl5_dense3(pinn_instance.fl5_dense2(pinn_instance.fl5_dense1(m5))))

        dm1 = inflow - f1 - f12 - pc1 + ir1 - fl1
        dm2 = f1 + f12 - f2 - f23 - pc2 + ir2 - fl2
        dm3 = f2 + f23 - f3 - f34 - pc3 + ir3 - fl3
        dm4 = f3 + f34 - f4 - f45 - pc4 + ir4 - fl4
        dm5 = f4 + f45 - f5 - f56 - pc5 + ir5 - fl5
        dm6 = f5 + f56 - f6 - pc6 + ir6
        return tf.concat([dm1, dm2, dm3, dm4, dm5, dm6], axis=1)

    @tf.function
    def system_func(t, state, input_vec):
        dm = rhs_batch(tf.reshape(state, [1, 6]), tf.reshape(input_vec, [1, 3]))
        return tf.reshape(dm, [6])

    return rhs_batch, system_func


@tf.function
def rk4_unclipped_tf(system_func, time_points, initial_state, dt, forced_data):
    """Unclipped RK4 integration to test whether the vector field drives moisture below 0."""
    num_steps = tf.shape(time_points)[0]
    states_ta = tf.TensorArray(dtype=tf.float64, size=num_steps, element_shape=initial_state.shape)
    states_ta = states_ta.write(0, initial_state)

    def loop_body(i, current_state, states_ta):
        t = time_points[i - 1]
        u = forced_data[i - 1]
        k1 = system_func(t, current_state, u)
        k2 = system_func(t + dt / 2.0, current_state + k1 * (dt / 2.0), u)
        k3 = system_func(t + dt / 2.0, current_state + k2 * (dt / 2.0), u)
        k4 = system_func(t + dt, current_state + k3 * dt, u)
        new_state = current_state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
        states_ta = states_ta.write(i, new_state)
        return i + 1, new_state, states_ta

    def loop_cond(i, current_state, states_ta):
        return i < num_steps

    _, _, states_ta = tf.while_loop(loop_cond, loop_body, [tf.constant(1), initial_state, states_ta])
    return states_ta.stack()


# ==============================================================================
# MODELS
# ==============================================================================
class BlackBoxNeuralODE(tf.keras.Model):
    def __init__(self, hidden_dim=350):
        super().__init__()
        glorot = tf.keras.initializers.GlorotNormal()
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation="mish", kernel_initializer=glorot)
        self.d2 = tf.keras.layers.Dense(hidden_dim, activation="mish", kernel_initializer=glorot)
        self.d3 = tf.keras.layers.Dense(hidden_dim, activation="mish", kernel_initializer=glorot)
        self.d4 = tf.keras.layers.Dense(hidden_dim, activation="mish", kernel_initializer=glorot)
        self.out = tf.keras.layers.Dense(6, activation="linear", kernel_initializer=glorot)

    def call(self, m, u):
        x = tf.concat([m, u], axis=1)
        return self.out(self.d4(self.d3(self.d2(self.d1(x)))))


def train_blackbox(data, hidden_dim, stage1_epochs=2000, stage2_epochs=0, stage2_lr=1e-4):
    """Train BlackBoxNeuralODE cleanly with Stage 1 (derivative loss) and optional Stage 2 (RK4 trajectory loss)."""
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)

    mlp = BlackBoxNeuralODE(hidden_dim=hidden_dim)
    _ = mlp(tf.zeros((1, 6), dtype=tf.float64), tf.zeros((1, 3), dtype=tf.float64))
    n_params = mlp.count_params()

    # Stage 1 normalization weights
    dm_mid_tr = data["dm_mid_tr"]
    dm_centered = dm_mid_tr - tf.reduce_mean(dm_mid_tr, axis=0, keepdims=True)
    dinv_std = tf.math.rsqrt(tf.reduce_mean(tf.square(dm_centered), axis=0))

    opt_s1 = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True, clipvalue=0.5)
    m_tr, in_tr = data["m_mid_tr"], data["in_mid_tr"]

    @tf.function
    def step_s1():
        with tf.GradientTape() as tape:
            dm_pred = mlp(m_tr, in_tr)
            derr = (dm_pred - dm_mid_tr) * dinv_std
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(derr), axis=1))
        grads = tape.gradient(loss, mlp.trainable_variables)
        opt_s1.apply_gradients(zip(grads, mlp.trainable_variables))
        return loss

    t0 = time.time()
    for ep in range(stage1_epochs + 1):
        l_s1 = step_s1()
        if ep % 500 == 0:
            print(f"  [BB h={hidden_dim} ({n_params:,} params) S1] Ep {ep:4d} | j1 = {l_s1.numpy():.5f}")
    print(f"  Stage 1 finished in {time.time() - t0:.1f}s")

    # Save Stage 1 weights separately so we can return a copy or run Stage 2 on top
    s1_weights = mlp.get_weights()

    if stage2_epochs == 0:
        return mlp, n_params

    # Stage 2: RK4 trajectory loss j2 matching pinn_training.py formulation
    st_tr_tf = data["st_tr_tf"]
    st_centered = st_tr_tf - tf.reduce_mean(st_tr_tf, axis=0, keepdims=True)
    inv_std = tf.math.rsqrt(tf.reduce_mean(tf.square(st_centered), axis=0))
    t_tr_tf = data["t_tr_tf"]
    m0_tr_np = data["m0_tr_np"]

    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=stage2_lr, decay_steps=max(stage2_epochs // 4, 25), decay_rate=0.9, staircase=True
    )
    opt_s2 = tf.keras.optimizers.Adam(learning_rate=lr_sched, amsgrad=True, clipvalue=0.5)

    @tf.function
    def sys_mlp(t, state, input_vec):
        return tf.reshape(mlp(tf.reshape(state, [1, 6]), tf.reshape(input_vec, [1, 3])), [6])

    @tf.function
    def step_s2():
        with tf.GradientTape() as tape:
            m_pred = rk4_integration(sys_mlp, t_tr_tf, m0_tr_np, 1.0, forced_data=in_tr)
            err = (m_pred - st_tr_tf) * inv_std
            j2 = tf.reduce_mean(tf.reduce_sum(tf.abs(err), axis=1))
        grads = tape.gradient(j2, mlp.trainable_variables)
        opt_s2.apply_gradients(zip(grads, mlp.trainable_variables))
        return j2

    best_j2 = float("inf")
    best_weights = s1_weights
    t0 = time.time()
    print(f"  Running Stage 2 RK4 tracking loss ({stage2_epochs} epochs, init lr={stage2_lr})...")
    for ep in range(stage2_epochs + 1):
        j2_val = float(step_s2().numpy())
        if np.isfinite(j2_val) and j2_val < best_j2:
            best_j2 = j2_val
            best_weights = mlp.get_weights()
        if ep % max(stage2_epochs // 5, 1) == 0:
            print(f"  [BB h={hidden_dim} S2] Ep {ep:4d} | j2 = {j2_val:.5f} (best={best_j2:.5f})")
    mlp.set_weights(best_weights)
    print(f"  Stage 2 finished in {time.time() - t0:.1f}s (restored best j2={best_j2:.5f})")

    # Build fresh S1 model for returning both S1 and S1+S2 seamlessly
    mlp_s1 = BlackBoxNeuralODE(hidden_dim=hidden_dim)
    _ = mlp_s1(tf.zeros((1, 6), dtype=tf.float64), tf.zeros((1, 3), dtype=tf.float64))
    mlp_s1.set_weights(s1_weights)
    return mlp_s1, mlp, n_params


def fit_linear_state_space(data):
    """Fit Classical Linear State-Space ODE: dm/dt = A*m + B*u + c0 via OLS on training derivatives."""
    X = np.hstack([data["m_mid_tr"].numpy(), data["in_mid_tr"].numpy(), np.ones((len(data["m_mid_tr"]), 1))])  # (839, 10)
    Y = data["dm_mid_tr"].numpy()  # (839, 6)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (10, 6)
    W_tf = tf.constant(W, dtype=tf.float64)
    A = W[:6, :].T   # (6, 6)
    B = W[6:9, :].T  # (6, 3)
    c0 = W[9, :]     # (6,)

    @tf.function
    def rhs_batch(m, u):
        ones = tf.ones((tf.shape(m)[0], 1), dtype=tf.float64)
        feats = tf.concat([m, u, ones], axis=1)
        return tf.matmul(feats, W_tf)

    @tf.function
    def sys_func(t, state, input_vec):
        feats = tf.concat([state, input_vec, tf.constant([1.0], dtype=tf.float64)], axis=0)
        return tf.reshape(tf.matmul(tf.reshape(feats, [1, 10]), W_tf), [6])

    return {"rhs_batch": rhs_batch, "sys_func": sys_func, "A": A, "B": B, "c0": c0, "n_params": W.size}


# ==============================================================================
# EXPERIMENT 1 & 2: ROLLING 3-DAY WINDOWS & MULTI-HORIZON ROLLOUTS
# ==============================================================================
def evaluate_unseen_windows_and_horizons(models_dict, data):
    df_u = data["df_unseen"]
    t_un = data["t_un_tf"]
    in_un = data["in_un_tf"]
    st_un = data["st_un_tf"]
    max_tr_rain = data["max_tr_rain"]
    max_tr_et0 = data["max_tr_et0"]

    records = []

    # Helper to run clipped RK4 on slice [s_idx : s_idx + length + 1] (length steps -> length mid-step outputs)
    def rollout_model(model_info, s_idx, n_steps):
        m0_np = st_un[s_idx].numpy()
        if model_info["type"] == "persistence":
            pred_scaled = np.tile(m0_np, (n_steps, 1))
            return pred_scaled * NORM["moisture"]
        t_slice = t_un[s_idx : s_idx + n_steps + 1]
        in_slice = in_un[s_idx : s_idx + n_steps]
        pred_tf = rk4_integration(model_info["sys_func"], t_slice, m0_np, 1.0, forced_data=in_slice)
        return (pred_tf.numpy() * NORM["moisture"])[:-1]

    # --- Part 1A: 30 Sliding 24h-Stride 3-Day Windows (72h each) across May 3 - June 4 ---
    # Also mark which windows belong to the 10 disjoint non-overlapping 72h-stride grid
    print("\nRunning 30 sliding 24h-stride 3-day windows (72h each) across unseen 792h dataset...")
    start_indices = list(range(0, len(df_u) - 72, 24))  # 0, 24, ..., 696 -> 30 windows
    for s_idx in start_indices:
        n_steps = 72
        w_df = df_u.iloc[s_idx : s_idx + n_steps]
        w_start_str = str(w_df["timestamp"].iloc[0])[:16]
        w_max_rain = float(w_df["rain"].max())
        w_max_et0 = float(w_df["et0"].max())
        w_tot_rain = float(w_df["rain"].sum())
        w_tot_irr = float(w_df["irrigation"].sum())
        is_nonoverlap = (s_idx % 72 == 0)
        is_extreme = (w_max_rain > max_tr_rain) or (w_max_et0 > max_tr_et0)

        meas = st_un[s_idx : s_idx + n_steps].numpy() * NORM["moisture"]
        for m_name, m_info in models_dict.items():
            pred = rollout_model(m_info, s_idx, n_steps)
            met = compute_metrics_flat(meas, pred)
            records.append({
                "eval_type": "Sliding_3Day",
                "horizon_hours": 72,
                "start_idx": s_idx,
                "start_time": w_start_str,
                "non_overlapping": is_nonoverlap,
                "extreme_event": is_extreme,
                "max_rain_mm_h": w_max_rain,
                "max_et0_mm_h": w_max_et0,
                "model": m_name,
                "params": m_info["params"],
                **met,
            })

    # --- Part 1B: Continuous Closed-Loop Long Horizons from t=216 (May 3 21:00) ---
    print("Running multi-horizon continuous closed-loop rollouts (3d, 7d, 14d, 33d)...")
    horizons = [
        ("Horizon_3d", 72),
        ("Horizon_7d", 168),
        ("Horizon_14d", 336),
        ("Horizon_33d", len(df_u) - 1),  # 791 hourly steps -> full 792 points
    ]
    for h_label, n_steps in horizons:
        meas = st_un[0:n_steps].numpy() * NORM["moisture"]
        w_df = df_u.iloc[0:n_steps]
        for m_name, m_info in models_dict.items():
            pred = rollout_model(m_info, 0, n_steps)
            met = compute_metrics_flat(meas, pred)
            # Also count unclipped RK4 negativity if ODE model
            neg_steps_pct = 0.0
            if m_info["type"] == "ode":
                t_slice_h = t_un[0 : n_steps + 1]
                in_slice_h = in_un[0 : n_steps]
                unclipped = (rk4_unclipped_tf(m_info["sys_func"], t_slice_h, st_un[0].numpy(), 1.0, in_slice_h).numpy() * NORM["moisture"])[:-1]
                neg_steps_pct = float(100.0 * np.mean(unclipped < 0.0))
            records.append({
                "eval_type": h_label,
                "horizon_hours": n_steps,
                "start_idx": 0,
                "start_time": str(w_df["timestamp"].iloc[0])[:16],
                "non_overlapping": True,
                "extreme_event": (w_df["rain"].max() > max_tr_rain) or (w_df["et0"].max() > max_tr_et0),
                "max_rain_mm_h": float(w_df["rain"].max()),
                "max_et0_mm_h": float(w_df["et0"].max()),
                "model": m_name,
                "params": m_info["params"],
                "unclipped_neg_pct": neg_steps_pct,
                **met,
            })

    return pd.DataFrame(records)


# ==============================================================================
# EXPERIMENT 3: CONTROL-ORIENTED PHYSICAL ADMISSIBILITY CHECK
# ==============================================================================
def check_physical_admissibility(models_dict, data):
    """
    Evaluate Jacobian signs and zero-input/zero-state behavior on:
      1) Domain grid: N=2000 uniform random points across [0,1]^6 x [0,1]^3
      2) Trajectory set: all 792 observed hourly points from df_unseen
    """
    np.random.seed(GLOBAL_SEED)
    m_dom = tf.constant(np.random.uniform(0.0, 1.0, size=(2000, 6)), dtype=tf.float64)
    u_dom = tf.constant(np.random.uniform(0.0, 1.0, size=(2000, 3)), dtype=tf.float64)

    m_traj = data["st_un_tf"]
    u_traj = data["in_un_tf"]

    # Zero-state & zero-input equilibrium probes: (m=0, qi=0, r=0, ET0 in {0, 0.1, 0.4, 0.8} mm/h)
    m_zero = tf.zeros((4, 6), dtype=tf.float64)
    u_zero = tf.constant(
        [
            [0.0, 0.0, 0.0 / NORM["et0"]],
            [0.0, 0.0, 0.1 / NORM["et0"]],
            [0.0, 0.0, 0.4 / NORM["et0"]],
            [0.0, 0.0, 0.8 / NORM["et0"]],
        ],
        dtype=tf.float64,
    )

    adm_rows = []
    scale_dm = NORM["moisture"]  # convert scaled d(m)/dt -> %VWC/h
    scale_u = tf.constant([NORM["irrigation"], NORM["rain"], NORM["et0"]], dtype=tf.float64)

    for m_name, m_info in models_dict.items():
        if m_info["type"] != "ode":
            continue
        rhs_batch = m_info["rhs_batch"]

        def compute_jac_stats(m_pts, u_pts):
            with tf.GradientTape() as tape:
                tape.watch(u_pts)
                dm_scaled = rhs_batch(m_pts, u_pts)
                dm_phys = dm_scaled * scale_dm  # (%VWC/h)
            # batch_jacobian wrt normalized u, then divide by scale_u -> (%VWC/h per physical unit of u)
            jac_norm = tape.batch_jacobian(dm_phys, u_pts)  # (N, 6, 3)
            jac_phys = (jac_norm / scale_u[None, None, :]).numpy()

            dq_l1 = jac_phys[:, 0, 0]       # d(dm1/dt)/dq_irr
            dq_all = jac_phys[:, :, 0]      # d(dmi/dt)/dq_irr
            dr_all = jac_phys[:, :, 1]      # d(dmi/dt)/dr
            det_all = jac_phys[:, :, 2]     # d(dmi/dt)/dET0

            tol = 1e-6
            return {
                "dq_l1_neg_pct": float(100.0 * np.mean(dq_l1 < -tol)),
                "dq_l1_min": float(np.min(dq_l1)),
                "dq_all_neg_pct": float(100.0 * np.mean(dq_all < -tol)),
                "dr_all_neg_pct": float(100.0 * np.mean(dr_all < -tol)),
                "det_pos_pct": float(100.0 * np.mean(det_all > tol)),
                "det_max": float(np.max(det_all)),
            }

        s_dom = compute_jac_stats(m_dom, u_dom)
        s_traj = compute_jac_stats(m_traj, u_traj)

        dm_zero_phys = (rhs_batch(m_zero, u_zero) * scale_dm).numpy()  # (4, 6) in %VWC/h
        dm_eq = dm_zero_phys[0]         # m=0, q=0, r=0, et0=0
        dm_dry_et = dm_zero_phys[1:]    # m=0, q=0, r=0, et0>0

        adm_rows.append({
            "model": m_name,
            "params": m_info["params"],
            # Domain [0,1]^9
            "dom_dq_l1_neg_pct": s_dom["dq_l1_neg_pct"],
            "dom_dq_l1_min": s_dom["dq_l1_min"],
            "dom_dq_all_neg_pct": s_dom["dq_all_neg_pct"],
            "dom_dr_all_neg_pct": s_dom["dr_all_neg_pct"],
            "dom_det_pos_pct": s_dom["det_pos_pct"],
            "dom_det_max": s_dom["det_max"],
            # Observed 792h unseen trajectory
            "traj_dq_l1_neg_pct": s_traj["dq_l1_neg_pct"],
            "traj_dq_all_neg_pct": s_traj["dq_all_neg_pct"],
            "traj_det_pos_pct": s_traj["det_pos_pct"],
            # Zero-state equilibrium m=0, u=0 (%VWC/h)
            "zero_eq_max_abs_dmdt": float(np.max(np.abs(dm_eq))),
            "zero_eq_max_pos_dmdt": float(np.max(dm_eq)),
            "zero_eq_min_neg_dmdt": float(np.min(dm_eq)),
            "dry_et_max_abs_dmdt": float(np.max(np.abs(dm_dry_et))),
            "dry_et_min_neg_dmdt": float(np.min(dm_dry_et)),
        })

    return pd.DataFrame(adm_rows)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    t_all = time.time()
    print("=" * 100)
    print("DIAGNOSTIC EXPERIMENTS ON FULL 792-HOUR UNSEEN DATASET (MAY 3 - JUNE 5, 2025)")
    print("=" * 100)

    data = load_train_and_full_unseen()
    print(f"Loaded Training: {len(data['df_tr'])} rows | Unseen Post-Training: {len(data['df_unseen'])} rows")
    print(f"Training Maxima: Rain={data['max_tr_rain']:.2f} mm/h, ET0={data['max_tr_et0']:.3f} mm/h")
    print(f"Unseen Maxima:   Rain={data['df_unseen']['rain'].max():.2f} mm/h, ET0={data['df_unseen']['et0'].max():.3f} mm/h")

    # 1. Proposed PINN (Full, 375,089 params)
    print("\n[1/5] Loading Proposed PINN (375,089 params)...")
    pinn = TwoTankPINN(neurons=56)
    _ = pinn(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64), tf.zeros((1, 1), tf.float64))
    pinn.load_weights(WEIGHTS_FILE)
    pinn_params = pinn.count_params()
    pinn_rhs_batch, pinn_sys_func = build_fast_pinn_rhs(pinn)

    # 2. Linear State-Space ODE
    print("[2/5] Fitting Classical Linear State-Space ODE (OLS, 60 params)...")
    lin_ss = fit_linear_state_space(data)

    # 3. Black-Box Large (~374k params): Stage 1 + Stage 2 (150 ep)
    print("[3/5] Training Capacity-Matched Black-Box Neural ODE (h=350, ~374k params)...")
    bb350_s1, bb350_s2, p350 = train_blackbox(
        data, hidden_dim=350, stage1_epochs=2000, stage2_epochs=150, stage2_lr=2e-5
    )

    # 4. Black-Box Small (~2.2k params): Stage 1 + Stage 2 (500 ep)
    print("[4/5] Training Low-Capacity Black-Box Neural ODE (h=24, ~2.2k params)...")
    bb24_s1, bb24_s2, p24 = train_blackbox(
        data, hidden_dim=24, stage1_epochs=2000, stage2_epochs=500, stage2_lr=1e-4
    )

    # Wrap callables cleanly
    def wrap_mlp(mlp_inst, n_params):
        @tf.function
        def rhs_batch(m, u):
            return mlp_inst(m, u)

        @tf.function
        def sys_func(t, state, input_vec):
            return tf.reshape(mlp_inst(tf.reshape(state, [1, 6]), tf.reshape(input_vec, [1, 3])), [6])

        def rhs_np(s, u):
            return mlp_inst(s.reshape(1, 6), u.reshape(1, 3)).numpy()[0]

        return {"type": "ode", "params": n_params, "rhs_batch": rhs_batch, "sys_func": sys_func, "rhs_np": rhs_np}

    def pinn_rhs_np(s, u):
        return pinn_rhs_batch(s.reshape(1, 6), u.reshape(1, 3)).numpy()[0]

    def lin_rhs_np(s, u):
        return (lin_ss["A"] @ s + lin_ss["B"] @ u + lin_ss["c0"])

    models_dict = {
        "Proposed PINN (Full)": {
            "type": "ode", "params": pinn_params,
            "rhs_batch": pinn_rhs_batch, "sys_func": pinn_sys_func, "rhs_np": pinn_rhs_np,
        },
        "Persistence": {"type": "persistence", "params": 0},
        "Linear State-Space ODE": {
            "type": "ode", "params": lin_ss["n_params"],
            "rhs_batch": lin_ss["rhs_batch"], "sys_func": lin_ss["sys_func"], "rhs_np": lin_rhs_np,
        },
        "BlackBox-374k (Stage 1)": wrap_mlp(bb350_s1, p350),
        "BlackBox-374k (Stage 1+2)": wrap_mlp(bb350_s2, p350),
        "BlackBox-2.2k (Stage 1)": wrap_mlp(bb24_s1, p24),
        "BlackBox-2.2k (Stage 1+2)": wrap_mlp(bb24_s2, p24),
    }

    print("\n[5/5] Running Rolling 3-Day Windows, Multi-Horizon Rollouts & Admissibility Checks...")
    df_win = evaluate_unseen_windows_and_horizons(models_dict, data)
    df_adm = check_physical_admissibility(models_dict, data)

    out_csv = os.path.join(SCRIPT_DIR, "diagnostic_unseen_windows.csv")
    df_win.to_csv(out_csv, index=False)
    adm_csv = os.path.join(SCRIPT_DIR, "diagnostic_admissibility.csv")
    df_adm.to_csv(adm_csv, index=False)

    # Build clean human-readable summary report
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("\n" + "=" * 115)
    log("EXPERIMENT 1A: 30 SLIDING 3-DAY WINDOWS (24h stride) & 10 NON-OVERLAPPING 3-DAY WINDOWS (May 3 - June 4)")
    log("=" * 115)
    df_sl = df_win[df_win["eval_type"] == "Sliding_3Day"]
    df_no = df_sl[df_sl["non_overlapping"]]
    df_ex = df_sl[df_sl["extreme_event"]]

    log(f"Counts per model: Sliding_3Day={df_sl['start_idx'].nunique()}, Non-Overlapping={df_no['start_idx'].nunique()}, Extreme-Event (rain>{data['max_tr_rain']:.2f} or ET0>{data['max_tr_et0']:.2f})={df_ex['start_idx'].nunique()}")
    log(f"{'Model':<27} | {'Params':>7} | {'All 30 Win Mean RMSE (SD)':>25} | {'10 Non-Ovlp Mean RMSE':>21} | {'Extreme Win RMSE':>16} | {'Worst Win RMSE':>14}")
    log("-" * 115)
    for m_name in models_dict:
        sub_all = df_sl[df_sl["model"] == m_name]
        sub_no = df_no[df_no["model"] == m_name]
        sub_ex = df_ex[df_ex["model"] == m_name]
        p_cnt = sub_all["params"].iloc[0]
        all_mean, all_sd = sub_all["rmse"].mean(), sub_all["rmse"].std()
        no_mean = sub_no["rmse"].mean()
        ex_mean = sub_ex["rmse"].mean()
        worst = sub_all["rmse"].max()
        log(f"{m_name:<27} | {p_cnt:>7,} | {all_mean:>14.3f} ({all_sd:.3f})    | {no_mean:>21.3f} | {ex_mean:>16.3f} | {worst:>14.3f}")

    # Win-rate table vs Proposed PINN on the 30 sliding windows
    log("\nHead-to-Head Across 30 Sliding 3-Day Windows (Mean R2 and Layer-1 RMSE):")
    log(f"{'Model':<27} | {'Mean R2':>9} | {'Median RMSE':>11} | {'Mean L1(5cm) RMSE':>17} | {'Ex-Event Mean R2':>16}")
    log("-" * 92)
    for m_name in models_dict:
        sub_all = df_sl[df_sl["model"] == m_name]
        sub_ex = df_ex[df_ex["model"] == m_name]
        log(f"{m_name:<27} | {sub_all['r2'].mean():>9.3f} | {sub_all['rmse'].median():>11.3f} | {sub_all['l1_rmse'].mean():>17.3f} | {sub_ex['r2'].mean():>16.3f}")

    log("\n" + "=" * 115)
    log("EXPERIMENT 1B & 2: MULTI-HORIZON CLOSED-LOOP ROLLOUTS FROM MAY 3 21:00 INITIAL CONDITION m(0)")
    log("=" * 115)
    log(f"{'Model':<27} | {'3d (72h) RMSE / R2':>20} | {'7d (168h) RMSE / R2':>20} | {'14d (336h) RMSE / R2':>21} | {'33d (791h) RMSE / R2':>21} | {'33d Unclip<0%':>13}")
    log("-" * 135)
    for m_name in models_dict:
        cells = []
        for h_lab in ["Horizon_3d", "Horizon_7d", "Horizon_14d", "Horizon_33d"]:
            row = df_win[(df_win["eval_type"] == h_lab) & (df_win["model"] == m_name)].iloc[0]
            cells.append(f"{row['rmse']:5.2f} / {row['r2']:6.3f}")
        row33 = df_win[(df_win["eval_type"] == "Horizon_33d") & (df_win["model"] == m_name)].iloc[0]
        unclip = f"{row33['unclipped_neg_pct']:.2f}%"
        log(f"{m_name:<27} | {cells[0]:>20} | {cells[1]:>20} | {cells[2]:>21} | {cells[3]:>21} | {unclip:>13}")

    log("\n" + "=" * 135)
    log("EXPERIMENT 3: CONTROL-ORIENTED PHYSICAL ADMISSIBILITY (JACOBIAN SIGN & ZERO-EQUILIBRIUM CHECKS)")
    log("=" * 135)
    log(f"{'Model':<27} | {'Dom dq1<0%':>10} | {'Trj dq1<0%':>10} | {'Dom dq_all<0%':>13} | {'Dom dr<0%':>9} | {'Dom dET0>0%':>11} | {'Trj dET0>0%':>11} | {'ZeroEq max+/min-':>18} | {'Dry+ET min(dm)':>14}")
    log("-" * 145)
    for _, r in df_adm.iterrows():
        zeq = f"{r['zero_eq_max_pos_dmdt']:+.3f}/{r['zero_eq_min_neg_dmdt']:+.3f}"
        log(
            f"{r['model']:<27} | {r['dom_dq_l1_neg_pct']:>9.2f}% | {r['traj_dq_l1_neg_pct']:>9.2f}% | {r['dom_dq_all_neg_pct']:>12.2f}% | "
            f"{r['dom_dr_all_neg_pct']:>8.2f}% | {r['dom_det_pos_pct']:>10.2f}% | {r['traj_det_pos_pct']:>10.2f}% | "
            f"{zeq:>18} | {r['dry_et_min_neg_dmdt']:>14.4f}"
        )
    log("-" * 145)
    log("Notes:")
    log("  - Dom/Trj dq1<0%: % of domain (2000 pts) / unseen trajectory (792 pts) where d(dm1/dt)/dq_irr < -1e-6 (irrigation lowers layer-1 rate).")
    log("  - Dom dq_all<0%: % across all 6 layers x 2000 pts where d(dmi/dt)/dq_irr < -1e-6.")
    log("  - Dom/Trj dET0>0%: % where higher ET0 INCREASES soil moisture rate (d(dmi/dt)/dET0 > +1e-6).")
    log("  - ZeroEq max+/min-: max positive (spontaneous water creation) & min negative (%VWC/h) at m=0, q_irr=0, r=0, ET0=0 (ideal: 0/0).")
    log("  - Dry+ET min(dm): most negative dm_i/dt (%VWC/h) when m=0, q_irr=0, r=0, ET0>0 (negative drift drives soil moisture < 0%).")
    log(f"\nTotal elapsed time: {(time.time() - t_all)/60.0:.2f} minutes")

    sum_txt = os.path.join(SCRIPT_DIR, "diagnostic_unseen_summary.txt")
    with open(sum_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved CSVs to {out_csv}, {adm_csv} and summary to {sum_txt}")


if __name__ == "__main__":
    main()
