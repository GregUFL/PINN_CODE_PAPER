#!/usr/bin/env python3
"""
consolidate_master_benchmarks.py
================================
Unifies and completes all 8 model configurations into:
  - benchmark_ablation_master.csv
  - benchmark_ablation_summary.txt
with ZERO missing cells, saving trained baseline/ablation weights under
Improved_Code/benchmark_weights/ so evaluations can be re-run without retraining.

8 Model Configurations:
  1. Proposed PINN (Full)                      [375,089 params, pretrained]
  2. Baseline 1: Persistence                   [0 params]
  3. Baseline 2: Linear State-Space ODE        [60 params, OLS]
  4. Baseline 3a: BlackBox-374k (Stage 1)      [374,156 params]
  5. Baseline 3b: BlackBox-374k (Stage 1+2)    [374,156 params]
  6. Baseline 3c: BlackBox-2.2k (Stage 1+2)    [2,190 params]
  7. Ablation 1: PINN Stage 1 Only (No J2)     [375,089 params]
  8. Ablation 2: PINN No Physics (lambda_c=0)  [375,089 params]
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.keras.backend.set_floatx("float64")

GLOBAL_SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
if not os.path.isdir(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "weights")
PINN_WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "pinn_weights_H56_lr7.97e-05.weights.h5")

BENCH_WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "benchmark_weights")
os.makedirs(BENCH_WEIGHTS_DIR, exist_ok=True)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, "code")
if os.path.isdir(CODE_DIR) and CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from tfsolverMIconstrb_float64 import rk4_integration
from pinn_model import TwoTankPINN

NORM = {"time_hours": 1.0, "rain": 30.0, "et0": 0.8, "irrigation": 1.0, "moisture": 40.0}
DEPTHS = ["5cm", "15cm", "25cm", "35cm", "45cm", "55cm"]


def set_seeds():
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_all_datasets():
    def read_merged(prefix):
        c = pd.read_json(os.path.join(DATA_DIR, f"campbell{prefix}_1h.json")).rename(
            columns={"measurement": "irrigation"}
        )
        d = pd.read_json(os.path.join(DATA_DIR, f"davis{prefix}_1h.json")).rename(
            columns={"measured_rain_mm_h": "rain", "potential_evapotranspiration_mm_h": "et0"}
        )
        i = pd.read_json(os.path.join(DATA_DIR, f"irrimeter{prefix}_1h.json"))[
            ["timestamp", "moisture_1", "moisture_2", "moisture_3", "moisture_4", "moisture_5", "moisture_6"]
        ]
        for df in (c, d, i):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = pd.merge(d[["timestamp", "rain", "et0"]], c[["timestamp", "irrigation"]], on="timestamp", how="inner")
        df = pd.merge(df, i, on="timestamp", how="inner")
        df["et0"] = df["et0"].clip(lower=0.0)
        df["time_hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0
        return df.sort_values("time_hours").dropna().reset_index(drop=True)

    df_tr = read_merged("")
    df_val_full = read_merged("v")
    df_unseen = df_val_full[df_val_full["time_hours"] >= 216].copy().reset_index(drop=True)
    df_unseen["time_hours"] = (df_unseen["timestamp"] - df_unseen["timestamp"].min()).dt.total_seconds() / 3600.0
    df_val3d = df_val_full[(df_val_full["time_hours"] >= 216) & (df_val_full["time_hours"] <= 288)].copy().reset_index(drop=True)

    def to_tensors(df):
        m = df[[f"moisture_{k}" for k in range(1, 7)]].values / NORM["moisture"]
        u = np.column_stack([
            df["irrigation"].values / NORM["irrigation"],
            df["rain"].values / NORM["rain"],
            df["et0"].values / NORM["et0"],
        ])
        t = df["time_hours"].values.reshape(-1, 1) / NORM["time_hours"]
        return (
            tf.constant(m, dtype=tf.float64),
            tf.constant(u, dtype=tf.float64),
            tf.constant(t, dtype=tf.float64),
        )

    st_tr, in_tr, t_tr = to_tensors(df_tr)
    st_val3d, in_val3d, t_val3d = to_tensors(df_val3d)
    st_un, in_un, t_un = to_tensors(df_unseen)

    dt = 1.0
    return {
        "df_tr": df_tr, "df_val3d": df_val3d, "df_unseen": df_unseen,
        "st_tr_tf": st_tr, "in_tr_tf": in_tr, "t_tr_tf": t_tr,
        "m_mid_tr": st_tr[:-1], "in_mid_tr": in_tr[:-1], "t_mid_tr": t_tr[:-1],
        "dm_mid_tr": (st_tr[1:] - st_tr[:-1]) / dt,
        "m0_tr_np": st_tr[0].numpy(),
        "meas_tr_denorm": st_tr[:-1].numpy() * NORM["moisture"],
        "st_val3d_tf": st_val3d, "in_mid_val3d": in_val3d[:-1], "t_val3d_tf": t_val3d,
        "m0_val3d_np": st_val3d[0].numpy(),
        "meas_val3d_denorm": st_val3d[:-1].numpy() * NORM["moisture"],
        "st_un_tf": st_un, "in_un_tf": in_un, "t_un_tf": t_un,
        "max_tr_rain": float(df_tr["rain"].max()),
        "max_tr_et0": float(df_tr["et0"].max()),
    }


# ==============================================================================
# FAST RK4 INTEGRATORS (CLIPPED & UNCLIPPED)
# ==============================================================================
def rk4_unclipped_tf(system_func, time_points, initial_state, dt, forced_data):
    num_steps = tf.shape(time_points)[0]
    init_st = tf.cast(initial_state, tf.float64)
    ta = tf.TensorArray(dtype=tf.float64, size=num_steps)
    ta = ta.write(0, init_st)

    def body(i, st, ta_):
        t = time_points[i - 1]
        u = forced_data[i - 1]
        k1 = system_func(t, st, u)
        k2 = system_func(t + 0.5 * dt, st + 0.5 * dt * k1, u)
        k3 = system_func(t + 0.5 * dt, st + 0.5 * dt * k2, u)
        k4 = system_func(t + dt, st + dt * k3, u)
        nst = st + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
        ta_ = ta_.write(i, nst)
        return i + 1, nst, ta_

    _, _, ta = tf.while_loop(lambda i, *_: i < num_steps, body, [tf.constant(1), init_st, ta])
    return ta.stack()


def build_pinn_wrappers(pinn_instance):
    @tf.function
    def rhs_batch(m_batch, u_batch):
        m1, m2, m3, m4, m5, m6 = [m_batch[:, i:i+1] for i in range(6)]
        qi, rain, et0 = u_batch[:, 0:1], u_batch[:, 1:2], u_batch[:, 2:3]
        f1 = pinn_instance.f1_out(pinn_instance.f1_dense3(pinn_instance.f1_dense2(pinn_instance.f1_dense1(m1))))
        f2 = pinn_instance.f2_out(pinn_instance.f2_dense3(pinn_instance.f2_dense2(pinn_instance.f2_dense1(m2))))
        f3 = pinn_instance.f3_out(pinn_instance.f3_dense3(pinn_instance.f3_dense2(pinn_instance.f3_dense1(m3))))
        f4 = pinn_instance.f4_out(pinn_instance.f4_dense3(pinn_instance.f4_dense2(pinn_instance.f4_dense1(m4))))
        f5 = pinn_instance.f5_out(pinn_instance.f5_dense3(pinn_instance.f5_dense2(pinn_instance.f5_dense1(m5))))
        f6 = pinn_instance.f6_out(pinn_instance.f6_dense3(pinn_instance.f6_dense2(pinn_instance.f6_dense1(m6))))
        f12 = pinn_instance.f12_out(pinn_instance.f12_dense3(pinn_instance.f12_dense2(pinn_instance.f12_dense1(tf.concat([m1, m2], 1)))))
        f23 = pinn_instance.f23_out(pinn_instance.f23_dense3(pinn_instance.f23_dense2(pinn_instance.f23_dense1(tf.concat([m2, m3], 1)))))
        f34 = pinn_instance.f34_out(pinn_instance.f34_dense3(pinn_instance.f34_dense2(pinn_instance.f34_dense1(tf.concat([m3, m4], 1)))))
        f45 = pinn_instance.f45_out(pinn_instance.f45_dense3(pinn_instance.f45_dense2(pinn_instance.f45_dense1(tf.concat([m4, m5], 1)))))
        f56 = pinn_instance.f56_out(pinn_instance.f56_dense3(pinn_instance.f56_dense2(pinn_instance.f56_dense1(tf.concat([m5, m6], 1)))))
        inflow = pinn_instance.inflow_out(pinn_instance.inflow_dense3(pinn_instance.inflow_dense2(pinn_instance.inflow_dense1(qi))))
        ir1 = pinn_instance.rain1_out(pinn_instance.rain1_dense3(pinn_instance.rain1_dense2(pinn_instance.rain1_dense1(rain))))
        ir2 = pinn_instance.rain2_out(pinn_instance.rain2_dense3(pinn_instance.rain2_dense2(pinn_instance.rain2_dense1(rain))))
        ir3 = pinn_instance.rain3_out(pinn_instance.rain3_dense3(pinn_instance.rain3_dense2(pinn_instance.rain3_dense1(rain))))
        ir4 = pinn_instance.rain4_out(pinn_instance.rain4_dense3(pinn_instance.rain4_dense2(pinn_instance.rain4_dense1(rain))))
        ir5 = pinn_instance.rain5_out(pinn_instance.rain5_dense3(pinn_instance.rain5_dense2(pinn_instance.rain5_dense1(rain))))
        ir6 = pinn_instance.rain6_out(pinn_instance.rain6_dense3(pinn_instance.rain6_dense2(pinn_instance.rain6_dense1(rain))))
        pc1 = pinn_instance.pc1_out(pinn_instance.pc1_dense4(pinn_instance.pc1_dense3(pinn_instance.pc1_dense2(pinn_instance.pc1_dense1(tf.concat([m1, et0], 1))))))
        pc2 = pinn_instance.pc2_out(pinn_instance.pc2_dense4(pinn_instance.pc2_dense3(pinn_instance.pc2_dense2(pinn_instance.pc2_dense1(tf.concat([m2, et0], 1))))))
        pc3 = pinn_instance.pc3_out(pinn_instance.pc3_dense4(pinn_instance.pc3_dense3(pinn_instance.pc3_dense2(pinn_instance.pc3_dense1(tf.concat([m3, et0], 1))))))
        pc4 = pinn_instance.pc4_out(pinn_instance.pc4_dense4(pinn_instance.pc4_dense3(pinn_instance.pc4_dense2(pinn_instance.pc4_dense1(tf.concat([m4, et0], 1))))))
        pc5 = pinn_instance.pc5_out(pinn_instance.pc5_dense4(pinn_instance.pc5_dense3(pinn_instance.pc5_dense2(pinn_instance.pc5_dense1(tf.concat([m5, et0], 1))))))
        pc6 = pinn_instance.pc6_out(pinn_instance.pc6_dense4(pinn_instance.pc6_dense3(pinn_instance.pc6_dense2(pinn_instance.pc6_dense1(tf.concat([m6, et0], 1))))))
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
    def sys_func(t, state, input_vec):
        m_b = tf.reshape(state, [1, 6])
        u_b = tf.reshape(input_vec, [1, 3])
        return tf.reshape(rhs_batch(m_b, u_b), [6])

    return rhs_batch, sys_func


# ==============================================================================
# MODEL DEFINITIONS & TRAINING WITH WEIGHT CACHING
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
        return self.out(self.d4(self.d3(self.d2(self.d1(tf.concat([m, u], axis=1))))))


def get_blackbox_models(data, hidden_dim, tag, stage1_epochs=2000, stage2_epochs=150, stage2_lr=2e-5):
    w_s1 = os.path.join(BENCH_WEIGHTS_DIR, f"{tag}_s1.weights.h5")
    w_s2 = os.path.join(BENCH_WEIGHTS_DIR, f"{tag}_s12.weights.h5")

    if os.path.isfile(w_s1) and os.path.isfile(w_s2):
        m_s1 = BlackBoxNeuralODE(hidden_dim=hidden_dim)
        _ = m_s1(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64))
        m_s2 = BlackBoxNeuralODE(hidden_dim=hidden_dim)
        _ = m_s2(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64))
        n_params = m_s1.count_params()
        print(f"  [Cache hit] Loading {tag} weights ({n_params:,} params)...")
        m_s1.load_weights(w_s1)
        m_s2.load_weights(w_s2)
        return m_s1, m_s2, n_params

    set_seeds()
    mlp = BlackBoxNeuralODE(hidden_dim=hidden_dim)
    _ = mlp(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64))
    n_params = mlp.count_params()

    dm_mid_tr = data["dm_mid_tr"]
    dinv_std = tf.math.rsqrt(tf.reduce_mean(tf.square(dm_mid_tr - tf.reduce_mean(dm_mid_tr, 0, keepdims=True)), 0))
    opt_s1 = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True, clipvalue=0.5)
    m_tr, in_tr = data["m_mid_tr"], data["in_mid_tr"]

    @tf.function
    def step_s1():
        with tf.GradientTape() as tape:
            derr = (mlp(m_tr, in_tr) - dm_mid_tr) * dinv_std
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(derr), axis=1))
        opt_s1.apply_gradients(zip(tape.gradient(loss, mlp.trainable_variables), mlp.trainable_variables))
        return loss

    t0 = time.time()
    print(f"  Training {tag} Stage 1 ({stage1_epochs} ep, {n_params:,} params)...")
    for ep in range(stage1_epochs + 1):
        l1 = step_s1()
        if ep % 500 == 0:
            print(f"    Ep {ep:4d} | j1 = {l1.numpy():.5f}")
    mlp.save_weights(w_s1)
    s1_weights = mlp.get_weights()
    print(f"    Stage 1 saved in {time.time() - t0:.1f}s")

    st_tr_tf = data["st_tr_tf"]
    inv_std = tf.math.rsqrt(tf.reduce_mean(tf.square(st_tr_tf - tf.reduce_mean(st_tr_tf, 0, keepdims=True)), 0))
    t_tr_tf, m0_tr_np = data["t_tr_tf"], data["m0_tr_np"]
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=stage2_lr, decay_steps=max(stage2_epochs // 4, 25), decay_rate=0.9, staircase=True
    )
    opt_s2 = tf.keras.optimizers.Adam(learning_rate=lr_sched, amsgrad=True, clipvalue=0.5)

    @tf.function
    def sys_mlp(t, s, u):
        return tf.reshape(mlp(tf.reshape(s, [1, 6]), tf.reshape(u, [1, 3])), [6])

    @tf.function
    def step_s2():
        with tf.GradientTape() as tape:
            mp = rk4_integration(sys_mlp, t_tr_tf, m0_tr_np, 1.0, forced_data=in_tr)
            j2 = tf.reduce_mean(tf.reduce_sum(tf.abs((mp - st_tr_tf) * inv_std), axis=1))
        opt_s2.apply_gradients(zip(tape.gradient(j2, mlp.trainable_variables), mlp.trainable_variables))
        return j2

    best_j2, best_w = float("inf"), s1_weights
    t0 = time.time()
    print(f"  Training {tag} Stage 2 ({stage2_epochs} ep, lr={stage2_lr})...")
    for ep in range(stage2_epochs + 1):
        j2_v = float(step_s2().numpy())
        if np.isfinite(j2_v) and j2_v < best_j2:
            best_j2, best_w = j2_v, mlp.get_weights()
        if ep % max(stage2_epochs // 5, 1) == 0:
            print(f"    Ep {ep:4d} | j2 = {j2_v:.5f} (best={best_j2:.5f})")
    mlp.set_weights(best_w)
    mlp.save_weights(w_s2)
    m_s1 = BlackBoxNeuralODE(hidden_dim=hidden_dim)
    _ = m_s1(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64))
    m_s2 = BlackBoxNeuralODE(hidden_dim=hidden_dim)
    _ = m_s2(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64))
    m_s1.load_weights(w_s1)
    m_s2.load_weights(w_s2)
    print(f"    Stage 2 saved in {time.time() - t0:.1f}s")
    return m_s1, m_s2, n_params


def get_pinn_ablation(data, constrained: bool, tag: str):
    w_file = os.path.join(BENCH_WEIGHTS_DIR, f"{tag}.weights.h5")
    if os.path.isfile(w_file):
        model = TwoTankPINN(neurons=56)
        _ = model(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64), tf.zeros((1, 1), tf.float64))
        print(f"  [Cache hit] Loading {tag} weights...")
        model.load_weights(w_file)
        return model

    set_seeds()
    model = TwoTankPINN(neurons=56)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=7.970521877412297e-05, decay_steps=1000, decay_rate=0.9, staircase=True
    )
    opt1 = tf.keras.optimizers.Adam(learning_rate=lr_sched, amsgrad=True, clipvalue=0.5)
    dm_tr = data["dm_mid_tr"]
    dinv_std = tf.math.rsqrt(tf.reduce_mean(tf.square(dm_tr - tf.reduce_mean(dm_tr, 0, keepdims=True)), 0))
    m_tr, in_tr, t_tr = data["m_mid_tr"], data["in_mid_tr"], data["t_mid_tr"]
    zero64 = tf.constant(0.0, dtype=tf.float64)
    eps = tf.constant(1e-6, dtype=tf.float64)
    l3, l4, l5 = 113982766.31047939, 1773295.4683835912, 126704655.89679742

    @tf.function
    def step_s1():
        with tf.GradientTape() as tape:
            out = model(m_tr, in_tr, t_tr)
            (f1, f2, f3, f4, f5, f6, f12, f23, f34, f45, f56, inflow,
             ir1, ir2, ir3, ir4, ir5, ir6, pc1, pc2, pc3, pc4, pc5, pc6,
             fl1, fl2, fl3, fl4, fl5) = out[:29]
            dm_pred = tf.concat([
                inflow + ir1 - f1 - f12 - pc1 - fl1,
                f1 + f12 - f2 - f23 - pc2 + ir2 - fl2,
                f2 + f23 - f3 - f34 - pc3 + ir3 - fl3,
                f3 + f34 - f4 - f45 - pc4 + ir4 - fl4,
                f4 + f45 - f5 - f56 - pc5 + ir5 - fl5,
                f5 + f56 - f6 - pc6 + ir6,
            ], axis=1)
            j1 = tf.reduce_mean(tf.reduce_sum(tf.square((dm_pred - dm_tr) * dinv_std), axis=1))
            if not constrained:
                loss = j1
            else:
                m_r = tf.random.uniform([1000, 6], 0.0, 1.0, dtype=tf.float64)
                u_r = tf.random.uniform([1000, 3], 0.0, 1.0, dtype=tf.float64)
                t_r = tf.random.uniform([1000, 1], 0.0, 1.0, dtype=tf.float64)
                out_r = model(m_r, u_r, t_r)
                pos_list = list(out_r[0:5]) + list(out_r[11:29])
                j3 = sum(tf.reduce_mean(tf.square(tf.minimum(zero64, x))) for x in pos_list)
                j4 = sum(tf.reduce_mean(tf.square(tf.minimum(zero64, x))) for x in out_r[29:65])
                qi, rain, et0 = in_tr[:, 0:1], in_tr[:, 1:2], in_tr[:, 2:3]
                m1, m2, m3, m4, m5, m6 = [m_tr[:, k:k+1] for k in range(6)]
                dms = [m1 - m2, m2 - m3, m3 - m4, m4 - m5, m5 - m6]
                z_pairs = (
                    [(f, mv) for f, mv in zip([f1, f2, f3, f4, f5], [m1, m2, m3, m4, m5])] +
                    [(f, dm) for f, dm in zip([f12, f23, f34, f45, f56], dms)] +
                    [(inflow, qi)] +
                    [(ir, rain) for ir in [ir1, ir2, ir3, ir4, ir5, ir6]] +
                    [(pc, mv) for pc, mv in zip([pc1, pc2, pc3, pc4, pc5, pc6], [m1, m2, m3, m4, m5, m6])] +
                    [(pc, et0) for pc in [pc1, pc2, pc3, pc4, pc5, pc6]] +
                    [(fl, mv) for fl, mv in zip([fl1, fl2, fl3, fl4, fl5], [m1, m2, m3, m4, m5])]
                )
                j5 = sum(tf.reduce_mean(tf.square(fx) * tf.cast(tf.abs(var) <= eps, tf.float64)) for fx, var in z_pairs)
                loss = j1 + l3 * j3 + l4 * j4 + l5 * j5
        opt1.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, j1

    t0 = time.time()
    print(f"  Training {tag} Stage 1 (3000 epochs, constrained={constrained})...")
    for ep in range(3001):
        tot, j1v = step_s1()
        if ep % 500 == 0:
            print(f"    Ep {ep:4d} | tot={tot.numpy():.4f}, j1={j1v.numpy():.4f}")
    print(f"    Stage 1 finished in {time.time() - t0:.1f}s")

    if not constrained:
        # Stage 2 unconstrained RK4 fine-tuning
        opt2 = tf.keras.optimizers.Adam(learning_rate=5e-6, amsgrad=True, clipvalue=0.5)
        st_tr_tf = data["st_tr_tf"]
        inv_std_s = tf.math.rsqrt(tf.reduce_mean(tf.square(st_tr_tf - tf.reduce_mean(st_tr_tf, 0, keepdims=True)), 0))
        _, sys_fast = build_pinn_wrappers(model)

        @tf.function
        def step_s2():
            with tf.GradientTape() as tape:
                mp = rk4_integration(sys_fast, data["t_tr_tf"], data["m0_tr_np"], 1.0, forced_data=in_tr)
                j2 = tf.reduce_mean(tf.reduce_sum(tf.abs((mp - st_tr_tf) * inv_std_s), axis=1))
            opt2.apply_gradients(zip(tape.gradient(j2, model.trainable_variables), model.trainable_variables))
            return j2

        t0 = time.time()
        print(f"  Training {tag} Stage 2 unconstrained RK4 (100 epochs)...")
        for ep in range(101):
            j2v = step_s2()
            if ep % 25 == 0:
                print(f"    Ep {ep:3d} | j2={j2v.numpy():.5f}")
        print(f"    Stage 2 finished in {time.time() - t0:.1f}s")

    model.save_weights(w_file)
    return model


def fit_linear_state_space(data):
    X = np.hstack([data["m_mid_tr"].numpy(), data["in_mid_tr"].numpy(), np.ones((len(data["m_mid_tr"]), 1))])
    Y = data["dm_mid_tr"].numpy()
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    W_tf = tf.constant(W, dtype=tf.float64)

    @tf.function
    def rhs_batch(m, u):
        ones = tf.ones((tf.shape(m)[0], 1), dtype=tf.float64)
        return tf.matmul(tf.concat([m, u, ones], axis=1), W_tf)

    @tf.function
    def sys_func(t, s, u):
        feats = tf.concat([s, u, tf.constant([1.0], dtype=tf.float64)], axis=0)
        return tf.reshape(tf.matmul(tf.reshape(feats, [1, 10]), W_tf), [6])

    return {"type": "ode", "params": int(W.size), "rhs_batch": rhs_batch, "sys_func": sys_func, "pinn_inst": None}


# ==============================================================================
# UNIFIED EVALUATION ENGINE (ZERO MISSING CELLS)
# ==============================================================================
def flat_rmse_r2(meas, pred):
    y_t, y_p = meas.reshape(-1), pred.reshape(-1)
    err = y_t - y_p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def rollout_slice(m_info, st_tf, t_tf, in_tf, s_idx, n_steps, unclipped=False):
    m0 = st_tf[s_idx].numpy()
    if m_info["type"] == "persistence":
        return np.tile(m0, (n_steps, 1)) * NORM["moisture"]
    t_sl = t_tf[s_idx : s_idx + n_steps + 1]
    in_sl = in_tf[s_idx : s_idx + n_steps]
    fn = rk4_unclipped_tf if unclipped else rk4_integration
    pred_tf = fn(m_info["sys_func"], t_sl, m0, 1.0, forced_data=in_sl)
    return (pred_tf.numpy() * NORM["moisture"])[:-1]


def check_internal_flux_violations(pinn_inst):
    if pinn_inst is None:
        return None, None
    np.random.seed(GLOBAL_SEED)
    m_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 6)), dtype=tf.float64)
    u_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 3)), dtype=tf.float64)
    t_rand = tf.constant(np.random.uniform(0.0, 1.0, (2000, 1)), dtype=tf.float64)
    out = pinn_inst(m_rand, u_rand, t_rand)
    pos_fluxes = list(out[0:5]) + list(out[11:29])  # 23 non-negative branches
    mono_derivs = list(out[29:65])                   # 36 partial derivatives
    neg_cnt = sum(int(np.sum(x.numpy() < -1e-4)) for x in pos_fluxes)
    mono_cnt = sum(int(np.sum(x.numpy() < -1e-4)) for x in mono_derivs)
    return 100.0 * neg_cnt / (len(pos_fluxes) * 2000), 100.0 * mono_cnt / (len(mono_derivs) * 2000)


def check_jacobian_and_equilibrium(m_info, data):
    if m_info["type"] == "persistence":
        return {
            "dom_dq_l1_neg_pct": 0.0, "dom_dq_l1_min": 0.0,
            "dom_dq_all_neg_pct": 0.0, "dom_dr_all_neg_pct": 0.0,
            "dom_det_pos_pct": 0.0, "dom_det_max": 0.0,
            "traj_dq_l1_neg_pct": 0.0, "traj_dq_all_neg_pct": 0.0,
            "traj_dr_all_neg_pct": 0.0, "traj_det_pos_pct": 0.0,
            "traj_det_max": 0.0,
            "zero_eq_max_pos_dmdt": 0.0, "zero_eq_min_neg_dmdt": 0.0,
            "dry_et_min_neg_dmdt": 0.0,
        }
    np.random.seed(GLOBAL_SEED)
    m_dom = tf.constant(np.random.uniform(0.0, 1.0, size=(2000, 6)), dtype=tf.float64)
    u_dom = tf.constant(np.random.uniform(0.0, 1.0, size=(2000, 3)), dtype=tf.float64)
    m_traj, u_traj = data["st_un_tf"], data["in_un_tf"]
    rhs_batch = m_info["rhs_batch"]
    scale_dm = NORM["moisture"]
    scale_u = tf.constant([NORM["irrigation"], NORM["rain"], NORM["et0"]], dtype=tf.float64)

    def jac_stats(m_pts, u_pts):
        with tf.GradientTape() as tape:
            tape.watch(u_pts)
            dm_phys = rhs_batch(m_pts, u_pts) * scale_dm
        jac = (tape.batch_jacobian(dm_phys, u_pts) / scale_u[None, None, :]).numpy()
        tol = 1e-6
        return {
            "dq_l1_neg_pct": float(100.0 * np.mean(jac[:, 0, 0] < -tol)),
            "dq_l1_min": float(np.min(jac[:, 0, 0])),
            "dq_all_neg_pct": float(100.0 * np.mean(jac[:, :, 0] < -tol)),
            "dr_all_neg_pct": float(100.0 * np.mean(jac[:, :, 1] < -tol)),
            "det_pos_pct": float(100.0 * np.mean(jac[:, :, 2] > tol)),
            "det_max": float(np.max(jac[:, :, 2])),
        }

    s_dom = jac_stats(m_dom, u_dom)
    s_trj = jac_stats(m_traj, u_traj)

    m_z = tf.zeros((4, 6), dtype=tf.float64)
    u_z = tf.constant([
        [0.0, 0.0, 0.0 / NORM["et0"]],
        [0.0, 0.0, 0.1 / NORM["et0"]],
        [0.0, 0.0, 0.4 / NORM["et0"]],
        [0.0, 0.0, 0.8 / NORM["et0"]],
    ], dtype=tf.float64)
    dm_z = (rhs_batch(m_z, u_z) * scale_dm).numpy()
    return {
        "dom_dq_l1_neg_pct": s_dom["dq_l1_neg_pct"],
        "dom_dq_l1_min": s_dom["dq_l1_min"],
        "dom_dq_all_neg_pct": s_dom["dq_all_neg_pct"],
        "dom_dr_all_neg_pct": s_dom["dr_all_neg_pct"],
        "dom_det_pos_pct": s_dom["det_pos_pct"],
        "dom_det_max": s_dom["det_max"],
        "traj_dq_l1_neg_pct": s_trj["dq_l1_neg_pct"],
        "traj_dq_all_neg_pct": s_trj["dq_all_neg_pct"],
        "traj_dr_all_neg_pct": s_trj["dr_all_neg_pct"],
        "traj_det_pos_pct": s_trj["det_pos_pct"],
        "traj_det_max": s_trj["det_max"],
        "zero_eq_max_pos_dmdt": float(np.max(dm_z[0])),
        "zero_eq_min_neg_dmdt": float(np.min(dm_z[0])),
        "dry_et_min_neg_dmdt": float(np.min(dm_z[1:])),
    }


def evaluate_model_complete(m_name, m_info, data):
    # 1. 35-day Training Rollout (839h)
    pred_tr = rollout_slice(m_info, data["st_tr_tf"], data["t_tr_tf"], data["in_tr_tf"], 0, len(data["meas_tr_denorm"]))
    tr_rmse, tr_r2 = flat_rmse_r2(data["meas_tr_denorm"], pred_tr)

    # 2. 3-day Validation Rollout (72h) overall + per-layer
    pred_val3d = rollout_slice(m_info, data["st_val3d_tf"], data["t_val3d_tf"], data["in_mid_val3d"], 0, len(data["meas_val3d_denorm"]))
    val3d_rmse, val3d_r2 = flat_rmse_r2(data["meas_val3d_denorm"], pred_val3d)
    layer_rmses = {}
    for idx, d_lbl in enumerate(DEPTHS):
        lr_rmse, _ = flat_rmse_r2(data["meas_val3d_denorm"][:, idx], pred_val3d[:, idx])
        layer_rmses[f"val3d_{d_lbl}_rmse"] = lr_rmse

    # 3. 30 Sliding 3-Day Unseen Windows (24h stride across 792h)
    df_u = data["df_unseen"]
    start_indices = list(range(0, len(df_u) - 72, 24))
    win_rmses, ext_rmses = [], []
    for s_idx in start_indices:
        w_df = df_u.iloc[s_idx : s_idx + 72]
        is_ext = (w_df["rain"].max() > data["max_tr_rain"]) or (w_df["et0"].max() > data["max_tr_et0"])
        meas_w = data["st_un_tf"][s_idx : s_idx + 72].numpy() * NORM["moisture"]
        pred_w = rollout_slice(m_info, data["st_un_tf"], data["t_un_tf"], data["in_un_tf"], s_idx, 72)
        w_rmse, _ = flat_rmse_r2(meas_w, pred_w)
        win_rmses.append(w_rmse)
        if is_ext:
            ext_rmses.append(w_rmse)

    # 4. Multi-Horizon Rollouts from May 3 21:00 (3d, 7d, 14d, 33d) + 33d unclipped <0%
    h_metrics = {}
    for h_lbl, n_steps in [("3d", 72), ("7d", 168), ("14d", 336), ("33d", len(df_u) - 1)]:
        meas_h = data["st_un_tf"][0:n_steps].numpy() * NORM["moisture"]
        pred_h = rollout_slice(m_info, data["st_un_tf"], data["t_un_tf"], data["in_un_tf"], 0, n_steps)
        hr, hr2 = flat_rmse_r2(meas_h, pred_h)
        h_metrics[f"h{h_lbl}_rmse"] = hr
        h_metrics[f"h{h_lbl}_r2"] = hr2
        if h_lbl == "33d":
            if m_info["type"] == "persistence":
                unclip_neg = 0.0
            else:
                pred_unclip = rollout_slice(m_info, data["st_un_tf"], data["t_un_tf"], data["in_un_tf"], 0, n_steps, unclipped=True)
                unclip_neg = float(100.0 * np.mean(pred_unclip < 0.0))
            h_metrics["h33d_unclipped_neg_pct"] = unclip_neg

    # 5. Jacobian admissibility & equilibrium + internal PINN flux constraints
    adm = check_jacobian_and_equilibrium(m_info, data)
    c1_pct, c2_pct = check_internal_flux_violations(m_info["pinn_inst"])

    return {
        "model": m_name,
        "params": m_info["params"],
        "train35d_rmse": tr_rmse,
        "train35d_r2": tr_r2,
        "val3d_rmse": val3d_rmse,
        "val3d_r2": val3d_r2,
        **layer_rmses,
        "win30_mean_rmse": float(np.mean(win_rmses)),
        "win30_sd_rmse": float(np.std(win_rmses, ddof=1)),
        "win30_median_rmse": float(np.median(win_rmses)),
        "win30_extreme_rmse": float(np.mean(ext_rmses)),
        "win30_worst_rmse": float(np.max(win_rmses)),
        **h_metrics,
        **adm,
        "flux_c1_neg_pct": c1_pct if c1_pct is not None else np.nan,
        "flux_c2_mon_pct": c2_pct if c2_pct is not None else np.nan,
    }


def compute_daily_uptake_diagnostics(pinn_model, data):
    """Verify exact neural evapotranspirative depletion pc_i (mm/day = %VWC/h*1h*24h for 100mm layers)."""
    out_tr = pinn_model(data["st_tr_tf"], data["in_tr_tf"], data["t_tr_tf"])
    pcs_tr = np.hstack([x.numpy() for x in out_tr[18:24]]) * NORM["moisture"]  # (%VWC/h = mm/h per 10cm)
    df_pc = pd.DataFrame(pcs_tr, columns=[f"pc_{d}" for d in DEPTHS])
    df_pc["total_mm_h"] = df_pc.sum(axis=1)
    df_pc["date"] = data["df_tr"]["timestamp"].dt.date
    df_pc["et0_mm_h"] = data["df_tr"]["et0"].values
    daily = df_pc.groupby("date").agg(
        hours=("total_mm_h", "count"),
        pc_sum_mm_d=("total_mm_h", "sum"),
        et0_mm_d=("et0_mm_h", "sum"),
        **{f"pc_{d}_mm_d": (f"pc_{d}", "sum") for d in DEPTHS},
    )
    full_days = daily[daily["hours"] == 24]
    out_un = pinn_model(data["st_un_tf"], data["in_un_tf"], data["t_un_tf"])
    pcs_un = np.hstack([x.numpy() for x in out_un[18:24]]) * NORM["moisture"]
    df_pc_u = pd.DataFrame(pcs_un, columns=[f"pc_{d}" for d in DEPTHS])
    df_pc_u["total_mm_h"] = df_pc_u.sum(axis=1)
    df_pc_u["date"] = data["df_unseen"]["timestamp"].dt.date
    df_pc_u["et0_mm_h"] = data["df_unseen"]["et0"].values
    daily_u = df_pc_u.groupby("date").agg(
        hours=("total_mm_h", "count"),
        pc_sum_mm_d=("total_mm_h", "sum"),
        et0_mm_d=("et0_mm_h", "sum"),
    )
    full_days_u = daily_u[daily_u["hours"] == 24]
    return full_days, full_days_u


def main():
    t0 = time.time()
    print("=" * 110)
    print("CONSOLIDATING MASTER BENCHMARK, ABLATION & CONTROL-ADMISSIBILITY SUITE (8 MODELS)")
    print("=" * 110)
    data = load_all_datasets()

    def wrap_bb(bb_inst, n_params):
        @tf.function
        def rhs_b(m, u):
            return bb_inst(m, u)
        @tf.function
        def sys_f(t, s, u):
            return tf.reshape(bb_inst(tf.reshape(s, [1, 6]), tf.reshape(u, [1, 3])), [6])
        return {"type": "ode", "params": n_params, "rhs_batch": rhs_b, "sys_func": sys_f, "pinn_inst": None}

    # 1. Proposed PINN (Full)
    print("\n[1/6] Loading Proposed PINN (Full)...")
    pinn_full = TwoTankPINN(neurons=56)
    _ = pinn_full(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64), tf.zeros((1, 1), tf.float64))
    pinn_full.load_weights(PINN_WEIGHTS_FILE)
    pinn_full_rhs, pinn_full_sys = build_pinn_wrappers(pinn_full)

    # 2. Linear State-Space ODE
    print("[2/6] Fitting Linear State-Space ODE...")
    lin_ss = fit_linear_state_space(data)

    # 3. BlackBox-374k (S1 and S1+2)
    print("[3/6] Loading/Training BlackBox-374k (h=350)...")
    bb350_s1, bb350_s2, p350 = get_blackbox_models(data, 350, "bb350", stage1_epochs=2000, stage2_epochs=150, stage2_lr=2e-5)

    # 4. BlackBox-2.2k (S1 and S1+2)
    print("[4/6] Loading/Training BlackBox-2.2k (h=24)...")
    _, bb24_s2, p24 = get_blackbox_models(data, 24, "bb24", stage1_epochs=2000, stage2_epochs=500, stage2_lr=1e-4)

    # 5. Ablation 1: Stage 1 Only
    print("[5/6] Loading/Training Ablation 1 (Stage 1 Only, No J2)...")
    pinn_ab1 = get_pinn_ablation(data, constrained=True, tag="ablation1_stage1_only")
    ab1_rhs, ab1_sys = build_pinn_wrappers(pinn_ab1)

    # 6. Ablation 2: No Physics (lambda_c = 0)
    print("[6/6] Loading/Training Ablation 2 (No Physics, lambda_c=0)...")
    pinn_ab2 = get_pinn_ablation(data, constrained=False, tag="ablation2_no_physics")
    ab2_rhs, ab2_sys = build_pinn_wrappers(pinn_ab2)

    models_ordered = [
        ("Proposed PINN (Full)", {"type": "ode", "params": pinn_full.count_params(), "rhs_batch": pinn_full_rhs, "sys_func": pinn_full_sys, "pinn_inst": pinn_full}),
        ("Baseline 1: Persistence", {"type": "persistence", "params": 0, "pinn_inst": None}),
        ("Baseline 2: Linear State-Space ODE", lin_ss),
        ("Baseline 3a: BlackBox-374k (Stage 1)", wrap_bb(bb350_s1, p350)),
        ("Baseline 3b: BlackBox-374k (Stage 1+2)", wrap_bb(bb350_s2, p350)),
        ("Baseline 3c: BlackBox-2.2k (Stage 1+2)", wrap_bb(bb24_s2, p24)),
        ("Ablation 1: PINN Stage 1 Only (No J2)", {"type": "ode", "params": pinn_ab1.count_params(), "rhs_batch": ab1_rhs, "sys_func": ab1_sys, "pinn_inst": pinn_ab1}),
        ("Ablation 2: PINN No Physics (lambda_c = 0)", {"type": "ode", "params": pinn_ab2.count_params(), "rhs_batch": ab2_rhs, "sys_func": ab2_sys, "pinn_inst": pinn_ab2}),
    ]

    print("\nEvaluating all 8 models across all horizons, 30 sliding windows, and admissibility checks...")
    rows = []
    for name, info in models_ordered:
        print(f"  -> Evaluating {name} ({info['params']:,} params)...")
        rows.append(evaluate_model_complete(name, info, data))

    df_master = pd.DataFrame(rows)
    master_csv = os.path.join(SCRIPT_DIR, "benchmark_ablation_master.csv")
    df_master.to_csv(master_csv, index=False)

    # Compute daily pc_i diagnostics on Proposed PINN
    daily_tr, daily_un = compute_daily_uptake_diagnostics(pinn_full, data)

    # Format summary report
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("\n" + "=" * 135)
    log("MASTER BENCHMARK, ABLATION & CONTROL-ADMISSIBILITY TABLE (ALL 8 CONFIGURATIONS)")
    log("=" * 135)
    log(f"{'Model':<42} | {'Params':>7} | {'Tr35d RMSE/R2':>15} | {'Val3d RMSE/R2':>15} | {'30Win Mean(SD)':>15} | {'WorstWin':>8} | {'33d R2':>7} | {'Unclip<0%':>9}")
    log("-" * 135)
    for _, r in df_master.iterrows():
        log(f"{r['model']:<42} | {int(r['params']):>7,} | {r['train35d_rmse']:5.2f} / {r['train35d_r2']:6.3f} | {r['val3d_rmse']:5.2f} / {r['val3d_r2']:6.3f} | {r['win30_mean_rmse']:5.2f} ({r['win30_sd_rmse']:.2f})   | {r['win30_worst_rmse']:8.2f} | {r['h33d_r2']:7.3f} | {r['h33d_unclipped_neg_pct']:8.2f}%")

    log("\nPER-LAYER 3-DAY VALIDATION RMSE (%VWC)")
    log("-" * 115)
    log(f"{'Model':<42} | {'5cm':>6} | {'15cm':>6} | {'25cm':>6} | {'35cm':>6} | {'45cm':>6} | {'55cm':>6} | {'Overall':>7}")
    log("-" * 115)
    for _, r in df_master.iterrows():
        log(f"{r['model']:<42} | {r['val3d_5cm_rmse']:6.2f} | {r['val3d_15cm_rmse']:6.2f} | {r['val3d_25cm_rmse']:6.2f} | {r['val3d_35cm_rmse']:6.2f} | {r['val3d_45cm_rmse']:6.2f} | {r['val3d_55cm_rmse']:6.2f} | {r['val3d_rmse']:7.2f}")

    log("\nCONTROL-ORIENTED ADMISSIBILITY & FLUX CONSTRAINTS")
    log("-" * 165)
    log(f"{'Model':<42} | {'Trj dqAll<0%':>12} | {'Trj dq1<0%':>10} | {'Trj dr<0%':>9} | {'Trj dET0>0%':>11} | {'Trj maxdET0':>11} | {'ZeroEq + / -':>14} | {'Flux c1/c2 %':>13}")
    log("-" * 165)
    for _, r in df_master.iterrows():
        c12 = f"{r['flux_c1_neg_pct']:.2f}/{r['flux_c2_mon_pct']:.2f}" if pd.notna(r["flux_c1_neg_pct"]) else "N/A"
        zeq = f"{r['zero_eq_max_pos_dmdt']:+.3f}/{r['zero_eq_min_neg_dmdt']:+.3f}"
        log(f"{r['model']:<42} | {r['traj_dq_all_neg_pct']:11.2f}% | {r['traj_dq_l1_neg_pct']:9.2f}% | {r['traj_dr_all_neg_pct']:8.2f}% | {r['traj_det_pos_pct']:10.2f}% | {r['traj_det_max']:+11.4f} | {zeq:>14} | {c12:>13}")

    log("\nNEURAL PLANT EVAPOTRANSPIRATIVE DEPLETION pc_i DIAGNOSTICS (PROPOSED PINN)")
    log("-" * 90)
    log(f"Training full 24h days (n={len(daily_tr)}): sum(pc_i) mean={daily_tr['pc_sum_mm_d'].mean():.2f} mm/d, IQR=[{daily_tr['pc_sum_mm_d'].quantile(0.25):.2f}, {daily_tr['pc_sum_mm_d'].quantile(0.75):.2f}], range=[{daily_tr['pc_sum_mm_d'].min():.2f}, {daily_tr['pc_sum_mm_d'].max():.2f}] mm/d | ET0 mean={daily_tr['et0_mm_d'].mean():.2f} mm/d")
    log(f"Unseen full 24h days   (n={len(daily_un)}): sum(pc_i) mean={daily_un['pc_sum_mm_d'].mean():.2f} mm/d, IQR=[{daily_un['pc_sum_mm_d'].quantile(0.25):.2f}, {daily_un['pc_sum_mm_d'].quantile(0.75):.2f}], range=[{daily_un['pc_sum_mm_d'].min():.2f}, {daily_un['pc_sum_mm_d'].max():.2f}] mm/d | ET0 mean={daily_un['et0_mm_d'].mean():.2f} mm/d")
    layer_means_tr = {d: daily_tr[f"pc_{d}_mm_d"].mean() for d in DEPTHS}
    tot_tr_mean = sum(layer_means_tr.values())
    shares = ", ".join([f"{d}: {v:.2f} mm/d ({100*v/tot_tr_mean:.1f}%)" for d, v in layer_means_tr.items()])
    log(f"Layer breakdown (Train mean): {shares}")

    log("\nCOUNTERFACTUAL IRRIGATION PERCOLATION BELOW 30 CM (PROPOSED PINN)")
    log("-" * 90)
    for tag_p, st_p, t_p, in_p in [
        ("Training (35d)", data["st_tr_tf"], data["t_tr_tf"], data["in_tr_tf"][:-1]),
        ("Unseen (33d)", data["st_un_tf"], data["t_un_tf"], data["in_un_tf"][:-1]),
    ]:
        p_w = rk4_integration(pinn_full_sys, t_p, st_p[0].numpy(), 1.0, forced_data=in_p).numpy()[:-1]
        in_no = np.array(in_p.numpy(), copy=True)
        in_no[:, 0] = 0.0
        p_n = rk4_integration(pinn_full_sys, t_p, st_p[0].numpy(), 1.0, forced_data=tf.constant(in_no, tf.float64)).numpy()[:-1]
        f3_w = pinn_full.f3_out(pinn_full.f3_dense3(pinn_full.f3_dense2(pinn_full.f3_dense1(tf.constant(p_w[:, 2:3], tf.float64))))).numpy()[:, 0] * NORM["moisture"]
        f3_n = pinn_full.f3_out(pinn_full.f3_dense3(pinn_full.f3_dense2(pinn_full.f3_dense1(tf.constant(p_n[:, 2:3], tf.float64))))).numpy()[:, 0] * NORM["moisture"]
        f34_w = pinn_full.f34_out(pinn_full.f34_dense3(pinn_full.f34_dense2(pinn_full.f34_dense1(tf.constant(p_w[:, 2:4], tf.float64))))).numpy()[:, 0] * NORM["moisture"]
        f34_n = pinn_full.f34_out(pinn_full.f34_dense3(pinn_full.f34_dense2(pinn_full.f34_dense1(tf.constant(p_n[:, 2:4], tf.float64))))).numpy()[:, 0] * NORM["moisture"]
        in_irr = pinn_full.inflow_out(pinn_full.inflow_dense3(pinn_full.inflow_dense2(pinn_full.inflow_dense1(in_p[:, 0:1])))).numpy()[:, 0] * NORM["moisture"]
        tot_irr = float(np.sum(in_irr))
        irr_h = float(np.sum(in_p[:, 0].numpy()))
        grav_d = float(np.sum(f3_w - f3_n))
        net_d = float(np.sum((f3_w + f34_w) - (f3_n + f34_n)))
        log(f"{tag_p:<16}: irr={tot_irr:.2f} mm ({irr_h:.1f} h, ~{tot_irr/irr_h:.2f} mm/h) | grav_below_30cm={grav_d:.2f} mm ({100*grav_d/tot_irr:.1f}%) | net_below_30cm={net_d:.2f} mm ({100*net_d/tot_irr:.1f}%)")
    log(f"Total elapsed time: {(time.time() - t0)/60.0:.2f} min")

    summary_txt = os.path.join(SCRIPT_DIR, "benchmark_ablation_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved master CSV to {master_csv}\nSaved summary TXT to {summary_txt}")


if __name__ == "__main__":
    main()
