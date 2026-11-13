#!/usr/bin/env python3
"""
regenerate_revised_figures.py
=============================
Regenerates revised Figures 2 and 3 in Improved_Paper/ directly from the
untouched PINN checkpoint (pinn_weights_H56_lr7.97e-05.weights.h5) to address:
  - Reviewer 1, Comment 8 (Fig. 2): add dark frame around heatmaps to separate
    white 0-error cells from background, and allocate more vertical space to boxplots.
  - Reviewer 1, Comment 7 & Reviewer 2, Comment 12 (Fig. 3): move concise depth row
    labels ("5 cm" .. "55 cm") cleanly to the left margin away from Column 6,
    apply scientific notation on Lateral outflow y-axis ticks, and improve panel legibility.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.keras.backend.set_floatx("float64")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
if not os.path.isdir(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "weights")
OUT_PAPER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Improved_Paper"))
os.makedirs(OUT_PAPER_DIR, exist_ok=True)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tfsolverMIconstrb_float64 import rk4_integration
from pinn_model import TwoTankPINN
from consolidate_master_benchmarks import build_pinn_wrappers, load_all_datasets, NORM


def generate_fig2(pinn, sys_func, data):
    print("Generating revised Fig. 2 (error_analysis_heatmap_boxplot.pdf)...")
    t_tr, in_tr, m0_tr = data["t_tr_tf"], data["in_mid_tr"], data["m0_tr_np"]
    t_val, in_val, m0_val = data["t_val3d_tf"], data["in_mid_val3d"], data["m0_val3d_np"]

    pred_tr = (rk4_integration(sys_func, t_tr, m0_tr, 1.0, forced_data=in_tr).numpy() * NORM["moisture"])[:-1]
    pred_val = (rk4_integration(sys_func, t_val, m0_val, 1.0, forced_data=in_val).numpy() * NORM["moisture"])[:-1]

    train_errors = data["meas_tr_denorm"] - pred_tr
    val_errors = data["meas_val3d_denorm"] - pred_val

    all_err = np.concatenate([train_errors.ravel(), val_errors.ravel()])
    vmin, vmax = float(all_err.min()), float(all_err.max())
    zero_pos = abs(vmin) / (abs(vmin) + abs(vmax))

    custom_colorscale = [
        [0.0, "rgb(5, 48, 97)"],
        [zero_pos * 0.5, "rgb(33, 102, 172)"],
        [zero_pos, "rgb(255, 255, 255)"],
        [zero_pos + (1.0 - zero_pos) * 0.5, "rgb(239, 138, 98)"],
        [1.0, "rgb(178, 24, 43)"],
    ]

    FONT_SCALE = 3
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.54, 0.46],  # Generous vertical space for boxplots (R-1.8)
        subplot_titles=["<b>Training</b>", "<b>Forecast</b>", "", ""],
        column_widths=[0.92, 0.08],
        horizontal_spacing=0.02,
        vertical_spacing=0.14,
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [{"type": "box"}, {"type": "box"}]],
    )

    time_tr = data["df_tr"]["timestamp"].to_numpy()[:-1]
    time_val = data["df_val3d"]["timestamp"].to_numpy()[:-1]

    fig.add_trace(
        go.Heatmap(
            z=train_errors.T, x=time_tr, y=[1, 2, 3, 4, 5, 6],
            colorscale=custom_colorscale, zmin=vmin, zmax=vmax,
            colorbar=dict(
                tickfont=dict(size=int(22 * FONT_SCALE)),
                len=0.52, thickness=30, x=1.01, y=0.75, yanchor="middle",
                tickvals=[vmin, 0, vmax], ticktext=[f"{vmin:.1f}", "<b>0</b>", f"{vmax:.1f}"],
                outlinewidth=2, outlinecolor="black",
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=val_errors.T, x=time_val, y=[1, 2, 3, 4, 5, 6],
            colorscale=custom_colorscale, zmin=vmin, zmax=vmax, showscale=False,
        ),
        row=1, col=2,
    )

    layer_names = ["5cm", "15cm", "25cm", "35cm", "45cm", "55cm"]
    for i in range(6):
        fig.add_trace(
            go.Box(
                y=train_errors[:, i], name=layer_names[i],
                marker=dict(color="rgba(31,119,180,0.7)", size=8),
                boxmean="sd", line=dict(width=2.5, color="rgb(31,119,180)"),
                fillcolor="rgba(31,119,180,0.5)", showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Box(
                y=val_errors[:, i], name=layer_names[i],
                marker=dict(color="rgba(255,127,14,0.7)", size=8),
                boxmean="sd", line=dict(width=2.5, color="rgb(255,127,14)"),
                fillcolor="rgba(255,127,14,0.5)", showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        template="plotly_white", height=1850, width=4800,
        font=dict(family="Arial", size=int(28 * FONT_SCALE)),
        margin=dict(l=200, r=150, t=120, b=110), bargap=0,
    )
    fig.update_annotations(font=dict(size=int(35 * FONT_SCALE), family="Arial"))

    # Dark frame around heatmaps to separate white 0-error region from page background (R-1.8)
    frame_style = dict(showline=True, linewidth=3, linecolor="black", mirror=True)
    fig.update_yaxes(
        title=dict(text="Error (%VWC)", font=dict(size=int(28 * FONT_SCALE))),
        tickvals=np.arange(1, 7), ticktext=layer_names, autorange="reversed",
        tickfont=dict(size=int(24 * FONT_SCALE)), row=1, col=1, **frame_style,
    )
    fig.update_yaxes(
        title=None, tickvals=np.arange(1, 7), showticklabels=False,
        autorange="reversed", row=1, col=2, **frame_style,
    )
    fig.update_xaxes(tickformat="%b %d", tickfont=dict(size=int(24 * FONT_SCALE)), row=1, col=1, **frame_style)
    fig.update_xaxes(tickformat="%b %d", tickfont=dict(size=int(24 * FONT_SCALE)), nticks=1, row=1, col=2, **frame_style)

    label_vals = np.arange(np.floor(vmin / 5.0) * 5.0, np.ceil(vmax / 5.0) * 5.0 + 5.0, 5.0)
    grid_vals = np.arange(np.floor(vmin / 2.5) * 2.5, np.ceil(vmax / 2.5) * 2.5 + 2.5, 2.5)

    for col in [1, 2]:
        fig.update_yaxes(
            title=dict(text="Distribution", font=dict(size=int(20 * FONT_SCALE))) if col == 1 else None,
            range=[vmin - 0.5, vmax + 0.5],
            tickfont=dict(size=int(24 * FONT_SCALE)), showticklabels=(col == 1),
            tickmode="array", tickvals=label_vals, showgrid=True, gridwidth=1, gridcolor="lightgray",
            zeroline=True, zerolinewidth=3, zerolinecolor="black",
            minor=dict(tickmode="array", tickvals=grid_vals, showgrid=True, gridwidth=1, gridcolor="rgba(211,211,211,0.5)"),
            row=2, col=col,
        )
        fig.update_xaxes(tickvals=list(range(6)), ticktext=layer_names, tickfont=dict(size=int(20 * FONT_SCALE)), row=2, col=col)

    out_pdf = os.path.join(OUT_PAPER_DIR, "error_analysis_heatmap_boxplot.pdf")
    fig.write_image(out_pdf, format="pdf", width=4800, height=1850, scale=2)
    print(f"  Saved {out_pdf}")


def generate_fig3(pinn, data):
    print("Generating revised Fig. 3 (pinn_functions_vertical_by_layer.pdf)...")
    m_r = np.linspace(0, 1, 100).reshape(-1, 1)
    q_r = np.linspace(0, 1, 100).reshape(-1, 1)
    r_r = np.linspace(0, 1, 100).reshape(-1, 1)
    e_r = np.linspace(0, 1, 100).reshape(-1, 1)
    mg, eg = np.meshgrid(m_r, e_r)
    m1g, m2g = np.meshgrid(m_r, m_r)

    f_out = [getattr(pinn, f"f{i+1}_out")(getattr(pinn, f"f{i+1}_dense3")(getattr(pinn, f"f{i+1}_dense2")(getattr(pinn, f"f{i+1}_dense1")(m_r)))).numpy().flatten() for i in range(6)]
    inf_out = pinn.inflow_out(pinn.inflow_dense3(pinn.inflow_dense2(pinn.inflow_dense1(q_r)))).numpy().flatten()
    ir_out = [getattr(pinn, f"rain{i+1}_out")(getattr(pinn, f"rain{i+1}_dense3")(getattr(pinn, f"rain{i+1}_dense2")(getattr(pinn, f"rain{i+1}_dense1")(r_r)))).numpy().flatten() for i in range(6)]

    f_inter = []
    for flw in ["f12", "f23", "f34", "f45", "f56"]:
        inp = np.stack([m1g.ravel(), m2g.ravel()], axis=1)
        x = getattr(pinn, f"{flw}_dense1")(inp)
        x = getattr(pinn, f"{flw}_dense2")(x)
        x = getattr(pinn, f"{flw}_dense3")(x)
        f_inter.append(getattr(pinn, f"{flw}_out")(x).numpy().reshape(m1g.shape))

    pc_grids = []
    for i in range(1, 7):
        inp = np.stack([mg.ravel(), eg.ravel()], axis=1)
        x = getattr(pinn, f"pc{i}_dense1")(inp)
        for j in range(2, 5):
            x = getattr(pinn, f"pc{i}_dense{j}")(x)
        pc_grids.append(getattr(pinn, f"pc{i}_out")(x).numpy().reshape(mg.shape))

    m_r_tf = tf.constant(m_r, dtype=tf.float64)
    fl_out = []
    for i in range(1, 6):
        x = getattr(pinn, f"fl{i}_dense1")(m_r_tf)
        x = getattr(pinn, f"fl{i}_dense2")(x)
        x = getattr(pinn, f"fl{i}_dense3")(x)
        fl_out.append(getattr(pinn, f"fl{i}_out")(x).numpy().flatten())

    df_tr = data["df_tr"]
    bounds = {k: (float(df_tr[f"moisture_{k}"].min()), float(df_tr[f"moisture_{k}"].max())) for k in range(1, 7)}

    FONT_SCALE = 2.8
    specs = [[{"type": "xy"}, {"type": "xy"}, {"type": "contour"}, {"type": "contour"}, {"type": "xy"}, {"type": "xy"}]]
    for _ in range(4):
        specs.append([{"type": "xy"}, {"type": "xy"}, {"type": "contour"}, {"type": "contour"}, {"type": "xy"}, None])
    specs.append([{"type": "xy"}, {"type": "xy"}, None, {"type": "contour"}, None, None])

    fig = make_subplots(
        rows=6, cols=6,
        row_heights=[0.17, 0.17, 0.17, 0.17, 0.17, 0.15],
        column_widths=[0.15, 0.15, 0.20, 0.20, 0.15, 0.15],
        subplot_titles=[""] * 36, specs=specs,
        horizontal_spacing=0.065, vertical_spacing=0.042,
    )

    zmin_inter = min(np.min(40 * g) for g in f_inter)
    zmax_inter = max(np.max(40 * g) for g in f_inter)
    zmin_pc = min(np.min(40 * g) for g in pc_grids)
    zmax_pc = max(np.max(40 * g) for g in pc_grids)

    m1d, r1d, q1d, e1d = m_r.squeeze(), r_r.squeeze(), q_r.squeeze(), e_r.squeeze()
    for layer in range(1, 7):
        row = layer
        fig.add_trace(go.Scatter(x=30 * r1d, y=40 * ir_out[layer - 1], mode="lines", line=dict(color="blue", width=4), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=40 * m1d, y=40 * f_out[layer - 1], mode="lines", line=dict(color="blue", width=4), showlegend=False), row=row, col=2)
        vmin, vmax = bounds[layer]
        fig.add_vrect(x0=vmin, x1=vmax, row=row, col=2, fillcolor="rgba(0,150,200,0.15)", line_width=0, layer="below")

        if layer < 6:
            fig.add_trace(
                go.Contour(
                    x=40 * m1d, y=40 * m1d, z=40 * f_inter[layer - 1], coloraxis="coloraxis1",
                    contours=dict(start=zmin_inter, end=zmax_inter, size=(zmax_inter - zmin_inter) / 30, showlines=False),
                    showscale=(layer == 5),
                ),
                row=row, col=3,
            )
            x0, x1 = bounds[layer]
            y0, y1 = bounds[layer + 1]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor="rgba(0,150,200,0.5)", line_width=0, layer="above", row=row, col=3)

        fig.add_trace(
            go.Contour(
                x=40 * m1d, y=0.8 * e1d, z=40 * pc_grids[layer - 1], coloraxis="coloraxis2",
                contours=dict(start=zmin_pc, end=zmax_pc, size=(zmax_pc - zmin_pc) / 30, showlines=False),
                showscale=(layer == 6),
            ),
            row=row, col=4,
        )
        fig.add_shape(type="rect", x0=vmin, x1=vmax, y0=0.0, y1=0.8, fillcolor="rgba(0,150,200,0.5)", line_width=0, layer="above", row=row, col=4)

        if layer <= 5:
            fig.add_trace(go.Scatter(x=40 * m1d, y=40 * fl_out[layer - 1], mode="lines", line=dict(color="blue", width=4), showlegend=False), row=row, col=5)
            fig.add_vrect(x0=vmin, x1=vmax, row=row, col=5, fillcolor="rgba(0,150,200,0.15)", line_width=0, layer="below")

    fig.add_trace(go.Scatter(x=q1d, y=40 * inf_out, mode="lines", line=dict(color="blue", width=4), showlegend=False), row=1, col=6)

    fig.update_layout(
        coloraxis1=dict(
            colorscale="Viridis", cmin=zmin_inter, cmax=zmax_inter,
            colorbar=dict(
                title="Inter-layer flow (%VWC/h)", orientation="h", x=0.40, xanchor="center",
                y=0.09, yanchor="top", len=0.16, thickness=14,
                tickfont=dict(size=int(9 * FONT_SCALE)), title_font=dict(size=int(10 * FONT_SCALE)), title_side="bottom",
            ),
        ),
        coloraxis2=dict(
            colorscale="Viridis", cmin=zmin_pc, cmax=zmax_pc,
            colorbar=dict(
                title="Plant uptake (%VWC/h)", orientation="h", x=0.60, xanchor="center",
                y=-0.04, yanchor="top", len=0.16, thickness=14,
                tickfont=dict(size=int(9 * FONT_SCALE)), title_font=dict(size=int(10 * FONT_SCALE)), title_side="bottom",
            ),
        ),
    )

    ax_t = dict(tickfont=dict(size=int(9.5 * FONT_SCALE)), title_font=dict(size=int(10.5 * FONT_SCALE)), title_standoff=4, showticklabels=True)
    for r in range(1, 7):
        fig.update_xaxes(title_text="Rain (mm/h)", row=r, col=1, **ax_t)
        fig.update_yaxes(title_text=f"fr<sub>{r}</sub> (%VWC/h)", row=r, col=1, **ax_t)
        fig.update_xaxes(title_text=f"m<sub>{r}</sub> (%VWC)", row=r, col=2, **ax_t)
        fig.update_yaxes(title_text=f"f<sub>{r}</sub> (%VWC/h)", row=r, col=2, **ax_t)
        fig.update_xaxes(title_text=f"m<sub>{r}</sub> (%VWC)", row=r, col=4, **ax_t)
        fig.update_yaxes(title_text="ET<sub>0</sub> (mm/h)", row=r, col=4, **ax_t)
        if r < 6:
            fig.update_xaxes(title_text=f"m<sub>{r}</sub> (%VWC)", row=r, col=3, **ax_t)
            fig.update_yaxes(title_text=f"m<sub>{r+1}</sub> (%VWC)", row=r, col=3, **ax_t)
            fig.update_xaxes(title_text=f"m<sub>{r}</sub> (%VWC)", row=r, col=5, **ax_t)
            # Scientific notation on Lateral outflow y-axis to save horizontal margin (R-1.7)
            fig.update_yaxes(title_text=f"fl<sub>{r}</sub> (%VWC/h)", tickformat=".1e", exponentformat="e", nticks=4, row=r, col=5, **ax_t)

    fig.update_xaxes(title_text="Irrigation Time (h/h)", row=1, col=6, **ax_t)
    fig.update_yaxes(title_text="Inflow (%VWC/h)", row=1, col=6, **ax_t)

    col_headers = ["Rain", "Vertical<br>flow", "Inter-layer<br>flow", "Plant<br>uptake", "Lateral<br>outflow", "Irrigation"]
    col_pos = [0.06, 0.23, 0.40, 0.60, 0.785, 0.945]
    for lbl, xp in zip(col_headers, col_pos):
        fig.add_annotation(text=f"<b>{lbl}</b>", xref="paper", yref="paper", x=xp, y=1.015,
                           xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=int(13 * FONT_SCALE), family="Arial"))

    # Left-side concise depth labels for all 6 rows (R-1.7)
    depth_lbls = ["5 cm", "15 cm", "25 cm", "35 cm", "45 cm", "55 cm"]
    y_centers = [0.925, 0.750, 0.575, 0.400, 0.225, 0.060]
    for lbl, yp in zip(depth_lbls, y_centers):
        fig.add_annotation(text=f"<b>{lbl}</b>", xref="paper", yref="paper", x=-0.062, y=yp,
                           xanchor="right", yanchor="middle", showarrow=False, font=dict(size=int(13 * FONT_SCALE), family="Arial"))

    fig.update_yaxes(automargin=True)
    fig.update_layout(
        template="plotly_white", width=3600, height=2400,
        margin=dict(l=450, r=80, t=190, b=170),
        font=dict(family="Arial", size=int(13 * FONT_SCALE)), showlegend=False,
    )

    out_pdf = os.path.join(OUT_PAPER_DIR, "pinn_functions_vertical_by_layer.pdf")
    fig.write_image(out_pdf, format="pdf", width=3600, height=2400, scale=2)
    print(f"  Saved {out_pdf}")


def main():
    data = load_all_datasets()
    pinn = TwoTankPINN(neurons=56)
    _ = pinn(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64), tf.zeros((1, 1), tf.float64))
    w_file = os.path.join(WEIGHTS_DIR, "pinn_weights_H56_lr7.97e-05.weights.h5")
    pinn.load_weights(w_file)
    _, sys_func = build_pinn_wrappers(pinn)
    generate_fig2(pinn, sys_func, data)
    generate_fig3(pinn, data)


if __name__ == "__main__":
    main()
