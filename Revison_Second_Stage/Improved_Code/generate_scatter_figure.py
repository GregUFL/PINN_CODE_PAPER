#!/usr/bin/env python3
"""
Generate publication-grade Predicted vs. Measured Soil Moisture Scatterplot (R-1.3c)
using Plotly + Kaleido for visual consistency with Figures 1-3.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.keras.backend.set_floatx("float64")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
if not os.path.isdir(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "weights")
OUT_PAPER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Improved_Paper"))
os.makedirs(OUT_PAPER_DIR, exist_ok=True)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tfsolverMIconstrb_float64 import rk4_integration
from pinn_model import TwoTankPINN
from consolidate_master_benchmarks import build_pinn_wrappers, load_all_datasets, NORM, DEPTHS


def main():
    data = load_all_datasets()
    pinn = TwoTankPINN(neurons=56)
    _ = pinn(tf.zeros((1, 6), tf.float64), tf.zeros((1, 3), tf.float64), tf.zeros((1, 1), tf.float64))
    w_file = os.path.join(WEIGHTS_DIR, "pinn_weights_H56_lr7.97e-05.weights.h5")
    pinn.load_weights(w_file)
    _, sys_func = build_pinn_wrappers(pinn)

    pred_tr = (rk4_integration(sys_func, data["t_tr_tf"], data["m0_tr_np"], 1.0, forced_data=data["in_mid_tr"]).numpy() * NORM["moisture"])[:-1]
    meas_tr = data["meas_tr_denorm"]

    pred_val = (rk4_integration(sys_func, data["t_val3d_tf"], data["m0_val3d_np"], 1.0, forced_data=data["in_mid_val3d"]).numpy() * NORM["moisture"])[:-1]
    meas_val = data["meas_val3d_denorm"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Training Period (35 days, N = 5,034)</b>",
            "<b>Held-Out 3-Day Forecast (N = 432)</b>",
        ],
        horizontal_spacing=0.10,
    )

    lims = [1.5, 24.5]
    panels = [
        (1, meas_tr, pred_tr, 0.35, 5, True),
        (2, meas_val, pred_val, 0.78, 7, False),
    ]

    for col, meas, pred, op_val, sz, show_leg in panels:
        for i, d in enumerate(DEPTHS):
            lbl = d.replace("cm", " cm") if isinstance(d, str) else f"{d} cm"
            fig.add_trace(
                go.Scatter(
                    x=meas[:, i], y=pred[:, i],
                    mode="markers",
                    name=lbl,
                    legendgroup=f"d{i}",
                    showlegend=show_leg,
                    marker=dict(color=colors[i], size=sz, opacity=op_val),
                ),
                row=1, col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=lims, y=lims, mode="lines",
                name="1:1 line", legendgroup="ref", showlegend=show_leg,
                line=dict(color="black", width=2, dash="dash"),
            ),
            row=1, col=col,
        )
        m_flat, p_flat = meas.ravel(), pred.ravel()
        rmse = float(np.sqrt(np.mean((m_flat - p_flat) ** 2)))
        r2 = float(1.0 - np.sum((m_flat - p_flat) ** 2) / np.sum((m_flat - np.mean(m_flat)) ** 2))
        r_val, p_val = pearsonr(m_flat, p_flat)
        slope, intercept = np.polyfit(m_flat, p_flat, 1)
        fig.add_trace(
            go.Scatter(
                x=lims, y=[slope * x + intercept for x in lims], mode="lines",
                name="OLS fit", legendgroup="ols", showlegend=show_leg,
                line=dict(color="#333333", width=1.8),
            ),
            row=1, col=col,
        )

        # Stats box annotation anchored to respective subplot domain
        xref = "x domain" if col == 1 else "x2 domain"
        yref = "y domain" if col == 1 else "y2 domain"
        p_txt = "p < 1e-300" if p_val == 0.0 else f"p = {p_val:.2e}"
        sign_str = "+" if intercept >= 0 else "-"
        stats_html = (
            f"Overall RMSE = {rmse:.2f} %VWC<br>"
            f"Overall R<sup>2</sup> = {r2:.3f}<br>"
            f"Pearson r = {r_val:.3f} ({p_txt})<br>"
            f"OLS: y = {slope:.2f}x {sign_str} {abs(intercept):.2f}"
        )
        fig.add_annotation(
            text=stats_html, xref=xref, yref=yref, x=0.04, y=0.96,
            xanchor="left", yanchor="top", showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#999999", borderwidth=1, borderpad=6,
            font=dict(size=14, family="Arial", color="black"),
        )

    for col in (1, 2):
        fig.update_xaxes(
            title_text="Measured Soil Moisture (%VWC)", range=lims,
            showgrid=True, gridcolor="rgba(200,200,200,0.6)",
            linecolor="black", mirror=True, linewidth=1.2,
            tickfont=dict(size=14), title_font=dict(size=16), row=1, col=col,
        )
        fig.update_yaxes(
            title_text="Predicted Soil Moisture (%VWC)" if col == 1 else "", range=lims,
            showgrid=True, gridcolor="rgba(200,200,200,0.6)",
            linecolor="black", mirror=True, linewidth=1.2,
            tickfont=dict(size=14), title_font=dict(size=16), row=1, col=col,
        )

    fig.update_annotations(font_size=17)
    fig.update_layout(
        template="plotly_white", width=1180, height=560,
        margin=dict(l=75, r=30, t=55, b=115),
        font=dict(family="Arial", color="black"),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.17, yanchor="top",
            font=dict(size=26), itemsizing="constant",
        ),
    )

    out_pdf = os.path.join(OUT_PAPER_DIR, "predicted_vs_measured_scatter.pdf")
    fig.write_image(out_pdf, format="pdf", width=1180, height=560, scale=2)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
