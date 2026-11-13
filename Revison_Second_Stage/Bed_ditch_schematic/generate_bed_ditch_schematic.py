#!/usr/bin/env python3
"""
generate_bed_ditch_schematic.py
===============================
Publication-grade Nature Portfolio schematic generator using:
  HTML5 + CSS Grid + Inline SVG + KaTeX + Headless Google Chrome (Playwright)

Layout: Clean 2-panel side-by-side vector infographic (1280 x 580 px):
  - Panel a (710px card, 686x504 stage): Double-row raised-bed & drainage-ditch cross-section
  - Panel b (534px card, 510x462 stage): 1D finite-volume PINN model (Eq. 8)

Generates:
  - bed_ditch_schematic.html
  - bed_ditch_schematic_improved.pdf    (crisp vector PDF)
  - bed_ditch_schematic_improved_2.pdf  (synchronized copy)
  - bed_ditch_schematic_improved.png    (300 DPI raster preview)
  - Synced to Improved_Paper/bed_ditch_schematic.pdf
"""

import os
import shutil
import numpy as np
from playwright.sync_api import sync_playwright

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAGE_W = 1280
PAGE_H = 580
DPI_SCALE = 300.0 / 96.0  # 3.125 -> 4000 x 1812 px @ 300 DPI


def build_botanical_tree_svg(cx, ground_y, seed=42, scale=1.22, rx_left_scale=1.0, rx_right_scale=1.0):
    """
    Generate realistic botanical SVG citrus tree using:
      - Smooth tapered Bezier woody trunk & low scaffold limbs (#4A3728)
      - Deep under-skirt foliage shadow path (#1B4332)
      - Dense instantiations of <use href="#leaf-cluster"> (14 pointed elliptic
        citrus leaves per rosette with linearGradient shading).
      - Subtle muted-gold citrus fruit accents (#D99B26) tucked inside leaves.
    """
    rng = np.random.default_rng(seed)
    parts = []

    # Ground shadow beneath low skirt
    rx = 66.0 * scale
    parts.append(
        f'<ellipse cx="{cx:.1f}" cy="{ground_y + 1.5:.1f}" rx="{rx * 0.90:.1f}" '
        f'ry="{4.0 * scale:.1f}" fill="#D6CFC2" opacity="0.55"/>'
    )

    # 1. Tapered woody trunk splitting low (~22% of total tree height) & scaffold limbs
    trunk_d = (
        f"M {cx - 5.0*scale:.1f},{ground_y:.1f} "
        f"Q {cx - 3.8*scale:.1f},{ground_y - 14*scale:.1f} {cx - 3.6*scale:.1f},{ground_y - 24*scale:.1f} "
        f"Q {cx - 13*scale:.1f},{ground_y - 38*scale:.1f} {cx - 26*scale:.1f},{ground_y - 48*scale:.1f} "
        f"L {cx - 22*scale:.1f},{ground_y - 51*scale:.1f} "
        f"Q {cx - 9*scale:.1f},{ground_y - 40*scale:.1f} {cx - 1.4*scale:.1f},{ground_y - 28*scale:.1f} "
        f"L {cx + 1.4*scale:.1f},{ground_y - 28*scale:.1f} "
        f"Q {cx + 10*scale:.1f},{ground_y - 40*scale:.1f} {cx + 24*scale:.1f},{ground_y - 50*scale:.1f} "
        f"L {cx + 28*scale:.1f},{ground_y - 47*scale:.1f} "
        f"Q {cx + 14*scale:.1f},{ground_y - 37*scale:.1f} {cx + 3.6*scale:.1f},{ground_y - 24*scale:.1f} "
        f"Q {cx + 3.8*scale:.1f},{ground_y - 14*scale:.1f} {cx + 5.2*scale:.1f},{ground_y:.1f} Z"
    )
    parts.append(f'<path d="{trunk_d}" fill="#4A3728" stroke="#36261B" stroke-width="0.6"/>')

    # 2. Deep evergreen scalloped base silhouette (#1B4332) for solid botanical mass
    cy = ground_y - 52.0 * scale
    ry_top = 45.0 * scale
    ry_bot = 22.0 * scale
    pts = []
    for k in range(72):
        th = 2 * np.pi * k / 72.0
        mod = 1.0 + 0.055 * np.cos(7 * th + 0.4) + 0.03 * np.sin(11 * th)
        rx_k = rx * (rx_left_scale if np.cos(th) < 0 else rx_right_scale)
        px = cx + rx_k * mod * np.cos(th)
        py = cy - (ry_top if np.sin(th) >= 0 else ry_bot) * mod * np.sin(th)
        pts.append(f"{px:.1f},{py:.1f}")
    parts.append(
        f'<polygon points="{" ".join(pts)}" fill="#1B4332" opacity="0.95"/>'
    )

    # 3. Instantiated high-density botanical leaf-rosette clusters (<use href="#leaf-cluster">)
    clusters = []
    for _ in range(80):
        u = np.sqrt(rng.uniform(0.03, 0.96))
        ang = rng.uniform(-0.18 * np.pi, 1.18 * np.pi)
        rx_ang = rx * (rx_left_scale if np.cos(ang) < 0 else rx_right_scale)
        px = cx + u * 0.88 * rx_ang * np.cos(ang)
        py = cy - u * (0.88 * ry_top if np.sin(ang) >= 0 else 0.78 * ry_bot) * np.sin(ang)
        if py > ground_y - 23.0 * scale:
            py = ground_y - 23.0 * scale - rng.uniform(0, 5.0 * scale)
        sc = rng.uniform(0.68, 1.12) * scale
        rot = rng.uniform(0, 360)
        h_norm = ((ground_y - 23 * scale) - py) / (ry_top + ry_bot)
        sun = h_norm - 0.14 * ((px - cx) / rx) + rng.uniform(-0.15, 0.15)
        if sun < 0.30:
            href = "#leaf-cluster-dark"
            op = 0.90
        elif sun < 0.65:
            href = "#leaf-cluster-mid"
            op = 0.92
        else:
            href = "#leaf-cluster-sun"
            op = 0.88
        clusters.append((py, f'<use href="{href}" transform="translate({px:.1f},{py:.1f}) rotate({rot:.1f}) scale({sc:.2f})" opacity="{op:.2f}"/>'))

    clusters.sort(key=lambda item: item[0], reverse=True)
    parts.extend(c[1] for c in clusters)

    # 4. Subtle muted-gold grapefruit accents tucked inside foliage
    for _ in range(5):
        fx = cx + rng.uniform(-0.52 * rx, 0.52 * rx)
        fy = cy - rng.uniform(-0.18 * ry_bot, 0.38 * ry_top)
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="{1.8 * scale:.2f}" '
            f'fill="#D99B26" opacity="0.88"/>'
        )

    return "\n        ".join(parts)


def build_fibrous_roots_svg(cx, ground_y, max_depth_y, x_min=None, x_max=None, seed=88):
    """Smooth tapered Bezier SVG fibrous roots strictly within 0-30 cm above clay horizon."""
    rng = np.random.default_rng(seed)
    paths = []

    def grow(x0, y0, ang, length, depth, sw):
        ctrl_ang = ang + rng.uniform(-0.16, 0.16)
        x1 = x0 + length * np.cos(ang)
        if x_min is not None and x1 < x_min:
            x1 = x_min + 1.0
        if x_max is not None and x1 > x_max:
            x1 = x_max - 1.0
        y1 = min(y0 + length * np.sin(ang), max_depth_y - 2.0)
        xm = x0 + 0.55 * length * np.cos(ctrl_ang)
        if x_min is not None and xm < x_min:
            xm = x_min + 1.0
        if x_max is not None and xm > x_max:
            xm = x_max - 1.0
        ym = min(y0 + 0.55 * length * np.sin(ctrl_ang), max_depth_y - 2.0)
        paths.append(
            f'<path d="M {x0:.1f},{y0:.1f} Q {xm:.1f},{ym:.1f} {x1:.1f},{y1:.1f}" '
            f'fill="none" stroke="#6B4C35" stroke-width="{sw:.2f}" stroke-linecap="round" opacity="0.88"/>'
        )
        if depth >= 4 or y1 >= max_depth_y - 4.0:
            return
        spread = np.deg2rad(24 - 2 * depth)
        for da in (-spread, spread):
            na = ang + da + rng.uniform(-0.12, 0.12)
            if y1 > ground_y + 0.62 * (max_depth_y - ground_y) and np.sin(na) > 0.35:
                na = 0.22 if np.cos(na) >= 0 else np.pi - 0.22
            grow(x1, y1, na, length * rng.uniform(0.64, 0.76), depth + 1, max(0.38, sw * 0.65))

    for deg, L0, sw0 in [(142, 26, 2.2), (120, 25, 2.0), (92, 26, 2.2), (68, 25, 2.0), (44, 25, 1.8)]:
        grow(cx, ground_y + 0.5, np.deg2rad(deg), L0, 0, sw0)

    x_lo = x_min if x_min is not None else cx - 44
    x_hi = x_max if x_max is not None else cx + 44
    for _ in range(22):
        rx = rng.uniform(x_lo + 4, x_hi - 4)
        ry = ground_y + rng.uniform(16, 52)
        dx = rng.uniform(-8, 8)
        dy = rng.uniform(-3, 4)
        paths.append(
            f'<path d="M {rx:.1f},{ry:.1f} Q {rx + 0.5*dx:.1f},{ry + dy - 1.5:.1f} {min(max(rx + dx, x_lo + 1), x_hi - 1):.1f},{min(ry + dy, max_depth_y - 2):.1f}" '
            f'fill="none" stroke="#7C5A40" stroke-width="0.52" stroke-linecap="round" opacity="0.70"/>'
        )

    return "\n        ".join(paths)


def build_html():
    # -------------------------------------------------------------------------
    # PANEL a SVG GEOMETRY (Exact 686 x 504 px stage)
    # -------------------------------------------------------------------------
    # Vertical Soil Scale (2 px = 1 cm depth):
    # Raised-bed crown surface (z = 0 cm): y_surf = 260.0
    # Argillic clay horizon (z ≈ 30 cm): y_clay = 320.0 (60 px below surface)
    # Ditch standing water level (z ≈ 44 cm): y_water = 348.0
    # Ditch floor / invert (z ≈ 52 cm): y_ditch = 364.0 (104 px below bed crown)
    # Sentek probe tip (z = 60 cm): y_probe_bot = 380.0 (120 px total probe length)
    y_surf = 260.0
    y_clay = 320.0
    y_water = 348.0
    y_ditch = 364.0
    y_probe_bot = 380.0

    # Larger, symmetric, lush 6-year-old grapefruit trees (scale = 1.22, symmetric)
    tree1_svg = build_botanical_tree_svg(232.0, y_surf, seed=42, scale=1.22, rx_left_scale=1.0, rx_right_scale=1.0)
    roots1_svg = build_fibrous_roots_svg(232.0, y_surf, y_clay, x_min=190.0, x_max=274.0, seed=51)

    tree2_svg = build_botanical_tree_svg(454.0, y_surf, seed=77, scale=1.22, rx_left_scale=1.0, rx_right_scale=1.0)
    roots2_svg = build_fibrous_roots_svg(454.0, y_surf, y_clay, x_min=412.0, x_max=484.0, seed=88)

    # Gentle Swale & Bed Contour across 0..686 px
    bed_surface_d = (
        f"M 0,312 C 14,312 26,{y_ditch:.0f} 44,{y_ditch:.0f} L 82,{y_ditch:.0f} "
        f"C 112,{y_ditch:.0f} 132,{y_surf:.0f} 162,{y_surf:.0f} "
        f"L 524,{y_surf:.0f} "
        f"C 554,{y_surf:.0f} 574,{y_ditch:.0f} 604,{y_ditch:.0f} L 642,{y_ditch:.0f} "
        f"C 660,{y_ditch:.0f} 672,312 686,312"
    )
    topsoil_fill_d = bed_surface_d + " L 686,504 L 0,504 Z"

    # Ditch water body polygons (strictly inside the ditch depression below y_water=348)
    # Left ditch water: x = 25..99 at y_water = 348, floor x = 44..82 at y_ditch = 364
    left_water_poly_d = (
        f"M 25,{y_water:.0f} L 99,{y_water:.0f} "
        f"C 93,354 87,{y_ditch:.0f} 82,{y_ditch:.0f} L 44,{y_ditch:.0f} "
        f"C 39,{y_ditch:.0f} 32,354 25,{y_water:.0f} Z"
    )
    # Right ditch water: x = 587..661 at y_water = 348, floor x = 604..642 at y_ditch = 364
    right_water_poly_d = (
        f"M 587,{y_water:.0f} L 661,{y_water:.0f} "
        f"C 654,354 647,{y_ditch:.0f} 642,{y_ditch:.0f} L 604,{y_ditch:.0f} "
        f"C 599,{y_ditch:.0f} 593,354 587,{y_water:.0f} Z"
    )

    # Smooth undulating argillic clay horizon (z ≈ 30 cm, y ≈ 320)
    clay_xs = np.linspace(0, 686, 120)
    clay_ys = y_clay + 2.0 * np.sin(0.024 * clay_xs + 0.4) + 1.0 * np.cos(0.05 * clay_xs)
    clay_line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(clay_xs, clay_ys))
    clay_fill_d = f"M 0,504 L " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(clay_xs, clay_ys)) + " L 686,504 Z"

    # Local water table / capillary fringe curve (y = 360 at ditches, arching to 352 under bed center 343)
    wt_xs = np.linspace(46, 640, 90)
    wt_ys = 360.0 - 8.0 * np.exp(-((wt_xs - 343.0) / 150.0) ** 2)
    wt_line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(wt_xs, wt_ys))
    wt_band_d = (
        "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(wt_xs, wt_ys)) +
        " L " + " L ".join(f"{x:.1f},{y + 8.0:.1f}" for x, y in zip(wt_xs[::-1], wt_ys[::-1])) + " Z"
    )

    # Turf ribbons
    grass_left_d = f"M 95,356 C 112,{y_ditch:.0f} 132,{y_surf:.0f} 174,{y_surf:.0f}"
    grass_mid_d = f"M 296,{y_surf:.0f} L 390,{y_surf:.0f}"
    grass_right_d = f"M 512,{y_surf:.0f} C 554,{y_surf:.0f} 574,{y_ditch:.0f} 591,356"

    # Lateral drainage arrows along left and right bed slopes (z = 5, 15, 25, 35, 45 cm)
    # Soil scale: 2 px = 1 cm depth. Sensor depths: 270, 290, 310, 330, 350, 370
    sensor_zs_a = [270.0, 290.0, 310.0, 330.0, 350.0, 370.0]
    lat_arrows_a = []
    for z_cm in (5, 15, 25, 35, 45):
        yy = y_surf + z_cm * 2.0
        frac = (yy - y_surf) / (y_ditch - y_surf)
        xl = 162.0 - 80.0 * frac
        xr = 524.0 + 80.0 * frac
        lat_arrows_a.append(
            f'<line x1="{xl:.1f}" y1="{yy:.1f}" x2="{xl - 24:.1f}" y2="{yy:.1f}" '
            f'stroke="#0D9488" stroke-width="1.6" marker-end="url(#arr-teal)"/>'
        )
        lat_arrows_a.append(
            f'<line x1="{xr:.1f}" y1="{yy:.1f}" x2="{xr + 24:.1f}" y2="{yy:.1f}" '
            f'stroke="#0D9488" stroke-width="1.6" marker-end="url(#arr-teal)"/>'
        )

    # -------------------------------------------------------------------------
    # PANEL b SVG GEOMETRY (Exact 510 x 462 px stage)
    # -------------------------------------------------------------------------
    c_y0 = 46.0
    c_dz = 62.0
    c_x0, c_x1 = 104.0, 222.0
    c_xc = 163.0

    c_compartments = []
    c_labels_html = []
    for idx, z_mid in enumerate((5, 15, 25, 35, 45, 55), start=1):
        yt = c_y0 + (idx - 1) * c_dz
        ym = yt + 0.5 * c_dz
        fill_id = "url(#c-sand-grad)" if idx <= 3 else "url(#c-clay-grad)"
        c_compartments.append(
            f'<rect x="{c_x0}" y="{yt:.1f}" width="{c_x1 - c_x0}" height="{c_dz:.1f}" '
            f'fill="{fill_id}" stroke="#94A3B8" stroke-width="0.9"/>'
        )
        if idx == 6:
            c_compartments.append(
                f'<rect x="{c_x0 + 1}" y="{yt + 1:.1f}" width="{c_x1 - c_x0 - 2}" height="{c_dz - 2:.1f}" '
                f'fill="rgba(56,189,248,0.14)"/>'
            )
        # Red Sentek capacitive ring at left edge
        c_compartments.append(
            f'<rect x="{c_x0 + 3}" y="{ym - 6:.1f}" width="7" height="12" rx="1.5" '
            f'fill="#DC2626" stroke="#7F1D1D" stroke-width="0.6"/>'
        )
        # Centered compartment state
        c_labels_html.append(
            f'<div class="math-center" style="left:{c_xc:.1f}px; top:{ym:.1f}px;">'
            f'<span class="k" data-tex="z_{idx}={z_mid}\\,\\text{{cm}}\\;(m_{idx})"></span></div>'
        )

    # Vertical interface fluxes (gravity f_i down, matric f_{i,i+1} bidirectional)
    c_flux_arrows = []
    for i in range(1, 6):
        yi = c_y0 + i * c_dz
        c_flux_arrows.append(
            f'<line x1="132" y1="{yi - 8:.1f}" x2="132" y2="{yi + 8:.1f}" '
            f'stroke="#1D4ED8" stroke-width="1.8" marker-end="url(#arr-blue)"/>'
        )
        c_labels_html.append(
            f'<div class="math-tag c-blue" style="left:142px; top:{yi - 7:.1f}px;">'
            f'<span class="k" data-tex="f_{i}"></span></div>'
        )
        c_flux_arrows.append(
            f'<line x1="194" y1="{yi - 8:.1f}" x2="194" y2="{yi + 8:.1f}" '
            f'stroke="#6D28D9" stroke-width="1.6" stroke-dasharray="3 2" '
            f'marker-start="url(#arr-violet-up)" marker-end="url(#arr-violet-dn)"/>'
        )
        c_labels_html.append(
            f'<div class="math-tag c-violet" style="left:180px; top:{yi - 7:.1f}px;">'
            f'<span class="k" data-tex="f_{{{i},{i+1}}}"></span></div>'
        )

    # Deep percolation f_6(m_6) at z = 60 cm
    y_bot = c_y0 + 6 * c_dz
    c_flux_arrows.append(
        f'<line x1="144" y1="{y_bot:.1f}" x2="144" y2="{y_bot + 28:.1f}" '
        f'stroke="#1D4ED8" stroke-width="2.0" marker-end="url(#arr-blue)"/>'
    )
    c_labels_html.append(
        f'<div class="math-left c-blue" style="left:154px; top:{y_bot + 14:.1f}px;">'
        f'<span class="k" data-tex="f_6(m_6)"></span> <span class="subnote">(deep percolation)</span></div>'
    )

    # Infiltration into deeper layers fr_2..fr_6
    # Shift fr_3 arrow up by 6px (ym - 12) and fr_4 arrow down by 6px (ym + 4)
    # so there is generous vertical space around z = 30 cm.
    for i in range(2, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        if i == 3:
            y_start = ym - 12.0
            y_end = ym - 4.0
            y_lbl = ym - 14.0
        elif i == 4:
            y_start = ym
            y_end = ym + 8.0
            y_lbl = ym - 2.0
        else:
            y_start = ym - 6.0
            y_end = ym + 2.0
            y_lbl = ym - 8.0

        c_flux_arrows.append(
            f'<line x1="68" y1="{y_start:.1f}" x2="102" y2="{y_end:.1f}" '
            f'stroke="#1D4ED8" stroke-width="1.5" marker-end="url(#arr-blue)"/>'
        )
        c_labels_html.append(
            f'<div class="math-right c-blue" style="left:64px; top:{y_lbl:.1f}px;">'
            f'<span class="k" data-tex="fr_{i}"></span></div>'
        )

    # Lateral outflows fl_i and apparent ET0 sinks S_i
    for i in range(1, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        y_fl = ym - 12.0
        y_et = ym + 12.0
        if i <= 5:
            c_flux_arrows.append(
                f'<line x1="{c_x1}" y1="{y_fl:.1f}" x2="254" y2="{y_fl:.1f}" '
                f'stroke="#0D9488" stroke-width="1.6" marker-end="url(#arr-teal)"/>'
            )
            c_labels_html.append(
                f'<div class="math-left c-teal" style="left:259px; top:{y_fl:.1f}px;">'
                f'<span class="k" data-tex="fl_{i}"></span></div>'
            )
        else:
            c_labels_html.append(
                f'<div class="math-left c-muted" style="left:232px; top:{y_fl - 1:.1f}px; font-size:10.6px;">'
                f'<span class="k" data-tex="fl_6=0"></span></div>'
            )

        c_flux_arrows.append(
            f'<line x1="{c_x1}" y1="{y_et:.1f}" x2="298" y2="{y_et:.1f}" '
            f'stroke="#D97706" stroke-width="1.7" marker-end="url(#arr-amber)"/>'
        )
        c_labels_html.append(
            f'<div class="math-left c-amber" style="left:232px; top:{y_et - 8:.1f}px; font-size:11.2px;">'
            f'<span class="k" data-tex="S_{i}"></span></div>'
        )

    # Right-hand physical regime brackets (wrapped to avoid right margin overflow)
    regimes = [
        (c_y0 + 10,  c_y0 + 52,  "Surface soil<br/>evaporation"),
        (c_y0 + 72,  c_y0 + 176, "Shallow root-zone uptake<br/><span class='subnote'>(peaks at 25 cm)</span>"),
        (c_y0 + 196, c_y0 + 300, "Restricted clay-layer<br/>depletion"),
        (c_y0 + 322, c_y0 + 364, "Water-table &amp; ditch<br/>capillary drawdown"),
    ]
    for y0_r, y1_r, html_txt in regimes:
        c_flux_arrows.append(
            f'<path d="M 304,{y0_r:.1f} L 312,{y0_r:.1f} L 312,{y1_r:.1f} L 304,{y1_r:.1f}" '
            f'fill="none" stroke="#D97706" stroke-width="1.3"/>'
        )
        c_labels_html.append(
            f'<div class="regime-lbl" style="left:318px; top:{0.5*(y0_r + y1_r):.1f}px; font-size:10.8px;">{html_txt}</div>'
        )

    # Depth ticks along x=30
    c_ticks_svg = []
    for z_val in range(0, 61, 10):
        yy = c_y0 + (z_val / 10.0) * c_dz
        c_ticks_svg.append(
            f'<line x1="24" y1="{yy:.1f}" x2="30" y2="{yy:.1f}" stroke="#475569" stroke-width="1.0"/>'
            f'<text x="20" y="{yy + 3.8:.1f}" text-anchor="end" font-size="10.8" font-weight="500" fill="#475569">{z_val}</text>'
        )

    # -------------------------------------------------------------------------
    # HTML + CSS MARKUP
    # -------------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Double-Row Raised-Bed &amp; 1D PINN Architecture</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
  @page {{
    size: {PAGE_W}px {PAGE_H}px;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  html, body {{
    width: {PAGE_W}px;
    height: {PAGE_H}px;
    background: #FFFFFF;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }}
  .page-grid {{
    display: grid;
    grid-template-columns: 710px 534px;
    gap: 12px;
    padding: 12px;
    width: {PAGE_W}px;
    height: {PAGE_H}px;
  }}
  .card {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    height: 556px;
  }}
  .panel-hdr {{
    font-size: 14.5px;
    font-weight: 700;
    color: #0F172A;
    height: 20px;
    line-height: 20px;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
    display: flex;
    align-items: baseline;
    gap: 7px;
  }}
  .panel-letter {{
    font-size: 16px;
    font-weight: 800;
    color: #0F172A;
  }}

  /* Vector stage containers */
  .diagram-stage {{
    position: relative;
    flex: none;
  }}
  .stage-a {{
    width: 686px;
    height: 504px;
  }}
  .stage-b {{
    width: 510px;
    height: 462px;
  }}
  .diagram-stage svg {{
    display: block;
  }}

  /* Modern Nature Typography */
  .lbl {{
    position: absolute;
    font-size: 11.8px;
    font-weight: 600;
    line-height: 1.25;
    color: #0F172A;
    pointer-events: none;
    white-space: nowrap;
  }}
  .lbl-muted {{ color: #475569; font-weight: 500; }}
  .lbl-blue {{ color: #1D4ED8; }}
  .lbl-water {{ color: #0284C7; }}
  .lbl-teal {{ color: #0D9488; }}
  .lbl-amber {{ color: #D97706; }}
  .lbl-green {{ color: #2D5A27; }}
  .lbl-center {{ transform: translate(-50%, -50%); text-align: center; }}
  .lbl-right {{ transform: translate(-100%, -50%); text-align: right; }}
  .lbl-left {{ transform: translate(0, -50%); text-align: left; }}

  .pill {{
    position: absolute;
    z-index: 5;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 3px 8px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
    font-size: 11.0px;
    font-weight: 600;
    white-space: nowrap;
  }}

  .math-center {{
    position: absolute;
    transform: translate(-50%, -50%);
    font-size: 11.8px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .math-left {{
    position: absolute;
    transform: translate(0, -50%);
    font-size: 11.8px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .math-right {{
    position: absolute;
    transform: translate(-100%, -50%);
    font-size: 11.8px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .math-tag {{
    position: absolute;
    transform: translate(-50%, -50%);
    font-size: 11.2px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .c-blue {{ color: #1D4ED8; }}
  .c-violet {{ color: #6D28D9; }}
  .c-teal {{ color: #0D9488; }}
  .c-amber {{ color: #D97706; }}
  .c-muted {{ color: #475569; }}
  .subnote {{ font-size: 10.2px; font-weight: 500; color: #475569; }}

  .regime-lbl {{
    position: absolute;
    transform: translate(0, -50%);
    font-size: 10.8px;
    line-height: 1.25;
    font-weight: 600;
    color: #0F172A;
    white-space: nowrap;
  }}

  .legend-bar {{
    margin-top: 6px;
    height: 42px;
    padding: 4px 10px;
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    align-content: center;
    gap: 4px 12px;
    font-size: 10.8px;
    font-weight: 500;
    color: #334155;
  }}
  .leg-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }}
  .leg-swatch {{
    width: 20px;
    height: 10px;
    flex-shrink: 0;
  }}

  .katex {{ font-size: 1.0em !important; }}
</style>
</head>
<body>
<div class="page-grid">

  <!-- PANEL a (Left Card: 710px wide) -->
  <div class="card">
    <div class="panel-hdr"><span class="panel-letter">a</span> Double-row raised-bed &amp; drainage-ditch cross-section</div>
    <div class="diagram-stage stage-a">
      <svg viewBox="0 0 686 504" class="stage-a">
        <defs>
          <linearGradient id="sand-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#FBF9F5"/>
            <stop offset="100%" stop-color="#EFECE4"/>
          </linearGradient>
          <linearGradient id="clay-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#DFD5C8"/>
            <stop offset="100%" stop-color="#CBBBA8"/>
          </linearGradient>
          <linearGradient id="water-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.45"/>
            <stop offset="100%" stop-color="#0284C7" stop-opacity="0.65"/>
          </linearGradient>
          <pattern id="sand-dots" width="14" height="14" patternUnits="userSpaceOnUse">
            <circle cx="3" cy="4" r="0.65" fill="#D6D0C4"/>
            <circle cx="10" cy="11" r="0.6" fill="#D6D0C4"/>
          </pattern>

          <!-- Botanical leaf gradients -->
          <linearGradient id="lf-dark" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#1B4332"/>
            <stop offset="100%" stop-color="#2D6A4F"/>
          </linearGradient>
          <linearGradient id="lf-mid" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#2D6A4F"/>
            <stop offset="100%" stop-color="#40916C"/>
          </linearGradient>
          <linearGradient id="lf-sun" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#40916C"/>
            <stop offset="100%" stop-color="#52B788"/>
          </linearGradient>

          <!-- Reusable 14-leaf rosette cluster -->
          <g id="leaf-cluster-dark">
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-dark)" transform="rotate(0)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-dark)" transform="rotate(28)"/>
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-dark)" transform="rotate(58)"/>
            <path d="M 0,0 Q 4.5,-2.8 9.5,0 Q 4.5,2.8 0,0 Z" fill="url(#lf-dark)" transform="rotate(88)"/>
            <path d="M 0,0 Q 5,-3.1 10.2,0 Q 5,3.1 0,0 Z" fill="url(#lf-dark)" transform="rotate(116)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-dark)" transform="rotate(146)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-dark)" transform="rotate(174)"/>
            <path d="M 0,0 Q 5,-3.2 10.4,0 Q 5,3.2 0,0 Z" fill="url(#lf-dark)" transform="rotate(205)"/>
            <path d="M 0,0 Q 4.6,-2.9 9.4,0 Q 4.6,2.9 0,0 Z" fill="url(#lf-dark)" transform="rotate(235)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-dark)" transform="rotate(266)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-dark)" transform="rotate(298)"/>
            <path d="M 0,0 Q 5,-3.0 10.2,0 Q 5,3.0 0,0 Z" fill="url(#lf-dark)" transform="rotate(330)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#2D6A4F" transform="translate(2,-1) rotate(45)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#2D6A4F" transform="translate(-2,1) rotate(215)"/>
          </g>
          <g id="leaf-cluster-mid">
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-mid)" transform="rotate(0)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-mid)" transform="rotate(28)"/>
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-mid)" transform="rotate(58)"/>
            <path d="M 0,0 Q 4.5,-2.8 9.5,0 Q 4.5,2.8 0,0 Z" fill="url(#lf-mid)" transform="rotate(88)"/>
            <path d="M 0,0 Q 5,-3.1 10.2,0 Q 5,3.1 0,0 Z" fill="url(#lf-mid)" transform="rotate(116)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-mid)" transform="rotate(146)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-mid)" transform="rotate(174)"/>
            <path d="M 0,0 Q 5,-3.2 10.4,0 Q 5,3.2 0,0 Z" fill="url(#lf-mid)" transform="rotate(205)"/>
            <path d="M 0,0 Q 4.6,-2.9 9.4,0 Q 4.6,2.9 0,0 Z" fill="url(#lf-mid)" transform="rotate(235)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-mid)" transform="rotate(266)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-mid)" transform="rotate(298)"/>
            <path d="M 0,0 Q 5,-3.0 10.2,0 Q 5,3.0 0,0 Z" fill="url(#lf-mid)" transform="rotate(330)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#40916C" transform="translate(2,-1) rotate(45)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#40916C" transform="translate(-2,1) rotate(215)"/>
          </g>
          <g id="leaf-cluster-sun">
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-sun)" transform="rotate(0)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-sun)" transform="rotate(28)"/>
            <path d="M 0,0 Q 5,-3.2 10.5,0 Q 5,3.2 0,0 Z" fill="url(#lf-sun)" transform="rotate(58)"/>
            <path d="M 0,0 Q 4.5,-2.8 9.5,0 Q 4.5,2.8 0,0 Z" fill="url(#lf-sun)" transform="rotate(88)"/>
            <path d="M 0,0 Q 5,-3.1 10.2,0 Q 5,3.1 0,0 Z" fill="url(#lf-sun)" transform="rotate(116)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-sun)" transform="rotate(146)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-sun)" transform="rotate(174)"/>
            <path d="M 0,0 Q 5,-3.2 10.4,0 Q 5,3.2 0,0 Z" fill="url(#lf-sun)" transform="rotate(205)"/>
            <path d="M 0,0 Q 4.6,-2.9 9.4,0 Q 4.6,2.9 0,0 Z" fill="url(#lf-sun)" transform="rotate(235)"/>
            <path d="M 0,0 Q 5,-3.0 10.0,0 Q 5,3.0 0,0 Z" fill="url(#lf-sun)" transform="rotate(266)"/>
            <path d="M 0,0 Q 4.8,-3.1 9.8,0 Q 4.8,3.1 0,0 Z" fill="url(#lf-sun)" transform="rotate(298)"/>
            <path d="M 0,0 Q 5,-3.0 10.2,0 Q 5,3.0 0,0 Z" fill="url(#lf-sun)" transform="rotate(330)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#52B788" transform="translate(2,-1) rotate(45)"/>
            <path d="M 0,0 Q 4,-2.4 7.5,0 Q 4,2.4 0,0 Z" fill="#52B788" transform="translate(-2,1) rotate(215)"/>
          </g>

          <!-- SVG Arrow Markers -->
          <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#1D4ED8"/>
          </marker>
          <marker id="arr-water" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#0284C7"/>
          </marker>
          <marker id="arr-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#0D9488"/>
          </marker>
          <marker id="arr-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#D97706"/>
          </marker>
          <marker id="arr-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#475569"/>
          </marker>

          <!-- ClipPath to guarantee NO soil fill/lines ever appear in open ditch air or water -->
          <clipPath id="ground-clip">
            <path d="{topsoil_fill_d}"/>
          </clipPath>
        </defs>

        <!-- SUBSURFACE HYDROGEOLOGY (Strictly clipped to ground contour) -->
        <g clip-path="url(#ground-clip)">
          <!-- Sandy Topsoil fill & subtle stipple -->
          <rect x="0" y="0" width="686" height="504" fill="url(#sand-grad)"/>
          <rect x="0" y="0" width="686" height="504" fill="url(#sand-dots)"/>

          <!-- Argillic Clay Horizon Subsoil fill (z >= 30 cm, y >= 320) -->
          <path d="{clay_fill_d}" fill="url(#clay-grad)"/>
          <path d="{clay_line_d}" fill="none" stroke="#64748B" stroke-width="1.3" stroke-dasharray="6 3.5"/>

          <!-- Local Water Table / Capillary Fringe band & line -->
          <path d="{wt_band_d}" fill="#38BDF8" opacity="0.16"/>
          <path d="{wt_line_d}" fill="none" stroke="#0284C7" stroke-width="1.5" stroke-dasharray="5 3"/>

          <!-- Clean ditch water bodies (confined strictly within the depression below y_water=348) -->
          <path d="{left_water_poly_d}" fill="url(#water-grad)"/>
          <path d="{right_water_poly_d}" fill="url(#water-grad)"/>
        </g>

        <!-- Clean Ditch Water Surface Lines & Level Triangles -->
        <line x1="25" y1="{y_water:.0f}" x2="99" y2="{y_water:.0f}" stroke="#0284C7" stroke-width="1.3"/>
        <polygon points="58,{y_water - 5:.0f} 66,{y_water - 5:.0f} 62,{y_water:.0f}" fill="#0284C7"/>

        <line x1="587" y1="{y_water:.0f}" x2="661" y2="{y_water:.0f}" stroke="#0284C7" stroke-width="1.3"/>
        <polygon points="620,{y_water - 5:.0f} 628,{y_water - 5:.0f} 624,{y_water:.0f}" fill="#0284C7"/>

        <!-- Ground Surface Outline & Smooth Rounded Turf Ribbons -->
        <path d="{bed_surface_d}" fill="none" stroke="#475569" stroke-width="1.4"/>
        <path d="{grass_left_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
        <path d="{grass_left_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>
        <path d="{grass_mid_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
        <path d="{grass_mid_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>
        <path d="{grass_right_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
        <path d="{grass_right_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>

        <!-- Fibrous Roots (0-30 cm) & Botanical Citrus Trees (Symmetric, scale=1.22) -->
        {roots1_svg}
        {roots2_svg}
        {tree1_svg}
        {tree2_svg}

        <!-- Weather Station Mast (x=343 in center alley) -->
        <line x1="343" y1="{y_surf:.0f}" x2="343" y2="122" stroke="#334155" stroke-width="2.0"/>
        <line x1="329" y1="129" x2="356" y2="129" stroke="#334155" stroke-width="1.4"/>
        <circle cx="329" cy="127" r="2.2" fill="#0F172A"/>
        <polygon points="338,139 348,139 350,131 336,131" fill="#475569" stroke="#0F172A" stroke-width="0.6"/>
        <rect x="339" y="143" width="8" height="10" fill="#F8FAFC" stroke="#475569" stroke-width="0.7"/>
        <rect x="334" y="166" width="18" height="12" rx="1" fill="#1E3A8A" stroke="#0F172A" stroke-width="0.7"/>

        <!-- Micro-sprinkler (x=414 under left skirt of Row 2 tree) -->
        <line x1="394" y1="{y_surf:.0f}" x2="434" y2="{y_surf:.0f}" stroke="#0F172A" stroke-width="1.8"/>
        <line x1="414" y1="{y_surf:.0f}" x2="414" y2="238" stroke="#0F172A" stroke-width="1.5"/>
        <rect x="411.5" y="234" width="5" height="4" fill="#0284C7"/>
        <polygon points="414,236 394,{y_surf:.0f} 434,{y_surf:.0f}" fill="#38BDF8" opacity="0.22"/>
        <line x1="414" y1="236" x2="400" y2="{y_surf - 4:.0f}" stroke="#0284C7" stroke-width="1.1" marker-end="url(#arr-water)"/>
        <line x1="414" y1="236" x2="428" y2="{y_surf - 4:.0f}" stroke="#0284C7" stroke-width="1.1" marker-end="url(#arr-water)"/>

        <!-- Solar Telemetry Mast (x=552 near right shoulder) & Cable to Sentek Probe -->
        <line x1="552" y1="272" x2="552" y2="205" stroke="#334155" stroke-width="1.8"/>
        <rect x="545" y="207" width="14" height="12" rx="1" fill="#F8FAFC" stroke="#334155" stroke-width="0.8"/>
        <polygon points="542,206 560,197 564,202 546,211" fill="#1E3A8A" stroke="#0F172A" stroke-width="0.7"/>
        <path d="M 494,256 Q 520,246 545,214" fill="none" stroke="#334155" stroke-width="1.1"/>

        <!-- Dashed Highlight Box around Sentek Probe (x=490, 20 cm from Row 2 trunk) -->
        <rect x="478" y="252" width="24" height="140" rx="4"
              fill="rgba(2,132,199,0.06)" stroke="#0284C7" stroke-width="1.2" stroke-dasharray="4 3"/>
        <!-- Sentek Drill & Drop Probe (x=490, length 120 px = 60 cm, sensor rings at 270, 290, 310, 330, 350, 370) -->
        <rect x="485" y="{y_surf:.0f}" width="10" height="{y_probe_bot - y_surf:.0f}" fill="#F8FAFC" stroke="#0F172A" stroke-width="1.1"/>
        <polygon points="485,{y_probe_bot:.0f} 495,{y_probe_bot:.0f} 490,{y_probe_bot + 8:.0f}" fill="#94A3B8" stroke="#0F172A" stroke-width="1.0"/>
        <rect x="483" y="{y_surf - 5:.0f}" width="14" height="5" rx="1.5" fill="#0F172A"/>
        {"".join(f'<rect x="483.5" y="{yc - 2.8:.1f}" width="13" height="5.6" rx="1" fill="#DC2626" stroke="#7F1D1D" stroke-width="0.6"/>' for yc in sensor_zs_a)}

        <!-- Hydrological Flow Arrows in Panel a -->
        <!-- Rainfall r (x=130, 142, 154) -->
        <line x1="130" y1="55" x2="126" y2="100" stroke="#1D4ED8" stroke-width="1.2" stroke-dasharray="4 3"/>
        <line x1="142" y1="55" x2="138" y2="100" stroke="#1D4ED8" stroke-width="1.2" stroke-dasharray="4 3"/>
        <line x1="154" y1="55" x2="150" y2="105" stroke="#1D4ED8" stroke-width="1.8" marker-end="url(#arr-blue)"/>

        <!-- ET0 above right shoulder & ditch evaporation -->
        <line x1="545" y1="105" x2="545" y2="55" stroke="#D97706" stroke-width="1.9" marker-end="url(#arr-amber)"/>
        <line x1="625" y1="344" x2="625" y2="296" stroke="#D97706" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#arr-amber)"/>

        <!-- Lateral ditch outflows fl_1..fl_5 -->
        {"".join(lat_arrows_a)}

        <!-- Top Bed crown dimension line (spanning x=162..524) -->
        <line x1="162" y1="16" x2="162" y2="254" stroke="#CBD5E1" stroke-width="0.8" stroke-dasharray="3 3"/>
        <line x1="524" y1="16" x2="524" y2="254" stroke="#CBD5E1" stroke-width="0.8" stroke-dasharray="3 3"/>
        <line x1="166" y1="16" x2="228" y2="16" stroke="#475569" stroke-width="1.0" marker-start="url(#arr-slate)"/>
        <line x1="458" y1="16" x2="520" y2="16" stroke="#475569" stroke-width="1.0" marker-end="url(#arr-slate)"/>

        <!-- 20 cm trunk-to-probe dimension (x=454 to x=490) -->
        <line x1="456" y1="246" x2="488" y2="246" stroke="#0F172A" stroke-width="1.0" marker-start="url(#arr-slate)" marker-end="url(#arr-slate)"/>

        <!-- Clean Thin Leader Lines with Circular Anchor Dots -->
        <!-- Row 1 tree leader -->
        <line x1="232" y1="118" x2="232" y2="138" stroke="#475569" stroke-width="0.9"/>
        <circle cx="232" cy="138" r="2.2" fill="#0F172A"/>

        <!-- Row 2 tree leader -->
        <line x1="454" y1="118" x2="454" y2="138" stroke="#475569" stroke-width="0.9"/>
        <circle cx="454" cy="138" r="2.2" fill="#0F172A"/>

        <!-- Weather station leader -->
        <line x1="343" y1="104" x2="343" y2="122" stroke="#475569" stroke-width="0.9"/>
        <circle cx="343" cy="122" r="2.2" fill="#0F172A"/>

        <!-- Micro-sprinkler leader (short diagonal to nozzle at 414, 236) -->
        <line x1="388" y1="218" x2="414" y2="236" stroke="#0284C7" stroke-width="1.2"/>
        <circle cx="414" cy="236" r="2.2" fill="#0284C7"/>

        <!-- Solar telemetry leader -->
        <line x1="582" y1="182" x2="558" y2="205" stroke="#475569" stroke-width="0.9"/>
        <circle cx="558" cy="205" r="2.2" fill="#0F172A"/>

        <!-- Lateral drainage leader from open white space -->
        <line x1="82" y1="234" x2="108" y2="290" stroke="#0D9488" stroke-width="0.9"/>
        <circle cx="108" cy="290" r="2.2" fill="#0D9488"/>

        <!-- Argillic clay horizon leader -->
        <line x1="320" y1="330" x2="320" y2="320" stroke="#475569" stroke-width="0.9"/>
        <circle cx="320" cy="320" r="2.2" fill="#475569"/>

        <!-- Water table leader -->
        <line x1="320" y1="374" x2="320" y2="354" stroke="#475569" stroke-width="0.9"/>
        <circle cx="320" cy="354" r="2.2" fill="#0284C7"/>

        <!-- Flanking ditch leader -->
        <line x1="62" y1="304" x2="62" y2="348" stroke="#475569" stroke-width="0.9"/>
        <circle cx="62" cy="348" r="2.2" fill="#0284C7"/>
      </svg>

      <!-- Panel a Typography & Labels -->
      <div class="lbl lbl-center" style="left:343px; top:16px; background:#FFFFFF; padding:0 6px;">
        Double-row raised bed (~5–6 m wide)
      </div>

      <div class="lbl lbl-blue lbl-center" style="left:142px; top:42px;">Rainfall (<span class="k" data-tex="r"></span>)</div>
      <div class="lbl lbl-amber lbl-center" style="left:545px; top:42px;">Evapotranspiration (<span class="k" data-tex="ET_0"></span>)</div>

      <div class="lbl lbl-green lbl-center" style="left:232px; top:108px;">Grapefruit tree (Row 1)</div>
      <div class="lbl lbl-green lbl-center" style="left:454px; top:108px;">Monitored grapefruit tree (Row 2)</div>

      <div class="lbl lbl-center" style="left:343px; top:96px;">Weather station</div>

      <div class="pill lbl-water lbl-center" style="left:365px; top:210px; padding:3px 7px; border-color:#0284C7; z-index:6; background:#FFFFFF;">
        Micro-sprinkler<br/>(<span class="k" data-tex="q_{{\\mathrm{{irr}}}}"></span>)
      </div>
      <div class="lbl lbl-center" style="left:472px; top:238px; font-size:10.5px; font-weight:700;">20 cm</div>

      <!-- Solar telemetry -->
      <div class="lbl lbl-center" style="left:615px; top:182px;">Solar telemetry</div>

      <!-- Sentek probe badges placed directly below the probe tip -->
      <div class="pill lbl-water lbl-center" style="left:490px; top:404px; font-size:10.8px; font-weight:700; border-color:#0284C7; color:#0284C7;">
        Sentek probe (5–55 cm)
      </div>
      <div class="pill lbl-water lbl-center" style="left:490px; top:432px; font-size:10.5px; font-weight:700; border-color:#0284C7; color:#0284C7;">
        Expanded in Panel <strong>b</strong> &rarr;
      </div>

      <!-- Lateral drainage label in open air above left slope -->
      <div class="lbl lbl-teal lbl-center" style="left:82px; top:218px;">Lateral drainage<br/><span class="k" data-tex="fl_1 \\dots fl_5"></span></div>

      <!-- Ditch labels -->
      <div class="lbl lbl-water lbl-center" style="left:62px; top:288px; background:#FFFFFF; padding:2px 6px; border-radius:4px; box-shadow:0 1px 2px rgba(15,23,42,0.08);">
        Flanking<br/>drainage ditch
      </div>
      <div class="lbl lbl-amber lbl-center" style="left:625px; top:280px;">Ditch evaporation</div>

      <!-- Soil horizon labels (Sandy topsoil in root-free center zone x=275..410) -->
      <div class="lbl lbl-center" style="left:343px; top:290px; background: rgba(251,249,245,0.88); padding: 1px 6px; border-radius: 3px;">
        Sandy topsoil (Alfisol, 0–30 cm)
      </div>
      <div class="lbl lbl-center" style="left:320px; top:338px;">
        Argillic clay horizon (<span class="k" data-tex="z\\approx 30\\,\\text{{cm}}"></span>; low permeability)
      </div>
      <div class="lbl lbl-water lbl-center" style="left:320px; top:382px;">
        Local water table / capillary fringe
      </div>
    </div>
  </div>

  <!-- PANEL b (Right Card: 534px wide) -->
  <div class="card">
    <div class="panel-hdr"><span class="panel-letter">b</span> 1D finite-volume PINN model (Eq. 8)</div>
    <div class="diagram-stage stage-b">
      <svg viewBox="0 0 510 462" class="stage-b">
        <defs>
          <linearGradient id="c-sand-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#FBF9F5"/>
            <stop offset="100%" stop-color="#EFECE4"/>
          </linearGradient>
          <linearGradient id="c-clay-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#DFD5C8"/>
            <stop offset="100%" stop-color="#D1C2B0"/>
          </linearGradient>
          <marker id="arr-violet-dn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0,1.5 L 9,5 L 0,8.5 Z" fill="#6D28D9"/>
          </marker>
          <marker id="arr-violet-up" viewBox="0 0 10 10" refX="2" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 10,1.5 L 1,5 L 10,8.5 Z" fill="#6D28D9"/>
          </marker>
        </defs>

        <!-- Depth Axis z (cm) -->
        <line x1="30" y1="38" x2="30" y2="430" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
        {"".join(c_ticks_svg)}

        <!-- 6 Finite-Volume Compartments (c_x0=104, c_x1=222) -->
        {"".join(c_compartments)}
        <rect x="{c_x0}" y="{c_y0}" width="{c_x1 - c_x0}" height="{6 * c_dz}" fill="none" stroke="#0F172A" stroke-width="1.3"/>

        <!-- Argillic horizon boundary line at z = 30 cm -->
        <line x1="94" y1="{c_y0 + 3 * c_dz:.1f}" x2="234" y2="{c_y0 + 3 * c_dz:.1f}"
              stroke="#475569" stroke-width="1.8" stroke-dasharray="5 3"/>

        <!-- Surface Inflows at z = 0 cm (Layer 1) -->
        <line x1="132" y1="20" x2="132" y2="{c_y0 - 2:.1f}" stroke="#0284C7" stroke-width="1.9" marker-end="url(#arr-water)"/>
        <line x1="194" y1="20" x2="194" y2="{c_y0 - 2:.1f}" stroke="#1D4ED8" stroke-width="1.9" marker-end="url(#arr-blue)"/>

        <!-- Dotted pointer from Direct rainfall header to fr_2 -->
        <path d="M 52,94 Q 54,116 64,133" fill="none" stroke="#1D4ED8" stroke-width="1.0" stroke-dasharray="2.5 2.5" marker-end="url(#arr-blue)"/>

        <!-- Internal, lateral, deep percolation, and ET0 flux arrows & brackets -->
        {"".join(c_flux_arrows)}
      </svg>

      <!-- Panel b Labels -->
      <div class="lbl lbl-center" style="left:28px; top:24px; font-size:11.2px;">
        <span class="k" data-tex="z\\;(\\text{{cm}})"></span>
      </div>

      <!-- Direct rainfall header moved out of top row to left side above fr_2 -->
      <div class="lbl lbl-blue lbl-center" style="left:52px; top:76px; font-size:10.5px; line-height:1.2;">
        Direct<br/>rainfall<br/><span class="k" data-tex="fr_i(r)"></span>
      </div>

      <div class="lbl lbl-water lbl-center" style="left:132px; top:12px;">
        <span class="k" data-tex="fq_{{\\mathrm{{irr}}}}(q_{{\\mathrm{{irr}}}})"></span>
      </div>

      <div class="lbl lbl-blue lbl-center" style="left:194px; top:12px;">
        <span class="k" data-tex="fr_1(r)"></span>
      </div>

      <!-- Evapotranspirative sink header starting at x=248px -->
      <div class="lbl lbl-amber lbl-left" style="left:248px; top:16px; font-size:10.6px; line-height:1.2;">
        Evapotranspirative sink<br/><span class="k" data-tex="S_i(m_i, ET_0)\\;[pc_i]"></span>
      </div>

      <!-- Argillic horizon label with white background -->
      <div class="lbl lbl-right" style="left:98px; top:232px; font-size:9.8px; background:#FFFFFF; padding:0 2px;">
        Argillic horizon<br/><span class="lbl-muted">(30 cm)</span>
      </div>

      {"".join(c_labels_html)}
    </div>

    <!-- Horizontal Legend Bar -->
    <div class="legend-bar">
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="1" y1="5" x2="18" y2="5" stroke="#0284C7" stroke-width="1.8"/><polygon points="15,2 21,5 15,8" fill="#0284C7"/></svg>
        <span><span class="k" data-tex="fq_{{\\mathrm{{irr}}}}"></span> Irrigation</span>
      </div>
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="1" y1="5" x2="18" y2="5" stroke="#1D4ED8" stroke-width="1.8"/><polygon points="15,2 21,5 15,8" fill="#1D4ED8"/></svg>
        <span><span class="k" data-tex="f_i \\ge 0"></span> Gravity drainage</span>
      </div>
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="1" y1="5" x2="18" y2="5" stroke="#0D9488" stroke-width="1.8"/><polygon points="15,2 21,5 15,8" fill="#0D9488"/></svg>
        <span><span class="k" data-tex="fl_i \\ge 0"></span> Ditch outflow</span>
      </div>
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="1" y1="5" x2="18" y2="5" stroke="#1D4ED8" stroke-width="1.6" stroke-dasharray="3 2"/><polygon points="15,2 21,5 15,8" fill="#1D4ED8"/></svg>
        <span><span class="k" data-tex="fr_i(r)"></span> Direct rainfall</span>
      </div>
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="4" y1="5" x2="18" y2="5" stroke="#6D28D9" stroke-width="1.6" stroke-dasharray="2.5 1.5"/><polygon points="6,2 0,5 6,8" fill="#6D28D9"/><polygon points="16,2 22,5 16,8" fill="#6D28D9"/></svg>
        <span><span class="k" data-tex="f_{{i,i+1}}\\lessgtr 0"></span> Matric exchange</span>
      </div>
      <div class="leg-item">
        <svg class="leg-swatch" viewBox="0 0 22 10"><line x1="1" y1="5" x2="18" y2="5" stroke="#D97706" stroke-width="1.8"/><polygon points="15,2 21,5 15,8" fill="#D97706"/></svg>
        <span><span class="k" data-tex="S_i \\equiv pc_i"></span> ET sink</span>
      </div>
    </div>
  </div>

</div>

<script>
  document.querySelectorAll('.k').forEach(el => {{
    katex.render(el.getAttribute('data-tex'), el, {{
      throwOnError: false,
      displayMode: false
    }});
  }});
</script>
</body>
</html>
"""
    return html


def main():
    html_path = os.path.join(OUT_DIR, "bed_ditch_schematic.html")
    pdf_path = os.path.join(OUT_DIR, "bed_ditch_schematic_improved.pdf")
    png_path = os.path.join(OUT_DIR, "bed_ditch_schematic_improved.png")

    html_content = build_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Wrote {html_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        context = browser.new_context(
            viewport={"width": PAGE_W, "height": PAGE_H},
            device_scale_factor=DPI_SCALE,
        )
        page = context.new_page()
        page.goto(f"file://{html_path}", wait_until="networkidle")
        page.evaluate("() => document.fonts.ready")
        page.wait_for_timeout(400)

        # Render 300 DPI PNG
        page.screenshot(path=png_path, full_page=False)
        # Render crisp vector PDF
        page.pdf(
            path=pdf_path,
            width=f"{PAGE_W}px",
            height=f"{PAGE_H}px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
        )
        browser.close()

    # Synchronize copies
    pdf2_path = os.path.join(OUT_DIR, "bed_ditch_schematic_improved_2.pdf")
    shutil.copyfile(pdf_path, pdf2_path)

    improved_paper_pdf = os.path.join(
        os.path.dirname(OUT_DIR), "Improved_Paper", "bed_ditch_schematic.pdf"
    )
    if os.path.exists(os.path.dirname(improved_paper_pdf)):
        shutil.copyfile(pdf_path, improved_paper_pdf)
        print(f"Synced to {improved_paper_pdf}")

    print(f"Saved:\n  {pdf_path}\n  {pdf2_path}\n  {png_path}")


if __name__ == "__main__":
    main()
