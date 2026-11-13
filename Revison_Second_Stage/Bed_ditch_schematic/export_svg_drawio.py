#!/usr/bin/env python3
"""
export_svg_drawio.py
====================
Generates:
  1. bed_ditch_schematic.svg       (100% self-contained pure SVG, zero HTML divs)
  2. bed_ditch_schematic.drawio    (native Draw.io / diagrams.net XML with 100% discrete, selectable, editable elements)
  3. bed_ditch_schematic.drawio.svg (Draw.io SVG hybrid containing editable XML model)
  4. bed_ditch_schematic_vector.pdf (publication-ready vector PDF via Inkscape)
"""

import os
import re
import html
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import botanical tree and root generators
from generate_bed_ditch_schematic import build_botanical_tree_svg, build_fibrous_roots_svg

def create_pure_svg():
    PAGE_W = 1280
    PAGE_H = 580

    y_surf = 260.0
    y_clay = 320.0
    y_water = 348.0
    y_ditch = 364.0
    y_probe_bot = 380.0

    tree1_svg = build_botanical_tree_svg(232.0, y_surf, seed=42, scale=1.22, rx_left_scale=1.0, rx_right_scale=1.0)
    roots1_svg = build_fibrous_roots_svg(232.0, y_surf, y_clay, x_min=190.0, x_max=274.0, seed=51)

    tree2_svg = build_botanical_tree_svg(454.0, y_surf, seed=77, scale=1.22, rx_left_scale=1.0, rx_right_scale=1.0)
    roots2_svg = build_fibrous_roots_svg(454.0, y_surf, y_clay, x_min=412.0, x_max=484.0, seed=88)

    bed_surface_d = (
        f"M 0,312 C 14,312 26,{y_ditch:.0f} 44,{y_ditch:.0f} L 82,{y_ditch:.0f} "
        f"C 112,{y_ditch:.0f} 132,{y_surf:.0f} 162,{y_surf:.0f} "
        f"L 524,{y_surf:.0f} "
        f"C 554,{y_surf:.0f} 574,{y_ditch:.0f} 604,{y_ditch:.0f} L 642,{y_ditch:.0f} "
        f"C 660,{y_ditch:.0f} 672,312 686,312"
    )
    topsoil_fill_d = bed_surface_d + " L 686,504 L 0,504 Z"

    left_water_poly_d = (
        f"M 25,{y_water:.0f} L 99,{y_water:.0f} "
        f"C 93,354 87,{y_ditch:.0f} 82,{y_ditch:.0f} L 44,{y_ditch:.0f} "
        f"C 39,{y_ditch:.0f} 32,354 25,{y_water:.0f} Z"
    )
    right_water_poly_d = (
        f"M 587,{y_water:.0f} L 661,{y_water:.0f} "
        f"C 654,354 647,{y_ditch:.0f} 642,{y_ditch:.0f} L 604,{y_ditch:.0f} "
        f"C 599,{y_ditch:.0f} 593,354 587,{y_water:.0f} Z"
    )

    clay_xs = np.linspace(0, 686, 120)
    clay_ys = y_clay + 2.0 * np.sin(0.024 * clay_xs + 0.4) + 1.0 * np.cos(0.05 * clay_xs)
    clay_line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(clay_xs, clay_ys))
    clay_fill_d = f"M 0,504 L " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(clay_xs, clay_ys)) + " L 686,504 Z"

    wt_xs = np.linspace(46, 640, 90)
    wt_ys = 360.0 - 8.0 * np.exp(-((wt_xs - 343.0) / 150.0) ** 2)
    wt_line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(wt_xs, wt_ys))
    wt_band_d = (
        "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(wt_xs, wt_ys)) +
        " L " + " L ".join(f"{x:.1f},{y + 8.0:.1f}" for x, y in zip(wt_xs[::-1], wt_ys[::-1])) + " Z"
    )

    grass_left_d = f"M 95,356 C 112,{y_ditch:.0f} 132,{y_surf:.0f} 174,{y_surf:.0f}"
    grass_mid_d = f"M 296,{y_surf:.0f} L 390,{y_surf:.0f}"
    grass_right_d = f"M 512,{y_surf:.0f} C 554,{y_surf:.0f} 574,{y_ditch:.0f} 591,356"

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

    # Panel b geometry
    c_y0 = 46.0
    c_dz = 62.0
    c_x0, c_x1 = 104.0, 222.0
    c_xc = 163.0

    c_compartments = []
    c_labels_svg = []
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
                f'fill="#38BDF8" fill-opacity="0.14"/>'
            )
        c_compartments.append(
            f'<rect x="{c_x0 + 3}" y="{ym - 6:.1f}" width="7" height="12" rx="1.5" '
            f'fill="#DC2626" stroke="#7F1D1D" stroke-width="0.6"/>'
        )
        c_labels_svg.append(
            f'<text x="{c_xc:.1f}" y="{ym + 4.2:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.8" '
            f'font-weight="600" fill="#0F172A" text-anchor="middle">z<tspan font-size="9" dy="3">{idx}</tspan>'
            f'<tspan font-size="11.8" dy="-3"> = {z_mid} cm (m</tspan><tspan font-size="9" dy="3">{idx}</tspan>'
            f'<tspan font-size="11.8" dy="-3">)</tspan></text>'
        )

    c_flux_arrows = []
    for i in range(1, 6):
        yi = c_y0 + i * c_dz
        c_flux_arrows.append(
            f'<line x1="132" y1="{yi - 8:.1f}" x2="132" y2="{yi + 8:.1f}" '
            f'stroke="#1D4ED8" stroke-width="1.8" marker-end="url(#arr-blue)"/>'
        )
        c_labels_svg.append(
            f'<text x="142" y="{yi - 2:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.2" '
            f'font-weight="600" fill="#1D4ED8">f<tspan font-size="8.5" dy="2.5">{i}</tspan></text>'
        )
        c_flux_arrows.append(
            f'<line x1="194" y1="{yi - 8:.1f}" x2="194" y2="{yi + 8:.1f}" '
            f'stroke="#6D28D9" stroke-width="1.6" stroke-dasharray="3 2" '
            f'marker-start="url(#arr-violet-up)" marker-end="url(#arr-violet-dn)"/>'
        )
        c_labels_svg.append(
            f'<text x="180" y="{yi - 2:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.2" '
            f'font-weight="600" fill="#6D28D9" text-anchor="end">f<tspan font-size="8.5" dy="2.5">{i},{i+1}</tspan></text>'
        )

    y_bot = c_y0 + 6 * c_dz
    c_flux_arrows.append(
        f'<line x1="144" y1="{y_bot:.1f}" x2="144" y2="{y_bot + 28:.1f}" '
        f'stroke="#1D4ED8" stroke-width="2.0" marker-end="url(#arr-blue)"/>'
    )
    c_labels_svg.append(
        f'<text x="154" y="{y_bot + 18:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.8" '
        f'font-weight="600" fill="#1D4ED8">f<tspan font-size="9" dy="2.5">6</tspan>'
        f'<tspan font-size="11.8" dy="-2.5">(m</tspan><tspan font-size="9" dy="2.5">6</tspan>'
        f'<tspan font-size="11.8" dy="-2.5">)</tspan> <tspan font-size="10.2" font-weight="500" fill="#475569">(deep percolation)</tspan></text>'
    )

    for i in range(2, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        if i == 3:
            y_start = ym - 12.0
            y_end = ym - 4.0
            y_lbl = ym - 10.0
        elif i == 4:
            y_start = ym
            y_end = ym + 8.0
            y_lbl = ym + 4.0
        else:
            y_start = ym - 6.0
            y_end = ym + 2.0
            y_lbl = ym - 4.0

        c_flux_arrows.append(
            f'<line x1="68" y1="{y_start:.1f}" x2="102" y2="{y_end:.1f}" '
            f'stroke="#1D4ED8" stroke-width="1.5" marker-end="url(#arr-blue)"/>'
        )
        c_labels_svg.append(
            f'<text x="64" y="{y_lbl:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.8" '
            f'font-weight="600" fill="#1D4ED8" text-anchor="end">fr<tspan font-size="9" dy="2.5">{i}</tspan></text>'
        )

    for i in range(1, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        y_fl = ym - 12.0
        y_et = ym + 12.0
        if i <= 5:
            c_flux_arrows.append(
                f'<line x1="{c_x1}" y1="{y_fl:.1f}" x2="254" y2="{y_fl:.1f}" '
                f'stroke="#0D9488" stroke-width="1.6" marker-end="url(#arr-teal)"/>'
            )
            c_labels_svg.append(
                f'<text x="259" y="{y_fl + 4:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.8" '
                f'font-weight="600" fill="#0D9488">fl<tspan font-size="9" dy="2.5">{i}</tspan></text>'
            )
        else:
            c_labels_svg.append(
                f'<text x="232" y="{y_fl + 3:.1f}" font-family="Inter, Arial, sans-serif" font-size="10.6" '
                f'font-weight="500" fill="#475569">fl<tspan font-size="8.5" dy="2">6</tspan><tspan font-size="10.6" dy="-2"> = 0</tspan></text>'
            )

        c_flux_arrows.append(
            f'<line x1="{c_x1}" y1="{y_et:.1f}" x2="298" y2="{y_et:.1f}" '
            f'stroke="#D97706" stroke-width="1.7" marker-end="url(#arr-amber)"/>'
        )
        c_labels_svg.append(
            f'<text x="232" y="{y_et + 4:.1f}" font-family="Inter, Arial, sans-serif" font-size="11.2" '
            f'font-weight="600" fill="#D97706">S<tspan font-size="8.5" dy="2">{i}</tspan></text>'
        )

    regimes = [
        (c_y0 + 10,  c_y0 + 52,  ["Surface soil", "evaporation"]),
        (c_y0 + 72,  c_y0 + 176, ["Shallow root-zone uptake", "(peaks at 25 cm)"]),
        (c_y0 + 196, c_y0 + 300, ["Restricted clay-layer", "depletion"]),
        (c_y0 + 322, c_y0 + 364, ["Water-table &amp; ditch", "capillary drawdown"]),
    ]
    for y0_r, y1_r, lines in regimes:
        c_flux_arrows.append(
            f'<path d="M 304,{y0_r:.1f} L 312,{y0_r:.1f} L 312,{y1_r:.1f} L 304,{y1_r:.1f}" '
            f'fill="none" stroke="#D97706" stroke-width="1.3"/>'
        )
        yc = 0.5 * (y0_r + y1_r)
        if len(lines) == 1:
            c_labels_svg.append(
                f'<text x="318" y="{yc + 4:.1f}" font-family="Inter, Arial, sans-serif" font-size="10.8" '
                f'font-weight="600" fill="#0F172A">{lines[0]}</text>'
            )
        else:
            c_labels_svg.append(
                f'<text x="318" y="{yc - 2:.1f}" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="600" fill="#0F172A">'
                f'{lines[0]}<tspan x="318" dy="13" font-size="{"10.0" if "(" in lines[1] else "10.8"}" font-weight="{"500" if "(" in lines[1] else "600"}" fill="{"#475569" if "(" in lines[1] else "#0F172A"}">{lines[1]}</tspan></text>'
            )

    c_ticks_svg = []
    for z_val in range(0, 61, 10):
        yy = c_y0 + (z_val / 10.0) * c_dz
        c_ticks_svg.append(
            f'<line x1="24" y1="{yy:.1f}" x2="30" y2="{yy:.1f}" stroke="#475569" stroke-width="1.0"/>'
            f'<text x="20" y="{yy + 3.8:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#475569">{z_val}</text>'
        )

    # Build the full pure SVG
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 {PAGE_W} {PAGE_H}" width="{PAGE_W}" height="{PAGE_H}">
  <defs>
    <style>
      text {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    </style>
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

    <clipPath id="ground-clip">
      <path d="{topsoil_fill_d}"/>
    </clipPath>
  </defs>

  <!-- Background Canvas -->
  <rect x="0" y="0" width="{PAGE_W}" height="{PAGE_H}" fill="#FFFFFF"/>

  <!-- ========================================================================= -->
  <!-- CARD a (Panel a: Left, 710 x 556 px) -->
  <!-- ========================================================================= -->
  <rect x="12" y="12" width="710" height="556" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
  <text x="24" y="33" font-family="Inter, Arial, sans-serif" font-size="14.5" font-weight="700" fill="#0F172A">
    <tspan font-size="16" font-weight="800">a</tspan>  Double-row raised-bed &amp; drainage-ditch cross-section
  </text>

  <g transform="translate(24, 46)">
    <!-- Subsurface Soil -->
    <g clip-path="url(#ground-clip)">
      <rect x="0" y="0" width="686" height="504" fill="url(#sand-grad)"/>
      <rect x="0" y="0" width="686" height="504" fill="url(#sand-dots)"/>
      <path d="{clay_fill_d}" fill="url(#clay-grad)"/>
      <path d="{clay_line_d}" fill="none" stroke="#64748B" stroke-width="1.3" stroke-dasharray="6 3.5"/>
      <path d="{wt_band_d}" fill="#38BDF8" opacity="0.16"/>
      <path d="{wt_line_d}" fill="none" stroke="#0284C7" stroke-width="1.5" stroke-dasharray="5 3"/>
      <path d="{left_water_poly_d}" fill="url(#water-grad)"/>
      <path d="{right_water_poly_d}" fill="url(#water-grad)"/>
    </g>

    <!-- Water Surface Lines -->
    <line x1="25" y1="{y_water:.0f}" x2="99" y2="{y_water:.0f}" stroke="#0284C7" stroke-width="1.3"/>
    <polygon points="58,{y_water - 5:.0f} 66,{y_water - 5:.0f} 62,{y_water:.0f}" fill="#0284C7"/>
    <line x1="587" y1="{y_water:.0f}" x2="661" y2="{y_water:.0f}" stroke="#0284C7" stroke-width="1.3"/>
    <polygon points="620,{y_water - 5:.0f} 628,{y_water - 5:.0f} 624,{y_water:.0f}" fill="#0284C7"/>

    <!-- Ground Surface Outline & Turf Ribbons -->
    <path d="{bed_surface_d}" fill="none" stroke="#475569" stroke-width="1.4"/>
    <path d="{grass_left_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
    <path d="{grass_left_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>
    <path d="{grass_mid_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
    <path d="{grass_mid_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>
    <path d="{grass_right_d}" fill="none" stroke="#4D7C4D" stroke-width="5.0" stroke-linecap="round"/>
    <path d="{grass_right_d}" fill="none" stroke="#659B5E" stroke-width="1.8" stroke-linecap="round" transform="translate(0,-1.5)"/>

    <!-- Trees & Roots -->
    {roots1_svg}
    {roots2_svg}
    {tree1_svg}
    {tree2_svg}

    <!-- Weather Station Mast -->
    <line x1="343" y1="{y_surf:.0f}" x2="343" y2="122" stroke="#334155" stroke-width="2.0"/>
    <line x1="329" y1="129" x2="356" y2="129" stroke="#334155" stroke-width="1.4"/>
    <circle cx="329" cy="127" r="2.2" fill="#0F172A"/>
    <polygon points="338,139 348,139 350,131 336,131" fill="#475569" stroke="#0F172A" stroke-width="0.6"/>
    <rect x="339" y="143" width="8" height="10" fill="#F8FAFC" stroke="#475569" stroke-width="0.7"/>
    <rect x="334" y="166" width="18" height="12" rx="1" fill="#1E3A8A" stroke="#0F172A" stroke-width="0.7"/>

    <!-- Micro-sprinkler -->
    <line x1="394" y1="{y_surf:.0f}" x2="434" y2="{y_surf:.0f}" stroke="#0F172A" stroke-width="1.8"/>
    <line x1="414" y1="{y_surf:.0f}" x2="414" y2="238" stroke="#0F172A" stroke-width="1.5"/>
    <rect x="411.5" y="234" width="5" height="4" fill="#0284C7"/>
    <polygon points="414,236 394,{y_surf:.0f} 434,{y_surf:.0f}" fill="#38BDF8" opacity="0.22"/>
    <line x1="414" y1="236" x2="400" y2="{y_surf - 4:.0f}" stroke="#0284C7" stroke-width="1.1" marker-end="url(#arr-water)"/>
    <line x1="414" y1="236" x2="428" y2="{y_surf - 4:.0f}" stroke="#0284C7" stroke-width="1.1" marker-end="url(#arr-water)"/>

    <!-- Solar Telemetry Mast -->
    <line x1="552" y1="272" x2="552" y2="205" stroke="#334155" stroke-width="1.8"/>
    <rect x="545" y="207" width="14" height="12" rx="1" fill="#F8FAFC" stroke="#334155" stroke-width="0.8"/>
    <polygon points="542,206 560,197 564,202 546,211" fill="#1E3A8A" stroke="#0F172A" stroke-width="0.7"/>
    <path d="M 494,256 Q 520,246 545,214" fill="none" stroke="#334155" stroke-width="1.1"/>

    <!-- Sentek Probe -->
    <rect x="478" y="252" width="24" height="140" rx="4" fill="#0284C7" fill-opacity="0.06" stroke="#0284C7" stroke-width="1.2" stroke-dasharray="4 3"/>
    <rect x="485" y="{y_surf:.0f}" width="10" height="{y_probe_bot - y_surf:.0f}" fill="#F8FAFC" stroke="#0F172A" stroke-width="1.1"/>
    <polygon points="485,{y_probe_bot:.0f} 495,{y_probe_bot:.0f} 490,{y_probe_bot + 8:.0f}" fill="#94A3B8" stroke="#0F172A" stroke-width="1.0"/>
    <rect x="483" y="{y_surf - 5:.0f}" width="14" height="5" rx="1.5" fill="#0F172A"/>
    {"".join(f'<rect x="483.5" y="{yc - 2.8:.1f}" width="13" height="5.6" rx="1" fill="#DC2626" stroke="#7F1D1D" stroke-width="0.6"/>' for yc in sensor_zs_a)}

    <!-- Flow Arrows -->
    <line x1="130" y1="55" x2="126" y2="100" stroke="#1D4ED8" stroke-width="1.2" stroke-dasharray="4 3"/>
    <line x1="142" y1="55" x2="138" y2="100" stroke="#1D4ED8" stroke-width="1.2" stroke-dasharray="4 3"/>
    <line x1="154" y1="55" x2="150" y2="105" stroke="#1D4ED8" stroke-width="1.8" marker-end="url(#arr-blue)"/>

    <line x1="545" y1="105" x2="545" y2="55" stroke="#D97706" stroke-width="1.9" marker-end="url(#arr-amber)"/>
    <line x1="625" y1="344" x2="625" y2="296" stroke="#D97706" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#arr-amber)"/>

    {"".join(lat_arrows_a)}

    <!-- Dimension Lines -->
    <line x1="162" y1="16" x2="162" y2="254" stroke="#CBD5E1" stroke-width="0.8" stroke-dasharray="3 3"/>
    <line x1="524" y1="16" x2="524" y2="254" stroke="#CBD5E1" stroke-width="0.8" stroke-dasharray="3 3"/>
    <line x1="166" y1="16" x2="228" y2="16" stroke="#475569" stroke-width="1.0" marker-start="url(#arr-slate)"/>
    <line x1="458" y1="16" x2="520" y2="16" stroke="#475569" stroke-width="1.0" marker-end="url(#arr-slate)"/>

    <line x1="456" y1="246" x2="488" y2="246" stroke="#0F172A" stroke-width="1.0" marker-start="url(#arr-slate)" marker-end="url(#arr-slate)"/>

    <!-- Leader Lines -->
    <line x1="232" y1="118" x2="232" y2="138" stroke="#475569" stroke-width="0.9"/>
    <circle cx="232" cy="138" r="2.2" fill="#0F172A"/>

    <line x1="454" y1="118" x2="454" y2="138" stroke="#475569" stroke-width="0.9"/>
    <circle cx="454" cy="138" r="2.2" fill="#0F172A"/>

    <line x1="343" y1="104" x2="343" y2="122" stroke="#475569" stroke-width="0.9"/>
    <circle cx="343" cy="122" r="2.2" fill="#0F172A"/>

    <line x1="388" y1="218" x2="414" y2="236" stroke="#0284C7" stroke-width="1.2"/>
    <circle cx="414" cy="236" r="2.2" fill="#0284C7"/>

    <line x1="582" y1="182" x2="558" y2="205" stroke="#475569" stroke-width="0.9"/>
    <circle cx="558" cy="205" r="2.2" fill="#0F172A"/>

    <line x1="82" y1="234" x2="108" y2="290" stroke="#0D9488" stroke-width="0.9"/>
    <circle cx="108" cy="290" r="2.2" fill="#0D9488"/>

    <line x1="320" y1="330" x2="320" y2="320" stroke="#475569" stroke-width="0.9"/>
    <circle cx="320" cy="320" r="2.2" fill="#475569"/>

    <line x1="320" y1="374" x2="320" y2="354" stroke="#475569" stroke-width="0.9"/>
    <circle cx="320" cy="354" r="2.2" fill="#0284C7"/>

    <line x1="62" y1="304" x2="62" y2="348" stroke="#475569" stroke-width="0.9"/>
    <circle cx="62" cy="348" r="2.2" fill="#0284C7"/>

    <!-- Panel a Vector Text & Badges -->
    <rect x="236" y="6" width="214" height="20" rx="3" fill="#FFFFFF"/>
    <text x="343" y="20" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0F172A" text-anchor="middle">Double-row raised bed (~5–6 m wide)</text>

    <text x="142" y="46" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#1D4ED8" text-anchor="middle">Rainfall (<tspan font-style="italic">r</tspan>)</text>
    <text x="545" y="46" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#D97706" text-anchor="middle">Evapotranspiration (<tspan font-style="italic">ET</tspan><tspan font-size="9" dy="2.5">0</tspan><tspan font-size="11.8" dy="-2.5">)</tspan></text>

    <text x="232" y="112" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#2D5A27" text-anchor="middle">Grapefruit tree (Row 1)</text>
    <text x="454" y="112" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#2D5A27" text-anchor="middle">Monitored grapefruit tree (Row 2)</text>

    <text x="343" y="100" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0F172A" text-anchor="middle">Weather station</text>

    <!-- Micro-sprinkler Pill Badge -->
    <g transform="translate(325, 196)">
      <rect x="0" y="0" width="80" height="28" rx="5" fill="#FFFFFF" stroke="#0284C7" stroke-width="1"/>
      <text x="40" y="12" font-family="Inter, Arial, sans-serif" font-size="11.0" font-weight="600" fill="#0284C7" text-anchor="middle">Micro-sprinkler</text>
      <text x="40" y="23" font-family="Inter, Arial, sans-serif" font-size="10.5" font-weight="600" fill="#0284C7" text-anchor="middle">(<tspan font-style="italic">q</tspan><tspan font-size="8.5" dy="2">irr</tspan><tspan font-size="10.5" dy="-2">)</tspan></text>
    </g>

    <text x="472" y="242" font-family="Inter, Arial, sans-serif" font-size="10.5" font-weight="700" fill="#0F172A" text-anchor="middle">20 cm</text>
    <text x="615" y="186" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0F172A" text-anchor="middle">Solar telemetry</text>

    <!-- Sentek Probe Badges -->
    <g transform="translate(416, 392)">
      <rect x="0" y="0" width="148" height="24" rx="5" fill="#FFFFFF" stroke="#0284C7" stroke-width="1"/>
      <text x="74" y="16.5" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="700" fill="#0284C7" text-anchor="middle">Sentek probe (5–55 cm)</text>
    </g>
    <g transform="translate(424, 420)">
      <rect x="0" y="0" width="132" height="24" rx="5" fill="#FFFFFF" stroke="#0284C7" stroke-width="1"/>
      <text x="66" y="16.5" font-family="Inter, Arial, sans-serif" font-size="10.5" font-weight="700" fill="#0284C7" text-anchor="middle">Expanded in Panel <tspan font-weight="800">b</tspan> &#8594;</text>
    </g>

    <text x="82" y="214" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0D9488" text-anchor="middle">Lateral drainage</text>
    <text x="82" y="228" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0D9488" text-anchor="middle"><tspan font-style="italic">fl</tspan><tspan font-size="9" dy="2">1</tspan><tspan font-size="11.8" dy="-2"> ... </tspan><tspan font-style="italic">fl</tspan><tspan font-size="9" dy="2">5</tspan></text>

    <!-- Flanking Ditch Badge -->
    <g transform="translate(18, 274)">
      <rect x="0" y="0" width="88" height="28" rx="4" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
      <text x="44" y="12" font-family="Inter, Arial, sans-serif" font-size="11.0" font-weight="600" fill="#0284C7" text-anchor="middle">Flanking</text>
      <text x="44" y="23" font-family="Inter, Arial, sans-serif" font-size="11.0" font-weight="600" fill="#0284C7" text-anchor="middle">drainage ditch</text>
    </g>
    <text x="625" y="284" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#D97706" text-anchor="middle">Ditch evaporation</text>

    <!-- Topsoil & Clay Labels -->
    <rect x="254" y="278" width="178" height="20" rx="3" fill="#FBF9F5" opacity="0.92"/>
    <text x="343" y="293" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0F172A" text-anchor="middle">Sandy topsoil (Alfisol, 0–30 cm)</text>

    <text x="320" y="342" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0F172A" text-anchor="middle">Argillic clay horizon (<tspan font-style="italic">z</tspan> ≈ 30 cm; low permeability)</text>
    <text x="320" y="386" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0284C7" text-anchor="middle">Local water table / capillary fringe</text>
  </g>

  <!-- ========================================================================= -->
  <!-- CARD b (Panel b: Right, 534 x 556 px) -->
  <!-- ========================================================================= -->
  <rect x="734" y="12" width="534" height="556" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
  <text x="746" y="33" font-family="Inter, Arial, sans-serif" font-size="14.5" font-weight="700" fill="#0F172A">
    <tspan font-size="16" font-weight="800">b</tspan>  1D finite-volume PINN model (Eq. 8)
  </text>

  <g transform="translate(746, 46)">
    <!-- Depth Axis -->
    <line x1="30" y1="38" x2="30" y2="430" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    {"".join(c_ticks_svg)}

    <!-- 6 Finite-Volume Compartments -->
    {"".join(c_compartments)}
    <rect x="{c_x0}" y="{c_y0}" width="{c_x1 - c_x0}" height="{6 * c_dz}" fill="none" stroke="#0F172A" stroke-width="1.3"/>

    <!-- Argillic Horizon Boundary -->
    <line x1="94" y1="{c_y0 + 3 * c_dz:.1f}" x2="234" y2="{c_y0 + 3 * c_dz:.1f}" stroke="#475569" stroke-width="1.8" stroke-dasharray="5 3"/>

    <!-- Surface Inflow Arrows -->
    <line x1="132" y1="20" x2="132" y2="{c_y0 - 2:.1f}" stroke="#0284C7" stroke-width="1.9" marker-end="url(#arr-water)"/>
    <line x1="194" y1="20" x2="194" y2="{c_y0 - 2:.1f}" stroke="#1D4ED8" stroke-width="1.9" marker-end="url(#arr-blue)"/>
    <path d="M 52,94 Q 54,116 64,133" fill="none" stroke="#1D4ED8" stroke-width="1.0" stroke-dasharray="2.5 2.5" marker-end="url(#arr-blue)"/>

    <!-- Arrows & Fluxes -->
    {"".join(c_flux_arrows)}

    <!-- Headers & Top Labels -->
    <text x="28" y="28" font-family="Inter, Arial, sans-serif" font-size="11.2" font-weight="600" fill="#0F172A" text-anchor="middle"><tspan font-style="italic">z</tspan> (cm)</text>

    <!-- Direct Rainfall Left Header -->
    <text x="52" y="70" font-family="Inter, Arial, sans-serif" font-size="10.5" font-weight="600" fill="#1D4ED8" text-anchor="middle">
      Direct<tspan x="52" dy="12">rainfall</tspan><tspan x="52" dy="13"><tspan font-style="italic">fr</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.5" dy="-2">(<tspan font-style="italic">r</tspan>)</tspan></tspan>
    </text>

    <text x="132" y="16" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#0284C7" text-anchor="middle"><tspan font-style="italic">fq</tspan><tspan font-size="9" dy="2.5">irr</tspan><tspan font-size="11.8" dy="-2.5">(<tspan font-style="italic">q</tspan></tspan><tspan font-size="9" dy="2.5">irr</tspan><tspan font-size="11.8" dy="-2.5">)</tspan></text>
    <text x="194" y="16" font-family="Inter, Arial, sans-serif" font-size="11.8" font-weight="600" fill="#1D4ED8" text-anchor="middle"><tspan font-style="italic">fr</tspan><tspan font-size="9" dy="2.5">1</tspan><tspan font-size="11.8" dy="-2.5">(<tspan font-style="italic">r</tspan>)</tspan></text>

    <text x="248" y="16" font-family="Inter, Arial, sans-serif" font-size="10.6" font-weight="600" fill="#D97706">
      Evapotranspirative sink<tspan x="248" dy="13"><tspan font-style="italic">S</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.6" dy="-2">(<tspan font-style="italic">m</tspan></tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.6" dy="-2">, <tspan font-style="italic">ET</tspan></tspan><tspan font-size="8.5" dy="2">0</tspan><tspan font-size="10.6" dy="-2">) [<tspan font-style="italic">pc</tspan></tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.6" dy="-2">]</tspan></tspan>
    </text>

    <!-- Argillic Horizon Label -->
    <rect x="22" y="218" width="76" height="28" rx="2" fill="#FFFFFF"/>
    <text x="98" y="228" font-family="Inter, Arial, sans-serif" font-size="9.8" font-weight="600" fill="#0F172A" text-anchor="end">
      Argillic horizon<tspan x="98" dy="12" font-size="9.5" font-weight="500" fill="#475569">(30 cm)</tspan>
    </text>

    <!-- Labels from Loop -->
    {"".join(c_labels_svg)}
  </g>

  <!-- Horizontal Legend Bar -->
  <g transform="translate(746, 514)">
    <rect x="0" y="0" width="510" height="42" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"/>
    <!-- Col 1 -->
    <line x1="12" y1="13" x2="28" y2="13" stroke="#0284C7" stroke-width="1.8"/>
    <polygon points="26,10 32,13 26,16" fill="#0284C7"/>
    <text x="36" y="17" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">fq</tspan><tspan font-size="8.5" dy="2">irr</tspan><tspan font-size="10.8" dy="-2">&#160;Irrigation</tspan></text>

    <line x1="12" y1="29" x2="28" y2="29" stroke="#1D4ED8" stroke-width="1.6" stroke-dasharray="3 2"/>
    <polygon points="26,26 32,29 26,32" fill="#1D4ED8"/>
    <text x="36" y="33" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">fr</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.8" dy="-2">(<tspan font-style="italic">r</tspan>)&#160;Direct rainfall</tspan></text>

    <!-- Col 2 -->
    <line x1="182" y1="13" x2="198" y2="13" stroke="#1D4ED8" stroke-width="1.8"/>
    <polygon points="196,10 202,13 196,16" fill="#1D4ED8"/>
    <text x="206" y="17" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">f</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.8" dy="-2">&#160;≥ 0 Gravity drainage</tspan></text>

    <line x1="184" y1="29" x2="198" y2="29" stroke="#6D28D9" stroke-width="1.6" stroke-dasharray="2.5 1.5"/>
    <polygon points="185,26 180,29 185,32" fill="#6D28D9"/>
    <polygon points="197,26 202,29 197,32" fill="#6D28D9"/>
    <text x="206" y="33" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">f</tspan><tspan font-size="8.5" dy="2">i,i+1</tspan><tspan font-size="10.8" dy="-2">&#160;≷ 0 Matric exchange</tspan></text>

    <!-- Col 3 -->
    <line x1="362" y1="13" x2="378" y2="13" stroke="#0D9488" stroke-width="1.8"/>
    <polygon points="376,10 382,13 376,16" fill="#0D9488"/>
    <text x="386" y="17" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">fl</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.8" dy="-2">&#160;≥ 0 Ditch outflow</tspan></text>

    <line x1="362" y1="29" x2="378" y2="29" stroke="#D97706" stroke-width="1.8"/>
    <polygon points="376,26 382,29 376,32" fill="#D97706"/>
    <text x="386" y="33" font-family="Inter, Arial, sans-serif" font-size="10.8" font-weight="500" fill="#334155"><tspan font-style="italic">S</tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.8" dy="-2">&#160;≡ <tspan font-style="italic">pc</tspan></tspan><tspan font-size="8.5" dy="2">i</tspan><tspan font-size="10.8" dy="-2">&#160;ET sink</tspan></text>
  </g>
</svg>
"""
    return svg

def create_element_by_element_drawio():
    """
    Creates a 100% native Draw.io XML where EVERY SINGLE ELEMENT
    (soil layers, ditches, probe, trees, roots, weather station, sprinkler,
    all 6 compartments, all 32 arrows, axes, brackets, badges, and legend keys)
    is an individual, selectable, editable, draggable mxCell node.
    """
    y_surf = 260.0
    y_clay = 320.0
    y_water = 348.0
    y_ditch = 364.0
    y_probe_bot = 380.0

    # Panel a offsets
    ax0, ay0 = 24.0, 46.0

    # Botanical foliage defs for isolated tree SVG assets
    foliage_defs = '''<defs>
  <linearGradient id="lf-dark" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#1B4332"/><stop offset="100%" stop-color="#2D6A4F"/>
  </linearGradient>
  <linearGradient id="lf-mid" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#2D6A4F"/><stop offset="100%" stop-color="#40916C"/>
  </linearGradient>
  <linearGradient id="lf-sun" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#40916C"/><stop offset="100%" stop-color="#52B788"/>
  </linearGradient>
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
</defs>'''

    # Isolated SVG for Tree 1 (canopy & trunk: width=144, height=140)
    raw_t1 = build_botanical_tree_svg(232.0, 260.0, seed=42, scale=1.22)
    svg_t1 = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="160 120 144 140" width="144" height="140">{foliage_defs}{raw_t1}</svg>'
    enc_t1 = urllib.parse.quote(svg_t1)

    # Isolated SVG for Roots 1 (width=84, height=60)
    raw_r1 = build_fibrous_roots_svg(232.0, 260.0, 320.0, x_min=190.0, x_max=274.0, seed=51)
    svg_r1 = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="190 260 84 60" width="84" height="60">{raw_r1}</svg>'
    enc_r1 = urllib.parse.quote(svg_r1)

    # Isolated SVG for Tree 2 (width=144, height=140)
    raw_t2 = build_botanical_tree_svg(454.0, 260.0, seed=77, scale=1.22)
    svg_t2 = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="382 120 144 140" width="144" height="140">{foliage_defs}{raw_t2}</svg>'
    enc_t2 = urllib.parse.quote(svg_t2)

    # Isolated SVG for Roots 2 (width=84, height=60)
    raw_r2 = build_fibrous_roots_svg(454.0, 260.0, 320.0, x_min=412.0, x_max=484.0, seed=88)
    svg_r2 = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="412 260 84 60" width="84" height="60">{raw_r2}</svg>'
    enc_r2 = urllib.parse.quote(svg_r2)

    # Isolated SVG for Topsoil Sand Profile (width=686, height=504)
    bed_surface_d = (
        f"M 0,312 C 14,312 26,{y_ditch:.0f} 44,{y_ditch:.0f} L 82,{y_ditch:.0f} "
        f"C 112,{y_ditch:.0f} 132,{y_surf:.0f} 162,{y_surf:.0f} "
        f"L 524,{y_surf:.0f} "
        f"C 554,{y_surf:.0f} 574,{y_ditch:.0f} 604,{y_ditch:.0f} L 642,{y_ditch:.0f} "
        f"C 660,{y_ditch:.0f} 672,312 686,312"
    )
    topsoil_fill_d = bed_surface_d + " L 686,504 L 0,504 Z"
    svg_sand = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 686 504" width="686" height="504">'
        f'<defs>'
        f'<linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#FBF9F5"/><stop offset="100%" stop-color="#EFECE4"/></linearGradient>'
        f'<pattern id="sd" width="14" height="14" patternUnits="userSpaceOnUse">'
        f'<circle cx="3" cy="4" r="0.65" fill="#D6D0C4"/><circle cx="10" cy="11" r="0.6" fill="#D6D0C4"/>'
        f'</pattern>'
        f'<clipPath id="cp"><path d="{topsoil_fill_d}"/></clipPath>'
        f'</defs>'
        f'<g clip-path="url(#cp)">'
        f'<rect x="0" y="0" width="686" height="504" fill="url(#sg)"/>'
        f'<rect x="0" y="0" width="686" height="504" fill="url(#sd)"/>'
        f'</g>'
        f'</svg>'
    )
    enc_sand = urllib.parse.quote(svg_sand)

    # Isolated SVG for Clay Layer Profile
    clay_xs = np.linspace(0, 686, 120)
    clay_ys = y_clay + 2.0 * np.sin(0.024 * clay_xs + 0.4) + 1.0 * np.cos(0.05 * clay_xs)
    clay_fill_d = f"M 0,504 L " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(clay_xs, clay_ys)) + " L 686,504 Z"
    svg_clay = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 686 504" width="686" height="504">'
        f'<defs>'
        f'<linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#DFD5C8"/><stop offset="100%" stop-color="#CBBBA8"/></linearGradient>'
        f'<clipPath id="cp"><path d="{topsoil_fill_d}"/></clipPath>'
        f'</defs>'
        f'<g clip-path="url(#cp)">'
        f'<path d="{clay_fill_d}" fill="url(#cg)"/>'
        f'</g>'
        f'</svg>'
    )
    enc_clay = urllib.parse.quote(svg_clay)

    cells = []

    # Helper functions for clean Draw.io cell emission
    def add_vertex(cell_id, value, style, x, y, w, h, parent="1"):
        val_esc = html.escape(value).replace("\n", "&#xa;")
        cells.append(
            f'        <mxCell id="{cell_id}" value="{val_esc}" style="{style}" vertex="1" parent="{parent}">\n'
            f'          <mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>\n'
            f'        </mxCell>'
        )

    def add_edge(edge_id, value, style, x1, y1, x2, y2, waypoints=None, parent="1"):
        val_esc = html.escape(value).replace("\n", "&#xa;")
        wp_str = ""
        if waypoints:
            pts = "".join(f'<mxPoint x="{wx:.1f}" y="{wy:.1f}"/>' for wx, wy in waypoints)
            wp_str = f'<Array as="points">{pts}</Array>'
        cells.append(
            f'        <mxCell id="{edge_id}" value="{val_esc}" style="{style}" edge="1" parent="{parent}">\n'
            f'          <mxGeometry relative="1" as="geometry">\n'
            f'            <mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>\n'
            f'            <mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>\n'
            f'            {wp_str}\n'
            f'          </mxGeometry>\n'
            f'        </mxCell>'
        )

    # =========================================================================
    # 1. OUTER CARDS & HEADERS
    # =========================================================================
    add_vertex("card_a", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1;arcSize=4;", 12, 12, 710, 556)
    add_vertex("hdr_a", "<b><font style=\"font-size: 16px;\">a</font></b>&nbsp;&nbsp;<b><font style=\"font-size: 14.5px;\">Double-row raised-bed &amp; drainage-ditch cross-section</font></b>",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;fontColor=#0F172A;fontFamily=Helvetica;", 24, 20, 600, 26)

    add_vertex("card_b", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1;arcSize=4;", 734, 12, 534, 556)
    add_vertex("hdr_b", "<b><font style=\"font-size: 16px;\">b</font></b>&nbsp;&nbsp;<b><font style=\"font-size: 14.5px;\">1D finite-volume PINN model (Eq. 8)</font></b>",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;fontColor=#0F172A;fontFamily=Helvetica;", 746, 20, 450, 26)

    # =========================================================================
    # 2. PANEL a: SOIL PROFILE & TERRAIN (Discrete native/isolated elements)
    # =========================================================================
    add_vertex("soil_sand_layer", "", f"shape=image;image=data:image/svg+xml,{enc_sand};verticalAlign=top;aspect=fixed;imageAspect=0;", ax0, ay0, 686, 504)
    add_vertex("soil_clay_layer", "", f"shape=image;image=data:image/svg+xml,{enc_clay};verticalAlign=top;aspect=fixed;imageAspect=0;", ax0, ay0, 686, 504)

    # Ditch Water bodies (Trapezoids)
    add_vertex("ditch_water_l", "", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;flipV=1;fillColor=#38BDF8;strokeColor=#0284C7;fillOpacity=60;strokeWidth=1;",
               ax0 + 25, ay0 + y_water, 74, y_ditch - y_water)
    add_vertex("ditch_water_r", "", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;flipV=1;fillColor=#38BDF8;strokeColor=#0284C7;fillOpacity=60;strokeWidth=1;",
               ax0 + 587, ay0 + y_water, 74, y_ditch - y_water)

    # Water Surface Markers
    add_edge("water_surf_line_l", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.3;endArrow=none;", ax0 + 25, ay0 + y_water, ax0 + 99, ay0 + y_water)
    add_vertex("water_marker_l", "", "shape=triangle;direction=south;fillColor=#0284C7;strokeColor=none;", ax0 + 58, ay0 + y_water - 5, 8, 5)

    add_edge("water_surf_line_r", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.3;endArrow=none;", ax0 + 587, ay0 + y_water, ax0 + 661, ay0 + y_water)
    add_vertex("water_marker_r", "", "shape=triangle;direction=south;fillColor=#0284C7;strokeColor=none;", ax0 + 620, ay0 + y_water - 5, 8, 5)

    # Boundaries: Ground surface outline, Clay horizon, Local Water Table
    add_edge("line_ground_surface", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.4;endArrow=none;curved=1;",
             ax0, ay0 + 312, ax0 + 686, ay0 + 312,
             waypoints=[(ax0 + 44, ay0 + y_ditch), (ax0 + 82, ay0 + y_ditch), (ax0 + 162, ay0 + y_surf),
                        (ax0 + 524, ay0 + y_surf), (ax0 + 604, ay0 + y_ditch), (ax0 + 642, ay0 + y_ditch)])

    add_edge("line_clay_horizon", "", "edgeStyle=none;html=1;strokeColor=#64748B;strokeWidth=1.4;dashed=1;dashPattern=6 3.5;endArrow=none;",
             ax0, ay0 + y_clay, ax0 + 686, ay0 + y_clay)

    add_edge("line_water_table", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.5;dashed=1;dashPattern=5 3;endArrow=none;curved=1;",
             ax0 + 46, ay0 + 360, ax0 + 640, ay0 + 360, waypoints=[(ax0 + 343, ay0 + 352)])

    # Turf Ribbons (left, mid, right)
    add_edge("turf_left", "", "edgeStyle=none;html=1;strokeColor=#4D7C4D;strokeWidth=4.5;endArrow=none;curved=1;",
             ax0 + 95, ay0 + 356, ax0 + 174, ay0 + y_surf, waypoints=[(ax0 + 132, ay0 + y_surf)])
    add_edge("turf_mid", "", "edgeStyle=none;html=1;strokeColor=#4D7C4D;strokeWidth=4.5;endArrow=none;",
             ax0 + 296, ay0 + y_surf, ax0 + 390, ay0 + y_surf)
    add_edge("turf_right", "", "edgeStyle=none;html=1;strokeColor=#4D7C4D;strokeWidth=4.5;endArrow=none;curved=1;",
             ax0 + 512, ay0 + y_surf, ax0 + 591, ay0 + 356, waypoints=[(ax0 + 554, ay0 + y_surf)])

    # =========================================================================
    # 3. PANEL a: BOTANICAL TREES & ROOT NETWORKS (Selectable elements)
    # =========================================================================
    # Tree 1 & Roots 1
    add_vertex("roots_row1", "", f"shape=image;image=data:image/svg+xml,{enc_r1};verticalAlign=top;aspect=fixed;", ax0 + 190, ay0 + 260, 84, 60)
    add_vertex("tree_row1", "", f"shape=image;image=data:image/svg+xml,{enc_t1};verticalAlign=top;aspect=fixed;", ax0 + 160, ay0 + 120, 144, 140)
    add_vertex("lbl_tree_row1", "<b>Grapefruit tree (Row 1)</b>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=12;fontColor=#2D5A27;fontFamily=Helvetica;",
               ax0 + 156, ay0 + 98, 152, 24)
    add_edge("ldr_tree_row1", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 232, ay0 + 118, ax0 + 232, ay0 + 138)

    # Tree 2 & Roots 2
    add_vertex("roots_row2", "", f"shape=image;image=data:image/svg+xml,{enc_r2};verticalAlign=top;aspect=fixed;", ax0 + 412, ay0 + 260, 84, 60)
    add_vertex("tree_row2", "", f"shape=image;image=data:image/svg+xml,{enc_t2};verticalAlign=top;aspect=fixed;", ax0 + 382, ay0 + 120, 144, 140)
    add_vertex("lbl_tree_row2", "<b>Monitored grapefruit tree (Row 2)</b>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=12;fontColor=#2D5A27;fontFamily=Helvetica;",
               ax0 + 351, ay0 + 98, 206, 24)
    add_edge("ldr_tree_row2", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 454, ay0 + 118, ax0 + 454, ay0 + 138)

    # =========================================================================
    # 4. PANEL a: FIELD INSTRUMENTS (Discrete components)
    # =========================================================================
    # Weather Station (Davis Vantage Pro2)
    add_edge("weather_mast", "", "edgeStyle=none;html=1;strokeColor=#334155;strokeWidth=2.0;endArrow=none;", ax0 + 343, ay0 + y_surf, ax0 + 343, ay0 + 122)
    add_edge("weather_crossbar", "", "edgeStyle=none;html=1;strokeColor=#334155;strokeWidth=1.4;endArrow=none;", ax0 + 329, ay0 + 129, ax0 + 356, ay0 + 129)
    add_vertex("weather_cup", "", "shape=ellipse;fillColor=#0F172A;strokeColor=none;", ax0 + 327, ay0 + 125, 4.4, 4.4)
    add_vertex("weather_shield", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#475569;strokeWidth=0.7;", ax0 + 339, ay0 + 143, 8, 10)
    add_vertex("weather_panel", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#1E3A8A;strokeColor=#0F172A;strokeWidth=0.7;arcSize=4;", ax0 + 334, ay0 + 166, 18, 12)
    add_vertex("lbl_weather", "<b>Weather station</b>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=12;fontColor=#0F172A;fontFamily=Helvetica;",
               ax0 + 288, ay0 + 86, 110, 24)
    add_edge("ldr_weather", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 343, ay0 + 104, ax0 + 343, ay0 + 122)

    # Micro-sprinkler
    add_edge("sprinkler_pipe", "", "edgeStyle=none;html=1;strokeColor=#0F172A;strokeWidth=1.8;endArrow=none;", ax0 + 394, ay0 + y_surf, ax0 + 434, ay0 + y_surf)
    add_edge("sprinkler_riser", "", "edgeStyle=none;html=1;strokeColor=#0F172A;strokeWidth=1.5;endArrow=none;", ax0 + 414, ay0 + y_surf, ax0 + 414, ay0 + 238)
    add_vertex("sprinkler_nozzle", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=none;", ax0 + 411.5, ay0 + 234, 5, 4)
    add_vertex("sprinkler_cone", "", "shape=triangle;direction=south;fillColor=#38BDF8;strokeColor=none;opacity=22;", ax0 + 394, ay0 + 236, 40, y_surf - 236)
    add_edge("sprinkler_spray_l", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.1;endArrow=classic;endFill=1;endSize=4.5;",
             ax0 + 414, ay0 + 236, ax0 + 400, ay0 + y_surf - 4)
    add_edge("sprinkler_spray_r", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.1;endArrow=classic;endFill=1;endSize=4.5;",
             ax0 + 414, ay0 + 236, ax0 + 428, ay0 + y_surf - 4)
    add_vertex("badge_sprinkler", "<b>Micro-sprinkler</b><br>(<i>q</i><sub>irr</sub>)",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#0284C7;strokeWidth=1;fontColor=#0284C7;fontFamily=Helvetica;fontSize=11;align=center;",
               ax0 + 325, ay0 + 196, 80, 28)
    add_edge("ldr_sprinkler", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.2;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 388, ay0 + 218, ax0 + 414, ay0 + 236)

    # Sentek Probe (60 cm)
    add_vertex("probe_zoom_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#0284C7;fillOpacity=6;strokeColor=#0284C7;strokeWidth=1.2;dashed=1;dashPattern=4 3;",
               ax0 + 478, ay0 + 252, 24, 140)
    add_vertex("probe_tube", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#0F172A;strokeWidth=1.1;",
               ax0 + 485, ay0 + y_surf, 10, y_probe_bot - y_surf)
    add_vertex("probe_tip", "", "shape=triangle;direction=south;fillColor=#94A3B8;strokeColor=#0F172A;strokeWidth=1.0;",
               ax0 + 485, ay0 + y_probe_bot, 10, 8)
    add_vertex("probe_cap", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=none;",
               ax0 + 483, ay0 + y_surf - 5, 14, 5)

    sensor_zs_a = [270.0, 290.0, 310.0, 330.0, 350.0, 370.0]
    for s_idx, yc in enumerate(sensor_zs_a, start=1):
        add_vertex(f"probe_sens_{s_idx}", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#DC2626;strokeColor=#7F1D1D;strokeWidth=0.6;",
                   ax0 + 483.5, ay0 + yc - 2.8, 13, 5.6)

    add_vertex("badge_probe_1", "<b>Sentek probe (5–55 cm)</b>",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#0284C7;strokeWidth=1;fontColor=#0284C7;fontFamily=Helvetica;fontSize=10.8;fontStyle=1;align=center;",
               ax0 + 416, ay0 + 392, 148, 24)
    add_vertex("badge_probe_2", "<b>Expanded in Panel b &rarr;</b>",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#0284C7;strokeWidth=1;fontColor=#0284C7;fontFamily=Helvetica;fontSize=10.5;fontStyle=1;align=center;",
               ax0 + 424, ay0 + 420, 132, 24)

    # 20 cm dimension line
    add_edge("dim_probe_20cm", "", "edgeStyle=none;html=1;strokeColor=#0F172A;strokeWidth=1.0;startArrow=classic;endArrow=classic;startFill=1;endFill=1;startSize=4;endSize=4;",
             ax0 + 456, ay0 + 246, ax0 + 488, ay0 + 246)
    add_vertex("lbl_probe_20cm", "<b>20 cm</b>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#0F172A;fontFamily=Helvetica;",
               ax0 + 454, ay0 + 230, 36, 18)

    # Solar Telemetry
    add_edge("solar_mast", "", "edgeStyle=none;html=1;strokeColor=#334155;strokeWidth=1.8;endArrow=none;", ax0 + 552, ay0 + 272, ax0 + 552, ay0 + 205)
    add_vertex("solar_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#334155;strokeWidth=0.8;arcSize=4;", ax0 + 545, ay0 + 207, 14, 12)
    add_vertex("solar_panel", "", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fillColor=#1E3A8A;strokeColor=#0F172A;strokeWidth=0.7;rotation=15;",
               ax0 + 542, ay0 + 197, 20, 12)
    add_edge("solar_cable", "", "edgeStyle=none;html=1;strokeColor=#334155;strokeWidth=1.1;endArrow=none;curved=1;",
             ax0 + 494, ay0 + 256, ax0 + 545, ay0 + 214, waypoints=[(ax0 + 520, ay0 + 246)])
    add_vertex("lbl_solar", "<b>Solar telemetry</b>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontColor=#0F172A;fontFamily=Helvetica;",
               ax0 + 560, ay0 + 174, 110, 24)
    add_edge("ldr_solar", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 582, ay0 + 182, ax0 + 558, ay0 + 205)

    # =========================================================================
    # 5. PANEL a: HYDROLOGICAL ARROWS, DIMENSIONS & BADGES
    # =========================================================================
    # Bed width dimension line
    add_edge("dim_bed_ext_l", "", "edgeStyle=none;html=1;strokeColor=#CBD5E1;strokeWidth=0.8;dashed=1;dashPattern=3 3;endArrow=none;",
             ax0 + 162, ay0 + 16, ax0 + 162, ay0 + 254)
    add_edge("dim_bed_ext_r", "", "edgeStyle=none;html=1;strokeColor=#CBD5E1;strokeWidth=0.8;dashed=1;dashPattern=3 3;endArrow=none;",
             ax0 + 524, ay0 + 16, ax0 + 524, ay0 + 254)
    add_edge("dim_bed_line_l", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.0;startArrow=classic;startFill=1;startSize=4.5;endArrow=none;",
             ax0 + 166, ay0 + 16, ax0 + 228, ay0 + 16)
    add_edge("dim_bed_line_r", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.0;endArrow=classic;endFill=1;endSize=4.5;",
             ax0 + 458, ay0 + 16, ax0 + 520, ay0 + 16)
    add_vertex("badge_bed_width", "Double-row raised bed (~5–6 m wide)",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;fontColor=#0F172A;fontFamily=Helvetica;fontSize=11.8;fontStyle=1;align=center;",
               ax0 + 236, ay0 + 6, 214, 20)

    # Rainfall arrows
    add_edge("arr_rain_a1", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.2;dashed=1;dashPattern=4 3;endArrow=none;", ax0 + 130, ay0 + 55, ax0 + 126, ay0 + 100)
    add_edge("arr_rain_a2", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.2;dashed=1;dashPattern=4 3;endArrow=none;", ax0 + 142, ay0 + 55, ax0 + 138, ay0 + 100)
    add_edge("arr_rain_a3", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=5;", ax0 + 154, ay0 + 55, ax0 + 150, ay0 + 105)
    add_vertex("lbl_rain_a", "Rainfall (<i>r</i>)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
               ax0 + 100, ay0 + 34, 84, 22)

    # Evapotranspiration
    add_edge("arr_et_a", "", "edgeStyle=none;html=1;strokeColor=#D97706;strokeWidth=1.9;endArrow=classic;endFill=1;endSize=5.5;", ax0 + 545, ay0 + 105, ax0 + 545, ay0 + 55)
    add_vertex("lbl_et_a", "Evapotranspiration (<i>ET</i><sub>0</sub>)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#D97706;fontFamily=Helvetica;",
               ax0 + 460, ay0 + 34, 170, 22)

    # Ditch evaporation
    add_edge("arr_et_ditch", "", "edgeStyle=none;html=1;strokeColor=#D97706;strokeWidth=1.5;dashed=1;dashPattern=3 2;endArrow=classic;endFill=1;endSize=4.5;", ax0 + 625, ay0 + 344, ax0 + 625, ay0 + 296)
    add_vertex("lbl_et_ditch", "Ditch evaporation", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#D97706;fontFamily=Helvetica;",
               ax0 + 565, ay0 + 272, 120, 22)

    # Lateral drainage arrows on bed slopes (10 arrows)
    for z_cm in (5, 15, 25, 35, 45):
        yy = y_surf + z_cm * 2.0
        frac = (yy - y_surf) / (y_ditch - y_surf)
        xl = 162.0 - 80.0 * frac
        xr = 524.0 + 80.0 * frac
        add_edge(f"arr_lat_l_{z_cm}", "", "edgeStyle=none;html=1;strokeColor=#0D9488;strokeWidth=1.6;endArrow=classic;endFill=1;endSize=4.5;",
                 ax0 + xl, ay0 + yy, ax0 + xl - 24, ay0 + yy)
        add_edge(f"arr_lat_r_{z_cm}", "", "edgeStyle=none;html=1;strokeColor=#0D9488;strokeWidth=1.6;endArrow=classic;endFill=1;endSize=4.5;",
                 ax0 + xr, ay0 + yy, ax0 + xr + 24, ay0 + yy)

    add_vertex("lbl_lat_drain_a", "<b>Lateral drainage</b><br><i>fl</i><sub>1</sub> ... <i>fl</i><sub>5</sub>",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontColor=#0D9488;fontFamily=Helvetica;",
               ax0 + 27, ay0 + 202, 110, 32)
    add_edge("ldr_lat_drain", "", "edgeStyle=none;html=1;strokeColor=#0D9488;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 82, ay0 + 234, ax0 + 108, ay0 + 290)

    # Flanking Ditch badge
    add_vertex("badge_ditch", "<b>Flanking</b><br>drainage ditch",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1;fontColor=#0284C7;fontFamily=Helvetica;fontSize=11;align=center;",
               ax0 + 18, ay0 + 274, 88, 28)
    add_edge("ldr_ditch", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 62, ay0 + 304, ax0 + 62, ay0 + 348)

    # Topsoil Sandy label
    add_vertex("lbl_topsoil_sand", "Sandy topsoil (Alfisol, 0–30 cm)",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FBF9F5;strokeColor=none;fontColor=#0F172A;fontFamily=Helvetica;fontSize=11.8;fontStyle=1;align=center;fillOpacity=92;",
               ax0 + 254, ay0 + 278, 178, 20)

    # Clay Horizon label
    add_vertex("lbl_clay_horizon", "Argillic clay horizon (<i>z</i> ≈ 30 cm; low permeability)",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#0F172A;fontFamily=Helvetica;",
               ax0 + 175, ay0 + 330, 290, 22)
    add_edge("ldr_clay_horizon", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 320, ay0 + 330, ax0 + 320, ay0 + 320)

    # Local Water Table label
    add_vertex("lbl_water_table", "Local water table / capillary fringe",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#0284C7;fontFamily=Helvetica;",
               ax0 + 220, ay0 + 374, 200, 22)
    add_edge("ldr_water_table", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=0.9;endArrow=oval;endFill=1;endSize=3.5;",
             ax0 + 320, ay0 + 374, ax0 + 320, ay0 + 354)

    # =========================================================================
    # 6. PANEL b: 1D FINITE-VOLUME ODE PINN MODEL (100% Native Draw.io)
    # =========================================================================
    bx0, by0 = 746.0, 46.0
    c_y0 = 46.0
    c_dz = 62.0
    c_x0, c_x1 = 104.0, 222.0
    c_xc = 163.0

    # Depth Axis
    add_edge("axis_z_col", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.2;endArrow=classic;endFill=1;endSize=5;",
             bx0 + 30, by0 + 38, bx0 + 30, by0 + 430)
    add_vertex("lbl_z_axis", "<i>z</i> (cm)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.2;fontStyle=1;fontColor=#0F172A;fontFamily=Helvetica;",
               bx0 + 7, by0 + 16, 46, 20)

    for z_val in range(0, 61, 10):
        yy = c_y0 + (z_val / 10.0) * c_dz
        add_edge(f"tick_z_{z_val}", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.0;endArrow=none;",
                 bx0 + 24, by0 + yy, bx0 + 30, by0 + yy)
        add_vertex(f"lbl_tick_{z_val}", str(z_val), "text;html=1;strokeColor=none;fillColor=none;align=right;verticalAlign=middle;fontSize=10.8;fontColor=#475569;fontFamily=Helvetica;",
                   bx0 + 2, by0 + yy - 8, 18, 16)

    # 6 Soil Compartments
    sand_fill = "#FBF9F5"
    clay_fill = "#DFD5C8"
    for idx, z_mid in enumerate((5, 15, 25, 35, 45, 55), start=1):
        yt = c_y0 + (idx - 1) * c_dz
        ym = yt + 0.5 * c_dz
        c_fill = sand_fill if idx <= 3 else clay_fill
        comp_val = f"<b>z<sub>{idx}</sub> = {z_mid} cm (m<sub>{idx}</sub>)</b>"
        add_vertex(f"comp_{idx}", comp_val,
                   f"rounded=0;whiteSpace=wrap;html=1;fillColor={c_fill};strokeColor=#94A3B8;strokeWidth=0.9;fontColor=#0F172A;fontFamily=Helvetica;fontSize=11.8;align=center;",
                   bx0 + c_x0, by0 + yt, c_x1 - c_x0, c_dz)

        # Red sensor ring at left edge of each compartment
        add_vertex(f"sens_ring_{idx}", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#DC2626;strokeColor=#7F1D1D;strokeWidth=0.6;arcSize=3;",
                   bx0 + c_x0 + 3, by0 + ym - 6, 7, 12)

    # Layer 6 water tint overlay
    add_vertex("comp_6_water_tint", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#38BDF8;fillOpacity=14;strokeColor=none;",
               bx0 + c_x0 + 1, by0 + c_y0 + 5 * c_dz + 1, c_x1 - c_x0 - 2, c_dz - 2)

    # Outer border around entire column
    add_vertex("comp_column_border", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#0F172A;strokeWidth=1.3;",
               bx0 + c_x0, by0 + c_y0, c_x1 - c_x0, 6 * c_dz)

    # Argillic Horizon Boundary Line & Badge
    add_edge("line_col_argillic", "", "edgeStyle=none;html=1;strokeColor=#475569;strokeWidth=1.8;dashed=1;dashPattern=5 3;endArrow=none;",
             bx0 + 94, by0 + c_y0 + 3 * c_dz, bx0 + 234, by0 + c_y0 + 3 * c_dz)
    add_vertex("lbl_col_argillic", "<b>Argillic horizon</b><br><font color=\"#475569\">(30 cm)</font>",
               "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;fontColor=#0F172A;fontFamily=Helvetica;fontSize=9.8;align=right;",
               bx0 + 22, by0 + 218, 76, 28)

    # Top Inflow Arrows: fq_irr and fr_1
    add_edge("arr_fq_irr", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.9;endArrow=classic;endFill=1;endSize=5;",
             bx0 + 132, by0 + 20, bx0 + 132, by0 + c_y0 - 2)
    add_vertex("lbl_fq_irr", "<i>fq</i><sub>irr</sub>(<i>q</i><sub>irr</sub>)",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#0284C7;fontFamily=Helvetica;",
               bx0 + 96, by0 + 4, 72, 20)

    add_edge("arr_fr_1", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.9;endArrow=classic;endFill=1;endSize=5;",
             bx0 + 194, by0 + 20, bx0 + 194, by0 + c_y0 - 2)
    add_vertex("lbl_fr_1", "<i>fr</i><sub>1</sub>(<i>r</i>)",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
               bx0 + 168, by0 + 4, 52, 20)

    # Direct Rainfall Infiltration
    add_vertex("lbl_dir_rain_hdr", "<b>Direct</b><br>rainfall<br><i>fr</i><sub>i</sub>(<i>r</i>)",
               "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=10.5;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
               bx0 + 22, by0 + 52, 60, 46)
    add_edge("pointer_dir_rain", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.0;dashed=1;dashPattern=2.5 2.5;endArrow=classic;endFill=1;endSize=4;curved=1;",
             bx0 + 52, by0 + 94, bx0 + 64, by0 + 133, waypoints=[(bx0 + 54, by0 + 116)])

    # Infiltration arrows fr_2 ... fr_6
    for i in range(2, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        if i == 3:
            y_start, y_end, y_lbl = ym - 12.0, ym - 4.0, ym - 10.0
        elif i == 4:
            y_start, y_end, y_lbl = ym, ym + 8.0, ym + 4.0
        else:
            y_start, y_end, y_lbl = ym - 6.0, ym + 2.0, ym - 4.0

        add_edge(f"arr_fr_{i}", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.5;endArrow=classic;endFill=1;endSize=4.5;",
                 bx0 + 68, by0 + y_start, bx0 + 102, by0 + y_end)
        add_vertex(f"lbl_fr_{i}", f"<i>fr</i><sub>{i}</sub>",
                   "text;html=1;strokeColor=none;fillColor=none;align=right;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
                   bx0 + 36, by0 + y_lbl - 8, 28, 16)

    # Internal Vertical ODE Fluxes: Gravity drainage & Matric exchange
    for i in range(1, 6):
        yi = c_y0 + i * c_dz
        # Gravity drainage arrow
        add_edge(f"arr_f_{i}", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=5;",
                 bx0 + 132, by0 + yi - 8, bx0 + 132, by0 + yi + 8)
        add_vertex(f"lbl_f_{i}", f"<i>f</i><sub>{i}</sub>",
                   "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=11.2;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
                   bx0 + 138, by0 + yi - 10, 24, 16)

        # Matric exchange arrow (bidirectional purple dashed)
        add_edge(f"arr_mat_{i}", "", "edgeStyle=none;html=1;strokeColor=#6D28D9;strokeWidth=1.6;dashed=1;dashPattern=3 2;startArrow=classic;endArrow=classic;startFill=1;endFill=1;startSize=4.5;endSize=4.5;",
                 bx0 + 194, by0 + yi - 8, bx0 + 194, by0 + yi + 8)
        add_vertex(f"lbl_mat_{i}", f"<i>f</i><sub>{i},{i+1}</sub>",
                   "text;html=1;strokeColor=none;fillColor=none;align=right;verticalAlign=middle;fontSize=11.2;fontStyle=1;fontColor=#6D28D9;fontFamily=Helvetica;",
                   bx0 + 154, by0 + yi - 10, 36, 16)

    # Deep percolation arrow
    y_bot = c_y0 + 6 * c_dz
    add_edge("arr_deep_perc", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=2.0;endArrow=classic;endFill=1;endSize=5.5;",
             bx0 + 144, by0 + y_bot, bx0 + 144, by0 + y_bot + 28)
    add_vertex("lbl_deep_perc", "<i>f</i><sub>6</sub>(<i>m</i><sub>6</sub>) <font color=\"#475569\" style=\"font-size:10.2px; font-weight:normal;\">(deep percolation)</font>",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#1D4ED8;fontFamily=Helvetica;",
               bx0 + 154, by0 + y_bot + 8, 170, 20)

    # Lateral ditch drainage arrows fl_1 ... fl_5
    for i in range(1, 6):
        ym = c_y0 + (i - 0.5) * c_dz
        y_fl = ym - 12.0
        add_edge(f"arr_fl_{i}", "", "edgeStyle=none;html=1;strokeColor=#0D9488;strokeWidth=1.6;endArrow=classic;endFill=1;endSize=4.5;",
                 bx0 + c_x1, by0 + y_fl, bx0 + 254, by0 + y_fl)
        add_vertex(f"lbl_fl_{i}", f"<i>fl</i><sub>{i}</sub>",
                   "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=11.8;fontStyle=1;fontColor=#0D9488;fontFamily=Helvetica;",
                   bx0 + 258, by0 + y_fl - 8, 28, 16)

    # fl_6 = 0 label
    y_fl6 = c_y0 + 5.5 * c_dz - 12.0
    add_vertex("lbl_fl_6", "<i>fl</i><sub>6</sub> = 0",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.6;fontColor=#475569;fontFamily=Helvetica;",
               bx0 + 232, by0 + y_fl6 - 8, 48, 16)

    # Evapotranspirative sink header & arrows S_1 ... S_6
    add_vertex("lbl_et_sink_hdr", "Evapotranspirative sink<br><i>S</i><sub>i</sub>(<i>m</i><sub>i</sub>, <i>ET</i><sub>0</sub>) [<i>pc</i><sub>i</sub>]",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.6;fontStyle=1;fontColor=#D97706;fontFamily=Helvetica;",
               bx0 + 248, by0 + 4, 160, 32)

    for i in range(1, 7):
        ym = c_y0 + (i - 0.5) * c_dz
        y_et = ym + 12.0
        add_edge(f"arr_S_{i}", "", "edgeStyle=none;html=1;strokeColor=#D97706;strokeWidth=1.7;endArrow=classic;endFill=1;endSize=4.5;",
                 bx0 + c_x1, by0 + y_et, bx0 + 298, by0 + y_et)
        add_vertex(f"lbl_S_{i}", f"<i>S</i><sub>{i}</sub>",
                   "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=11.2;fontStyle=1;fontColor=#D97706;fontFamily=Helvetica;",
                   bx0 + 232, by0 + y_et - 8, 24, 16)

    # 4 Physical Regimes (Curly Brackets & Labels)
    regimes = [
        (c_y0 + 10,  c_y0 + 52,  "<b>Surface soil</b><br>evaporation"),
        (c_y0 + 72,  c_y0 + 176, "<b>Shallow root-zone uptake</b><br><font color=\"#475569\" style=\"font-size:10px;\">(peaks at 25 cm)</font>"),
        (c_y0 + 196, c_y0 + 300, "<b>Restricted clay-layer</b><br>depletion"),
        (c_y0 + 322, c_y0 + 364, "<b>Water-table &amp; ditch</b><br>capillary drawdown"),
    ]
    for r_idx, (y0_r, y1_r, r_text) in enumerate(regimes, start=1):
        add_vertex(f"bracket_regime_{r_idx}", "",
                   "shape=curlyBracket;rounded=1;flipH=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#D97706;strokeWidth=1.3;",
                   bx0 + 304, by0 + y0_r, 8, y1_r - y0_r)
        yc = 0.5 * (y0_r + y1_r)
        add_vertex(f"lbl_regime_{r_idx}", r_text,
                   "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#0F172A;fontFamily=Helvetica;",
                   bx0 + 318, by0 + yc - 14, 160, 28)

    # =========================================================================
    # 7. PANEL b: HORIZONTAL LEGEND BAR (Discrete cells)
    # =========================================================================
    leg_x, leg_y = bx0, 514.0
    add_vertex("legend_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=1;arcSize=6;",
               leg_x, leg_y, 510, 42)

    # Col 1: fq_irr & fr_i(r)
    add_edge("leg_arr_fq", "", "edgeStyle=none;html=1;strokeColor=#0284C7;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=4.5;",
             leg_x + 12, leg_y + 13, leg_x + 30, leg_y + 13)
    add_vertex("leg_lbl_fq", "<i>fq</i><sub>irr</sub>&nbsp;Irrigation",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 34, leg_y + 4, 110, 18)

    add_edge("leg_arr_fr", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.6;dashed=1;dashPattern=3 2;endArrow=classic;endFill=1;endSize=4.5;",
             leg_x + 12, leg_y + 29, leg_x + 30, leg_y + 29)
    add_vertex("leg_lbl_fr", "<i>fr</i><sub>i</sub>(<i>r</i>)&nbsp;Direct rainfall",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 34, leg_y + 20, 130, 18)

    # Col 2: f_i & f_i,i+1
    add_edge("leg_arr_f", "", "edgeStyle=none;html=1;strokeColor=#1D4ED8;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=4.5;",
             leg_x + 182, leg_y + 13, leg_x + 200, leg_y + 13)
    add_vertex("leg_lbl_f", "<i>f</i><sub>i</sub> ≥ 0&nbsp;Gravity drainage",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 204, leg_y + 4, 140, 18)

    add_edge("leg_arr_mat", "", "edgeStyle=none;html=1;strokeColor=#6D28D9;strokeWidth=1.6;dashed=1;dashPattern=2.5 1.5;startArrow=classic;endArrow=classic;startFill=1;endFill=1;startSize=4.5;endSize=4.5;",
             leg_x + 182, leg_y + 29, leg_x + 200, leg_y + 29)
    add_vertex("leg_lbl_mat", "<i>f</i><sub>i,i+1</sub> ≷ 0&nbsp;Matric exchange",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 204, leg_y + 20, 150, 18)

    # Col 3: fl_i & S_i
    add_edge("leg_arr_fl", "", "edgeStyle=none;html=1;strokeColor=#0D9488;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=4.5;",
             leg_x + 362, leg_y + 13, leg_x + 380, leg_y + 13)
    add_vertex("leg_lbl_fl", "<i>fl</i><sub>i</sub> ≥ 0&nbsp;Ditch outflow",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 384, leg_y + 4, 120, 18)

    add_edge("leg_arr_S", "", "edgeStyle=none;html=1;strokeColor=#D97706;strokeWidth=1.8;endArrow=classic;endFill=1;endSize=4.5;",
             leg_x + 362, leg_y + 29, leg_x + 380, leg_y + 29)
    add_vertex("leg_lbl_S", "<i>S</i><sub>i</sub> ≡ <i>pc</i><sub>i</sub>&nbsp;ET sink",
               "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=10.8;fontColor=#334155;fontFamily=Helvetica;",
               leg_x + 384, leg_y + 20, 110, 18)

    # Combine into full mxGraph XML
    drawio_xml = f"""<mxfile host="app.diagrams.net" modified="2026-11-12T19:50:00.000Z" agent="Antigravity" version="24.0.0" type="device">
  <diagram id="bed_ditch_schematic" name="Bed Ditch Schematic">
    <mxGraphModel dx="1280" dy="580" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1280" pageHeight="580" math="1" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{chr(10).join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    return drawio_xml

def main():
    import subprocess

    # 1. Pure Vector SVG (for Inkscape & direct display)
    svg_content = create_pure_svg()
    svg_path = os.path.join(OUT_DIR, "bed_ditch_schematic.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Wrote pure vector SVG: {svg_path}")

    # 2. Element-by-Element Draw.io XML
    drawio_content = create_element_by_element_drawio()
    drawio_path = os.path.join(OUT_DIR, "bed_ditch_schematic.drawio")
    with open(drawio_path, "w", encoding="utf-8") as f:
        f.write(drawio_content)
    print(f"Wrote element-by-element Draw.io XML: {drawio_path}")

    # 3. Hybrid Draw.io SVG (can be viewed as SVG AND edited in Draw.io)
    root = ET.fromstring(svg_content)
    root.attrib['content'] = drawio_content
    drawio_svg_path = os.path.join(OUT_DIR, "bed_ditch_schematic.drawio.svg")
    with open(drawio_svg_path, "w", encoding="utf-8") as f:
        f.write(ET.tostring(root, encoding='unicode'))
    print(f"Wrote hybrid Draw.io SVG: {drawio_svg_path}")

    # 4. Export vector PDF via Inkscape
    pdf_vec_path = os.path.join(OUT_DIR, "bed_ditch_schematic_vector.pdf")
    subprocess.run(["inkscape", f"--export-filename={pdf_vec_path}", svg_path], check=True)
    print(f"Wrote vector PDF: {pdf_vec_path}")

    # Synchronize with paper directory if it exists
    paper_dir = "/home/smartlab/Documents/Research/BARD/PINN/PINN_CODE_PAPER/Revison_Second_Stage/Improved_Paper"
    if os.path.exists(paper_dir):
        import shutil
        target_pdf = os.path.join(paper_dir, "bed_ditch_schematic.pdf")
        shutil.copy2(pdf_vec_path, target_pdf)
        print(f"Copied vector PDF to paper directory: {target_pdf}")

if __name__ == "__main__":
    main()
