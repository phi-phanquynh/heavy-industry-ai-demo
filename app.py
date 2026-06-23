from __future__ import annotations

import json
import hashlib
from datetime import datetime
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

try:
    from yfiles_graphs_for_streamlit import (
        Edge,
        EdgeStyle,
        FontWeight,
        LabelStyle,
        Layout,
        Node,
        StreamlitGraphWidget,
        TextAlignment,
        TextWrapping,
    )
except ImportError:  # pragma: no cover - fallback keeps the app usable without the optional component.
    Edge = EdgeStyle = FontWeight = LabelStyle = Layout = Node = StreamlitGraphWidget = None
    TextAlignment = TextWrapping = None


APP_NAME = "AI Variance Analysis Cockpit"
COMPANY_NAME = "Nippon Advanced Heavy Industries"
DATA_DIR = Path(__file__).resolve().parent / "data"

COMPARISON_MAP = {
    "Budget vs Actual": ("Budget", "Actual"),
    "Previous Forecast vs Latest Forecast": ("Previous Forecast", "Latest Forecast"),
    "Prior Year Actual vs Actual": ("Prior Year Actual", "Actual"),
    "Mid-term Plan vs Latest Forecast": ("Mid-term Plan", "Latest Forecast"),
}

SCENARIO_JA = {
    "Actual": "実績",
    "Budget": "予算",
    "Previous Forecast": "前回見込",
    "Latest Forecast": "最新見込",
    "Prior Year Actual": "前年実績",
    "Mid-term Plan": "中期計画",
}

KPI_META = {
    "Revenue": {
        "ja": "売上",
        "key": "revenue_jpy_mn",
        "driver_col": "impact_revenue_jpy_mn",
        "unit": "amount",
    },
    "Operating Profit": {
        "ja": "営業利益",
        "key": "operating_profit_jpy_mn",
        "driver_col": "impact_op_jpy_mn",
        "unit": "amount",
    },
    "Operating Profit Margin": {
        "ja": "営業利益率",
        "key": "op_margin_pct",
        "driver_col": "impact_op_jpy_mn",
        "unit": "margin",
    },
    "Cash Flow": {
        "ja": "キャッシュフロー",
        "key": "cash_flow_jpy_mn",
        "driver_col": "impact_cash_flow_jpy_mn",
        "unit": "amount",
    },
}

RISK_COLORS = {
    "Critical": "#ff4b5c",
    "High": "#ffb000",
    "Medium": "#39c5bb",
    "Low": "#6f7f8f",
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #071016;
            --panel: #101b24;
            --panel-2: #0c1720;
            --line: rgba(148, 163, 184, 0.28);
            --cyan: #39c5bb;
            --blue: #60a5fa;
            --green: #76d275;
            --red: #ff647c;
            --amber: #ffb000;
            --text: #edf6f9;
            --muted: #9fb2c3;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(7,16,22,0.98), rgba(7,16,22,1) 42%),
                repeating-linear-gradient(90deg, rgba(57,197,187,0.06) 0 1px, transparent 1px 120px);
        }
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #09131b;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            letter-spacing: 0;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .cockpit-title {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 16px;
            padding: 10px 0 14px 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 16px;
        }
        .cockpit-title h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.08;
        }
        .cockpit-title .subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 6px;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
        }
        .pill {
            border: 1px solid rgba(57,197,187,0.42);
            background: rgba(57,197,187,0.08);
            color: #d9fffb;
            padding: 5px 9px;
            border-radius: 6px;
            font-size: 0.76rem;
            white-space: nowrap;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(16,27,36,0.98), rgba(9,19,27,0.98));
            border: 1px solid var(--line);
            border-left: 3px solid var(--cyan);
            border-radius: 8px;
            padding: 14px 14px 12px 14px;
            min-height: 122px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .metric-value {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.05;
            white-space: nowrap;
        }
        .metric-delta {
            margin-top: 8px;
            font-size: 0.88rem;
        }
        .delta-good { color: var(--green); }
        .delta-bad { color: var(--red); }
        .delta-neutral { color: var(--muted); }
        .section-label {
            color: #d7e7ee;
            font-size: 1.02rem;
            font-weight: 650;
            margin: 10px 0 8px 0;
        }
        .insight-box {
            background: rgba(16, 27, 36, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            color: #dceff5;
            line-height: 1.65;
        }
        .guide-panel {
            background: linear-gradient(180deg, rgba(16,27,36,0.92), rgba(9,19,27,0.92));
            border: 1px solid rgba(57,197,187,0.28);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 0 0 14px 0;
            color: #dceff5;
        }
        .guide-title {
            color: #d9fffb;
            font-weight: 700;
            font-size: 0.98rem;
            margin-bottom: 6px;
        }
        .guide-panel p {
            margin: 0 0 8px 0;
            color: #c9dce5;
            line-height: 1.55;
        }
        .guide-panel ul {
            margin: 6px 0 0 1.1rem;
            padding: 0;
        }
        .guide-panel li {
            margin: 2px 0;
            color: #dceff5;
        }
        .guide-takeaway {
            margin-top: 10px;
            padding: 9px 10px;
            border-left: 3px solid var(--amber);
            background: rgba(255, 176, 0, 0.08);
            color: #fff2c2;
            border-radius: 4px;
        }
        .story-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 14px 0;
        }
        .story-card {
            background: rgba(12,23,32,0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            min-height: 132px;
        }
        .story-card b {
            color: #d9fffb;
            display: block;
            margin-bottom: 6px;
        }
        .story-card span {
            color: #c9dce5;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .stApp:has(#demo-briefing-page) {
            background: #f5f7fb !important;
        }
        .stApp:has(#demo-briefing-page) .block-container,
        .stApp:has(#demo-briefing-page) .block-container h1,
        .stApp:has(#demo-briefing-page) .block-container h2,
        .stApp:has(#demo-briefing-page) .block-container h3,
        .stApp:has(#demo-briefing-page) .block-container p,
        .stApp:has(#demo-briefing-page) .block-container li {
            color: #102033 !important;
        }
        .stApp:has(#demo-briefing-page) .cockpit-title {
            border-bottom: 1px solid rgba(15, 23, 42, 0.16);
        }
        .stApp:has(#demo-briefing-page) .cockpit-title .subtitle {
            color: #475569 !important;
        }
        .stApp:has(#demo-briefing-page) .pill {
            background: #e0f7f4;
            border-color: rgba(15, 118, 110, 0.24);
            color: #0f766e;
        }
        .stApp:has(#demo-briefing-page) .section-label {
            color: #0f172a;
        }
        .stApp:has(#demo-briefing-page) .story-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .stApp:has(#demo-briefing-page) .story-card b {
            color: #0f766e;
        }
        .stApp:has(#demo-briefing-page) .story-card span {
            color: #334155;
        }
        .briefing-hero {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 8px;
            padding: 24px 26px;
            box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
            margin-bottom: 16px;
        }
        .briefing-hero h2 {
            margin: 0 0 10px 0;
            font-size: 1.6rem;
            color: #0f172a !important;
        }
        .briefing-hero p {
            margin: 0;
            color: #334155 !important;
            line-height: 1.7;
        }
        .briefing-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 18px 0;
        }
        .briefing-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .briefing-card b {
            color: #0f766e;
            display: block;
            margin-bottom: 8px;
            font-size: 0.98rem;
        }
        .briefing-card span {
            color: #334155;
            line-height: 1.65;
            font-size: 0.92rem;
        }
        .briefing-flow {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin: 12px 0 18px 0;
        }
        .briefing-step {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-top: 4px solid #39c5bb;
            border-radius: 8px;
            padding: 12px;
            min-height: 126px;
        }
        .briefing-step b {
            color: #0f172a;
            display: block;
            margin-bottom: 6px;
        }
        .briefing-step span {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.55;
        }
        .briefing-message {
            background: #ecfeff;
            border: 1px solid rgba(8, 145, 178, 0.28);
            border-left: 5px solid #0891b2;
            border-radius: 8px;
            padding: 14px 16px;
            color: #164e63 !important;
            line-height: 1.7;
            margin: 10px 0 16px 0;
        }
        .briefing-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin: 10px 0 18px 0;
        }
        .briefing-table th {
            text-align: left;
            background: #102033;
            color: #ffffff;
            padding: 11px 12px;
            font-weight: 650;
        }
        .briefing-table td {
            color: #334155;
            padding: 11px 12px;
            border-top: 1px solid rgba(15, 23, 42, 0.10);
            vertical-align: top;
            line-height: 1.55;
        }
        .presentation-progress {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin: 8px 0 12px 0;
            color: #475569;
        }
        .presentation-progress b {
            color: #0f172a;
            font-size: 0.95rem;
        }
        .presentation-progress span {
            font-size: 0.82rem;
            font-weight: 700;
            color: #0f766e;
            white-space: nowrap;
        }
        .presentation-dots {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .presentation-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: rgba(100, 116, 139, 0.30);
        }
        .presentation-dot.active {
            width: 22px;
            background: #0f766e;
        }
        .presentation-slide {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 8px;
            box-shadow: 0 16px 44px rgba(15, 23, 42, 0.09);
            min-height: 540px;
            padding: 30px 34px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 18px;
        }
        .presentation-eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .presentation-slide h2 {
            color: #0f172a !important;
            font-size: 2.05rem;
            line-height: 1.15;
            margin: 0 0 10px 0;
        }
        .presentation-lead {
            color: #334155;
            font-size: 1.06rem;
            line-height: 1.78;
            max-width: 980px;
        }
        .presentation-card-grid {
            display: grid;
            gap: 12px;
            margin-top: 16px;
        }
        .presentation-card-grid.is-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .presentation-card-grid.is-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .presentation-card-grid.is-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .presentation-card-grid.is-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
        .presentation-card,
        .presentation-metric-card {
            background: #f8fafc;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            padding: 15px;
            min-height: 132px;
        }
        .presentation-card b,
        .presentation-metric-card b {
            color: #0f766e;
            display: block;
            margin-bottom: 8px;
            font-size: 0.98rem;
        }
        .presentation-card span,
        .presentation-metric-card span {
            color: #334155;
            line-height: 1.62;
            font-size: 0.92rem;
        }
        .presentation-metric-card strong {
            display: block;
            color: #0f172a;
            font-size: 1.18rem;
            line-height: 1.24;
            margin-bottom: 7px;
        }
        .presentation-flow {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin-top: 18px;
        }
        .presentation-flow.is-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .presentation-flow.is-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .presentation-flow.is-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
        .presentation-step {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-top: 4px solid #0f766e;
            border-radius: 8px;
            padding: 13px;
            min-height: 152px;
        }
        .presentation-step b {
            color: #0f172a;
            display: block;
            margin-bottom: 7px;
        }
        .presentation-step span {
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .architecture-diagram {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin-top: 18px;
            align-items: stretch;
        }
        .architecture-node {
            position: relative;
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border: 1px solid rgba(15, 23, 42, 0.14);
            border-top: 4px solid #0f766e;
            border-radius: 8px;
            padding: 14px;
            min-height: 144px;
        }
        .architecture-node:not(:last-child)::after {
            content: "\\2192";
            position: absolute;
            right: -15px;
            top: 50%;
            transform: translateY(-50%);
            color: #0f766e;
            font-weight: 900;
            font-size: 1.1rem;
            z-index: 2;
        }
        .architecture-node b {
            color: #0f172a;
            display: block;
            margin-bottom: 7px;
            font-size: 0.94rem;
        }
        .architecture-node span {
            color: #475569;
            display: block;
            font-size: 0.86rem;
            line-height: 1.52;
        }
        .architecture-node strong {
            color: #0f766e;
            display: block;
            font-size: 0.76rem;
            letter-spacing: 0;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .presentation-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 16px;
        }
        .presentation-table th {
            text-align: left;
            background: #102033;
            color: #ffffff;
            padding: 10px 11px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .presentation-table td {
            color: #334155;
            padding: 10px 11px;
            border-top: 1px solid rgba(15, 23, 42, 0.10);
            line-height: 1.5;
            vertical-align: top;
            font-size: 0.9rem;
        }
        .presentation-note {
            background: #ecfeff;
            border: 1px solid rgba(8, 145, 178, 0.28);
            border-left: 5px solid #0891b2;
            border-radius: 8px;
            padding: 13px 15px;
            color: #164e63;
            line-height: 1.65;
            margin-top: 16px;
        }
        .presentation-section-divider {
            display: flex;
            align-items: center;
            gap: 12px;
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            margin: 24px 0 15px 0;
            text-transform: uppercase;
        }
        .presentation-section-divider::before,
        .presentation-section-divider::after {
            content: "";
            height: 1px;
            flex: 1;
            background: rgba(15, 118, 110, 0.24);
        }
        .presentation-footer {
            border-top: 1px solid rgba(15, 23, 42, 0.12);
            padding-top: 11px;
            color: #64748b;
            font-size: 0.86rem;
        }
        .presentation-graph-header {
            min-height: auto;
            margin-bottom: 14px;
        }
        .presentation-graph-notes {
            background: rgba(9, 20, 30, 0.94);
            border: 1px solid rgba(57,197,187,0.24);
            border-radius: 8px;
            margin-top: 16px;
            padding: 20px;
        }
        .architecture-fallback {
            background: rgba(9, 20, 30, 0.94);
            border: 1px solid rgba(57,197,187,0.24);
            border-radius: 8px;
            padding: 18px;
        }
        .architecture-object-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .architecture-object-card {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.92), rgba(9, 20, 30, 0.94));
            border: 1px solid rgba(57,197,187,0.25);
            border-radius: 8px;
            padding: 14px;
            min-height: 132px;
        }
        .architecture-object-card strong {
            color: #39c5bb;
            display: block;
            font-size: 0.78rem;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .architecture-object-card b {
            color: #f8fdff;
            display: block;
            font-size: 1rem;
            margin-bottom: 8px;
        }
        .architecture-object-card span {
            color: #c8dce6;
            display: block;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .architecture-edge-list {
            display: grid;
            gap: 8px;
        }
        .architecture-edge-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 12px;
            align-items: center;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(159,178,195,0.18);
            border-radius: 8px;
            padding: 9px 11px;
        }
        .architecture-edge-row b {
            color: #f8fdff;
            font-size: 0.92rem;
        }
        .architecture-edge-row span {
            color: #39c5bb;
            font-size: 0.84rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .stApp:has(#presentation-focus-mode) [data-testid="stSidebar"],
        .stApp:has(#presentation-focus-mode) [data-testid="stToolbar"],
        .stApp:has(#presentation-focus-mode) [data-testid="stDecoration"],
        .stApp:has(#presentation-focus-mode) footer {
            display: none !important;
        }
        .stApp:has(#presentation-focus-mode) [data-testid="stHeader"] {
            background: transparent !important;
            pointer-events: none;
        }
        .stApp:has(#presentation-focus-mode) [data-testid="stHeader"] button,
        .stApp:has(#presentation-focus-mode) [data-testid="stHeader"] [role="button"] {
            pointer-events: auto !important;
        }
        .stApp:has(#presentation-focus-mode) button[data-testid="stBaseButton-headerNoPadding"] {
            visibility: visible !important;
            opacity: 1 !important;
        }
        .stApp:has(#presentation-focus-mode) .block-container {
            max-width: 100vw;
            padding: 0.75rem 1.15rem 1rem 1.15rem;
        }
        .stApp:has(#presentation-focus-mode) .cockpit-title {
            display: none;
        }
        .stApp:has(#presentation-focus-mode) .presentation-progress {
            margin: 0.35rem 0 0.55rem 0;
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide {
            min-height: calc(100vh - 132px);
            padding: clamp(22px, 3.1vw, 48px);
            box-shadow: none;
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide h2 {
            font-size: 2.25rem;
        }
        .foundation-hero {
            background:
                linear-gradient(135deg, rgba(16, 32, 51, 0.96), rgba(19, 78, 74, 0.92)),
                linear-gradient(90deg, rgba(255,255,255,0.06) 0 1px, transparent 1px 90px);
            border: 1px solid rgba(15, 23, 42, 0.16);
            border-radius: 10px;
            padding: 28px 30px;
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.16);
            margin-bottom: 14px;
        }
        .foundation-eyebrow {
            color: #99f6e4;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0;
            margin-bottom: 9px;
        }
        .foundation-hero h2 {
            color: #ffffff !important;
            font-size: 2.25rem;
            line-height: 1.08;
            margin: 0 0 13px 0;
            max-width: 980px;
        }
        .foundation-hero p {
            color: #dbeafe !important;
            font-size: 1.02rem;
            line-height: 1.78;
            margin: 0;
            max-width: 1040px;
        }
        .foundation-summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 18px 0;
        }
        .foundation-summary-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 10px;
            padding: 16px;
            min-height: 178px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        }
        .foundation-summary-card.risk { border-top: 5px solid #dc2626; }
        .foundation-summary-card.warning { border-top: 5px solid #f59e0b; }
        .foundation-summary-card.build { border-top: 5px solid #0f766e; }
        .foundation-summary-card b {
            display: block;
            color: #0f172a;
            font-size: 1.02rem;
            margin-bottom: 8px;
        }
        .foundation-summary-card span {
            display: block;
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.65;
        }
        .foundation-split {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 14px;
            margin: 10px 0 18px 0;
        }
        .foundation-panel {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 10px;
            padding: 17px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
            min-height: 360px;
        }
        .foundation-panel.danger {
            background: linear-gradient(180deg, #fff7f7, #ffffff);
            border-top: 5px solid #dc2626;
        }
        .foundation-panel.trusted {
            background: linear-gradient(180deg, #f0fdfa, #ffffff);
            border-top: 5px solid #0f766e;
        }
        .foundation-panel h3 {
            margin: 0 0 6px 0;
            color: #0f172a !important;
            font-size: 1.12rem;
        }
        .foundation-panel .caption {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.55;
            margin-bottom: 12px;
        }
        .fragment-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin: 10px 0 14px 0;
        }
        .fragment-pill {
            border: 1px dashed rgba(220, 38, 38, 0.32);
            background: rgba(254, 226, 226, 0.72);
            color: #7f1d1d;
            border-radius: 8px;
            padding: 9px 10px;
            font-size: 0.86rem;
            min-height: 42px;
        }
        .foundation-ai-box {
            border-radius: 9px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            background: #ffffff;
            padding: 12px;
            color: #334155;
            line-height: 1.6;
            margin-top: 8px;
        }
        .foundation-ai-box strong {
            color: #dc2626;
        }
        .trusted-stack {
            display: grid;
            gap: 8px;
            margin: 8px 0 14px 0;
        }
        .trusted-step {
            display: grid;
            grid-template-columns: 34px minmax(0, 1fr);
            gap: 10px;
            align-items: center;
            border: 1px solid rgba(15, 118, 110, 0.20);
            background: #ffffff;
            border-radius: 9px;
            padding: 9px 10px;
        }
        .trusted-step .num {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: #0f766e;
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .trusted-step b {
            color: #0f172a;
            display: block;
            margin-bottom: 1px;
        }
        .trusted-step span {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.42;
        }
        .foundation-detail-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 18px 0;
        }
        .foundation-detail-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            min-height: 232px;
        }
        .foundation-detail-card b {
            color: #0f172a;
            display: block;
            margin-bottom: 8px;
        }
        .foundation-detail-card p {
            color: #475569 !important;
            font-size: 0.88rem;
            line-height: 1.62;
            margin: 0 0 10px 0;
        }
        .foundation-tag {
            display: inline-block;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            padding: 4px 8px;
            font-size: 0.75rem;
            margin: 0 4px 5px 0;
        }
        .lineage-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .lineage-step {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-left: 4px solid #0ea5e9;
            border-radius: 8px;
            padding: 10px;
            min-height: 92px;
        }
        .lineage-step b {
            color: #0f172a;
            display: block;
            margin-bottom: 5px;
            font-size: 0.88rem;
        }
        .lineage-step span {
            color: #475569;
            font-size: 0.8rem;
            line-height: 1.45;
        }
        .foundation-domain-grid,
        .foundation-gate-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 18px 0;
        }
        .foundation-domain-card,
        .foundation-gate-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 10px;
            padding: 13px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            min-height: 126px;
        }
        .foundation-domain-card b,
        .foundation-gate-card b {
            color: #0f172a;
            display: block;
            margin-bottom: 6px;
        }
        .foundation-domain-card span,
        .foundation-gate-card span {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.52;
        }
        .foundation-domain-card em {
            color: #0f766e;
            display: block;
            font-style: normal;
            font-size: 0.78rem;
            margin-bottom: 6px;
        }
        .foundation-close {
            background: #102033;
            border-radius: 10px;
            border: 1px solid rgba(15, 23, 42, 0.16);
            padding: 18px 20px;
            color: #e2e8f0;
            line-height: 1.7;
            margin: 12px 0 8px 0;
        }
        .foundation-close b {
            color: #99f6e4;
        }
        .stApp:has(#foundation-page) .foundation-hero h2,
        .stApp:has(#foundation-page) .foundation-hero p,
        .stApp:has(#foundation-page) .foundation-eyebrow {
            color: inherit !important;
        }
        .stApp:has(#foundation-page) .foundation-hero h2 {
            color: #ffffff !important;
        }
        .stApp:has(#foundation-page) .foundation-hero p {
            color: #dbeafe !important;
        }
        .stApp:has(#foundation-page) .foundation-eyebrow {
            color: #99f6e4 !important;
        }
        .foundation-thesis {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 10px;
            padding: 16px 18px;
            margin: 12px 0 16px 0;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        }
        .foundation-thesis-row {
            display: grid;
            grid-template-columns: 1.1fr 1fr 1fr;
            gap: 14px;
        }
        .foundation-thesis-item {
            border-left: 4px solid #0f766e;
            padding-left: 12px;
        }
        .foundation-thesis-item.danger {
            border-left-color: #dc2626;
        }
        .foundation-thesis-item.warning {
            border-left-color: #f59e0b;
        }
        .foundation-thesis-item b {
            display: block;
            color: #0f172a;
            margin-bottom: 5px;
        }
        .foundation-thesis-item span {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        .foundation-pullquote {
            background: #102033;
            color: #e2e8f0;
            border-radius: 10px;
            border-left: 5px solid #dc2626;
            padding: 15px 17px;
            line-height: 1.65;
            margin: 12px 0 18px 0;
        }
        .foundation-pullquote b {
            color: #fca5a5;
        }
        .foundation-tabs-copy {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 8px;
            color: #334155;
            line-height: 1.7;
        }
        .foundation-tabs-copy b {
            color: #0f172a;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .snapshot-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px 0;
        }
        .snapshot-cell {
            background: rgba(12,23,32,0.90);
            border: 1px solid var(--line);
            border-top: 3px solid var(--cyan);
            border-radius: 8px;
            padding: 12px 13px;
            min-height: 112px;
        }
        .snapshot-cell b {
            display: block;
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .snapshot-cell strong {
            display: block;
            color: var(--text);
            font-size: 1.08rem;
            line-height: 1.24;
            margin-bottom: 5px;
        }
        .snapshot-cell span {
            color: #c9dce5;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .stApp:has(#demo-briefing-page) .snapshot-cell {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-top: 3px solid #0f766e;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .stApp:has(#demo-briefing-page) .snapshot-cell b {
            color: #64748b;
        }
        .stApp:has(#demo-briefing-page) .snapshot-cell strong {
            color: #0f172a;
        }
        .stApp:has(#demo-briefing-page) .snapshot-cell span {
            color: #475569;
        }
        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 14px;
        }
        .status-cell {
            background: rgba(12,23,32,0.88);
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 9px 10px;
            color: #d7e7ee;
            min-height: 58px;
            line-height: 1.35;
        }
        .status-cell b {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 2px;
        }
        .sidebar-meta {
            border-top: 1px solid rgba(148, 163, 184, 0.24);
            margin-top: 12px;
            padding-top: 10px;
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.5;
        }
        .sidebar-meta b {
            color: #d7e7ee;
        }
        .nav-current {
            background: rgba(57,197,187,0.10);
            border: 1px solid rgba(57,197,187,0.32);
            border-radius: 8px;
            padding: 10px 11px;
            margin: 12px 0 14px 0;
            color: #d7e7ee;
            line-height: 1.45;
        }
        .nav-current b {
            display: block;
            color: #39c5bb;
            font-size: 0.75rem;
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .nav-current span {
            display: block;
            color: #edf6f9;
            font-weight: 700;
        }
        div[data-testid="stMetric"] {
            background: rgba(16,27,36,0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
        }
        .stButton>button {
            border-radius: 6px;
            border: 1px solid rgba(57,197,187,0.38);
            background: rgba(57,197,187,0.12);
            color: #edf6f9;
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(180deg, rgba(57,197,187,0.48), rgba(57,197,187,0.22));
            border-color: rgba(57,197,187,0.95);
            box-shadow: inset 0 -2px 0 rgba(255,255,255,0.10), 0 0 0 1px rgba(57,197,187,0.22);
            color: #ffffff;
            font-weight: 750;
        }
        .stButton>button:hover {
            border-color: rgba(57,197,187,0.82);
            color: #ffffff;
            background: rgba(57,197,187,0.22);
        }
        .stApp:has(#demo-briefing-page),
        .stApp:has(#foundation-page) {
            background:
                linear-gradient(180deg, rgba(3, 8, 14, 0.98), rgba(6, 14, 22, 1) 58%),
                repeating-linear-gradient(90deg, rgba(57,197,187,0.08) 0 1px, transparent 1px 118px),
                repeating-linear-gradient(0deg, rgba(96,165,250,0.055) 0 1px, transparent 1px 96px) !important;
        }
        .stApp:has(#demo-briefing-page) .block-container,
        .stApp:has(#foundation-page) .block-container {
            max-width: 100vw;
            padding: 1.25rem 1.55rem 2.3rem 1.55rem;
        }
        .stApp:has(.presentation-slide) [data-testid="stToolbar"],
        .stApp:has(.presentation-slide) [data-testid="stDecoration"] {
            display: none !important;
        }
        .stApp:has(#proposal-component-page) [data-testid="stToolbar"],
        .stApp:has(#proposal-component-page) [data-testid="stDecoration"] {
            display: none !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stHeader"] {
            background: transparent !important;
            pointer-events: none;
            z-index: 999999;
        }
        .stApp:has(#proposal-component-page) [data-testid="stHeader"] {
            background: transparent !important;
            pointer-events: none;
            z-index: 999999;
        }
        .stApp:has(.presentation-slide) [data-testid="stHeader"] button,
        .stApp:has(.presentation-slide) [data-testid="stHeader"] [role="button"],
        .stApp:has(.presentation-slide) [data-testid="stSidebarCollapsedControl"] {
            pointer-events: auto !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebarCollapseButton"] {
            pointer-events: auto !important;
            position: relative !important;
            visibility: visible !important;
            z-index: 1000000;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] {
            max-width: 56px !important;
            min-width: 56px !important;
            transform: none !important;
            width: 56px !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
            background: #09131b !important;
            border-right: 1px solid rgba(57,197,187,0.22);
            min-width: 56px !important;
            overflow: hidden !important;
            width: 56px !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] > :not([data-testid="stSidebarHeader"]) {
            display: none !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] {
            display: none !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] {
            width: 56px !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {
            transform: translateX(-30px) !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
            transform: rotate(180deg);
        }
        .stApp:has(.presentation-slide) button[data-testid="stBaseButton-headerNoPadding"] {
            position: relative !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 1000001;
        }
        .stApp:has(.presentation-slide) .block-container {
            max-width: 100vw;
            padding-top: 0.65rem !important;
            padding-bottom: 2.25rem !important;
        }
        .proposal-turn-controls {
            align-items: center;
            bottom: 18px;
            display: flex;
            gap: 10px;
            left: 50%;
            position: fixed;
            transform: translateX(-50%);
            z-index: 999990;
        }
        .proposal-turn {
            align-items: center;
            background: rgba(8,20,31,0.92);
            border: 1px solid rgba(57,197,187,0.32);
            border-radius: 999px;
            box-shadow: 0 14px 30px rgba(0,0,0,0.28);
            color: #edf6f9 !important;
            display: inline-flex;
            font-size: 0.86rem;
            font-weight: 820;
            gap: 8px;
            line-height: 1;
            min-height: 42px;
            min-width: 104px;
            justify-content: center;
            padding: 0 14px;
            text-decoration: none !important;
        }
        .proposal-turn:hover {
            background: rgba(57,197,187,0.18);
            border-color: rgba(57,197,187,0.70);
            color: #ffffff !important;
        }
        .proposal-turn.is-disabled {
            color: rgba(159,178,195,0.40) !important;
            pointer-events: none;
            border-color: rgba(159,178,195,0.16);
        }
        .proposal-turn-index {
            color: #9fb2c3;
            font-size: 0.78rem;
            font-weight: 780;
            min-width: 54px;
            text-align: center;
        }
        .st-key-proposal-turn-controls {
            position: fixed;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            width: 50px;
            z-index: 999990;
        }
        .st-key-proposal-turn-controls [data-testid="stVerticalBlock"] {
            gap: 8px;
        }
        .st-key-proposal-turn-controls button,
        .st-key-proposal-turn-controls [data-testid="stLinkButton"] a,
        .st-key-proposal-turn-controls a {
            align-items: center;
            background: rgba(8,20,31,0.92) !important;
            border: 1px solid rgba(57,197,187,0.34) !important;
            border-radius: 999px !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.30);
            color: #edf6f9 !important;
            display: inline-flex;
            font-size: 1.1rem !important;
            font-weight: 900 !important;
            height: 42px;
            justify-content: center;
            min-height: 42px;
            padding: 0 !important;
            text-decoration: none !important;
            width: 42px;
        }
        .st-key-proposal-turn-controls button:hover,
        .st-key-proposal-turn-controls [data-testid="stLinkButton"] a:hover,
        .st-key-proposal-turn-controls a:hover {
            background: rgba(57,197,187,0.20) !important;
            border-color: rgba(57,197,187,0.72) !important;
            color: #ffffff !important;
        }
        .st-key-proposal-turn-controls button:disabled,
        .st-key-proposal-turn-controls [data-testid="stLinkButton"] a[aria-disabled="true"],
        .st-key-proposal-turn-controls a[aria-disabled="true"] {
            color: rgba(159,178,195,0.36) !important;
            border-color: rgba(159,178,195,0.16) !important;
            pointer-events: none;
        }
        .st-key-proposal-turn-controls .proposal-button-index {
            color: #9fb2c3;
            font-size: 0.72rem;
            font-weight: 820;
            line-height: 1;
            text-align: center;
        }
        .stApp:has(#demo-briefing-page) .block-container,
        .stApp:has(#demo-briefing-page) .block-container h1,
        .stApp:has(#demo-briefing-page) .block-container h2,
        .stApp:has(#demo-briefing-page) .block-container h3,
        .stApp:has(#demo-briefing-page) .block-container p,
        .stApp:has(#demo-briefing-page) .block-container li,
        .stApp:has(#foundation-page) .block-container,
        .stApp:has(#foundation-page) .block-container h1,
        .stApp:has(#foundation-page) .block-container h2,
        .stApp:has(#foundation-page) .block-container h3,
        .stApp:has(#foundation-page) .block-container p,
        .stApp:has(#foundation-page) .block-container li {
            color: #edf6f9 !important;
        }
        .stApp:has(#demo-briefing-page) .cockpit-title,
        .stApp:has(#foundation-page) .cockpit-title {
            border-bottom: 1px solid rgba(57,197,187,0.30);
            padding: 6px 0 10px 0;
            margin-bottom: 10px;
        }
        .stApp:has(.presentation-slide) .cockpit-title {
            display: none;
        }
        .stApp:has(#demo-briefing-page) .cockpit-title .subtitle,
        .stApp:has(#foundation-page) .cockpit-title .subtitle {
            color: #9fb2c3 !important;
        }
        .stApp:has(#demo-briefing-page) .pill,
        .stApp:has(#foundation-page) .pill {
            background: rgba(57,197,187,0.10);
            border-color: rgba(57,197,187,0.36);
            color: #d9fffb;
        }
        .presentation-progress {
            margin: 0.3rem 0 0.5rem 0;
            color: #9fb2c3;
        }
        .presentation-progress b {
            color: #edf6f9;
            font-size: 0.94rem;
        }
        .presentation-progress span {
            color: #39c5bb;
            font-size: 0.78rem;
        }
        .presentation-dot {
            background: rgba(159,178,195,0.32);
        }
        .presentation-dot.active {
            background: #39c5bb;
            box-shadow: 0 0 12px rgba(57,197,187,0.45);
        }
        .presentation-slide {
            position: relative;
            overflow: visible;
            display: block;
            background:
                linear-gradient(135deg, rgba(11, 23, 34, 0.98), rgba(6, 15, 24, 0.99)),
                linear-gradient(90deg, rgba(57,197,187,0.08), transparent 42%);
            border: 1px solid rgba(57,197,187,0.30);
            border-radius: 8px;
            box-shadow: 0 22px 70px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.06);
            min-height: 680px;
            padding: clamp(28px, 2.6vw, 44px);
        }
        .presentation-slide::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(57,197,187,0.08) 0 1px, transparent 1px 84px),
                linear-gradient(0deg, rgba(96,165,250,0.07) 0 1px, transparent 1px 72px);
            opacity: 0.36;
            pointer-events: none;
        }
        .presentation-slide > * {
            position: relative;
            z-index: 1;
        }
        .presentation-eyebrow {
            color: #39c5bb;
            font-size: 0.72rem;
            margin-bottom: 7px;
        }
        .presentation-slide h2 {
            color: #f8fdff !important;
            font-size: 2.2rem;
            line-height: 1.16;
            margin: 0 0 14px 0;
            max-width: 1240px;
        }
        .presentation-lead {
            color: #d6e7ee;
            font-size: 1.04rem;
            line-height: 1.72;
            max-width: 1180px;
        }
        .presentation-card-grid {
            gap: 14px;
            margin-top: 18px;
        }
        .presentation-card,
        .presentation-metric-card,
        .architecture-node {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.92), rgba(9, 20, 30, 0.94));
            border: 1px solid rgba(57,197,187,0.25);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .presentation-card,
        .presentation-metric-card {
            padding: 16px;
            min-height: 126px;
        }
        .presentation-card b,
        .presentation-metric-card b,
        .architecture-node strong {
            color: #39c5bb;
        }
        .presentation-card span,
        .presentation-metric-card span,
        .architecture-node span {
            color: #c8dce6;
            line-height: 1.58;
            font-size: 0.95rem;
        }
        .presentation-metric-card strong {
            color: #f8fdff;
            font-size: 1.16rem;
            margin-bottom: 7px;
        }
        .presentation-flow {
            gap: 14px;
            margin-top: 18px;
        }
        .presentation-step {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.92), rgba(9, 20, 30, 0.94));
            border: 1px solid rgba(57,197,187,0.25);
            border-top: 4px solid #39c5bb;
            padding: 14px;
            min-height: 144px;
        }
        .presentation-step b,
        .architecture-node b {
            color: #f8fdff;
        }
        .presentation-step span {
            color: #c8dce6;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .architecture-diagram {
            gap: 14px;
            margin-top: 18px;
        }
        .architecture-node {
            border-top: 4px solid #39c5bb;
            padding: 14px;
            min-height: 142px;
        }
        .architecture-node:not(:last-child)::after {
            color: #39c5bb;
        }
        .presentation-table {
            background: rgba(9, 20, 30, 0.92);
            border: 1px solid rgba(57,197,187,0.24);
            margin-top: 9px;
        }
        .presentation-table th {
            background: rgba(57,197,187,0.18);
            color: #d9fffb;
            padding: 6px 7px;
            font-size: 0.74rem;
        }
        .presentation-table td {
            color: #d6e7ee;
            padding: 6px 7px;
            border-top: 1px solid rgba(57,197,187,0.16);
            line-height: 1.25;
            font-size: 0.74rem;
        }
        .presentation-note {
            background: rgba(57,197,187,0.10);
            border: 1px solid rgba(57,197,187,0.30);
            border-left: 4px solid #39c5bb;
            color: #d9fffb;
            padding: 13px 15px;
            line-height: 1.62;
            margin-top: 16px;
        }
        .presentation-section-divider {
            color: #39c5bb;
            margin: 26px 0 16px 0;
        }
        .presentation-section-divider::before,
        .presentation-section-divider::after {
            background: linear-gradient(90deg, rgba(57,197,187,0.45), rgba(57,197,187,0.06));
        }
        .presentation-footer {
            border-top: 1px solid rgba(57,197,187,0.22);
            color: #9fb2c3;
            font-size: 0.86rem;
            padding-top: 12px;
            margin-top: 30px;
        }
        .presentation-slide.architecture-slide {
            min-height: auto;
            padding: clamp(22px, 2.2vw, 34px);
        }
        .architecture-slide .presentation-lead {
            max-width: none;
            font-size: 0.98rem;
            line-height: 1.62;
        }
        .architecture-slide .presentation-section-divider {
            margin: 18px 0 12px 0;
        }
        .architecture-lane-map {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 34px minmax(0, 1fr) 34px minmax(0, 1fr);
            gap: 10px;
            align-items: stretch;
            margin-top: 10px;
        }
        .architecture-lane {
            background: rgba(9, 20, 30, 0.82);
            border: 1px solid rgba(57,197,187,0.25);
            border-radius: 8px;
            padding: 12px;
            min-height: 330px;
        }
        .architecture-lane-title {
            border-bottom: 1px solid rgba(57,197,187,0.18);
            margin-bottom: 10px;
            padding-bottom: 8px;
        }
        .architecture-lane-title em {
            color: #39c5bb;
            display: block;
            font-size: 0.68rem;
            font-style: normal;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .architecture-lane-title b {
            color: #f8fdff;
            display: block;
            font-size: 0.95rem;
            margin-top: 3px;
        }
        .architecture-lane-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
        }
        .architecture-node-chip {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(7, 18, 28, 0.98));
            border: 1px solid rgba(159,178,195,0.18);
            border-left: 4px solid var(--accent, #39c5bb);
            border-radius: 8px;
            min-height: 74px;
            padding: 9px 10px;
        }
        .architecture-node-chip strong {
            color: #9ff7ef;
            display: block;
            font-size: 0.66rem;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .architecture-node-chip b {
            color: #f8fdff;
            display: block;
            font-size: 0.86rem;
            line-height: 1.25;
            margin-bottom: 5px;
        }
        .architecture-node-chip span {
            color: #c8dce6;
            display: -webkit-box;
            font-size: 0.75rem;
            line-height: 1.35;
            overflow: hidden;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
        }
        .architecture-lane-arrow {
            align-items: center;
            color: #39c5bb;
            display: flex;
            font-size: 1.6rem;
            font-weight: 900;
            justify-content: center;
        }
        .architecture-key-routes {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 12px;
        }
        .architecture-route-chip {
            background: rgba(57,197,187,0.08);
            border: 1px solid rgba(57,197,187,0.22);
            border-radius: 8px;
            color: #d6e7ee;
            font-size: 0.74rem;
            line-height: 1.35;
            padding: 8px 9px;
        }
        .architecture-route-chip b {
            color: #f8fdff;
            display: block;
            font-size: 0.76rem;
            margin-bottom: 3px;
        }
        .architecture-route-chip span {
            color: #39c5bb;
            font-weight: 800;
        }
        .briefing-hero,
        .briefing-card,
        .briefing-step,
        .foundation-thesis,
        .foundation-tabs-copy,
        .snapshot-cell,
        .stApp:has(#demo-briefing-page) .snapshot-cell {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(9, 20, 30, 0.95));
            border: 1px solid rgba(57,197,187,0.25);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .briefing-hero h2,
        .briefing-step b,
        .briefing-card b,
        .foundation-thesis-item b,
        .foundation-tabs-copy b,
        .stApp:has(#demo-briefing-page) .snapshot-cell strong {
            color: #f8fdff !important;
        }
        .briefing-hero p,
        .briefing-card span,
        .briefing-step span,
        .foundation-thesis-item span,
        .foundation-tabs-copy,
        .stApp:has(#demo-briefing-page) .snapshot-cell span {
            color: #c8dce6 !important;
        }
        .briefing-table {
            background: rgba(9,20,30,0.94);
            border: 1px solid rgba(57,197,187,0.24);
            box-shadow: none;
        }
        .briefing-table th {
            background: rgba(57,197,187,0.18);
            color: #d9fffb;
        }
        .briefing-table td {
            color: #d6e7ee;
            border-top: 1px solid rgba(57,197,187,0.16);
        }
        .stApp:has(#presentation-focus-mode) .block-container {
            padding: 0.75rem 1.15rem 2rem 1.15rem;
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide {
            min-height: calc(100vh - 96px);
            padding: clamp(28px, 3vw, 54px);
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide h2 {
            font-size: 2.25rem;
        }
        .foundation-flow-map {
            display: grid;
            grid-template-columns:
                minmax(220px, 1fr) 42px
                minmax(220px, 1fr) 42px
                minmax(220px, 1fr) 42px
                minmax(220px, 1fr);
            gap: 10px;
            align-items: stretch;
            margin-top: 14px;
        }
        .foundation-flow-block {
            position: relative;
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(9, 20, 30, 0.96));
            border: 1px solid rgba(57,197,187,0.25);
            border-radius: 8px;
            padding: 14px;
            min-height: 300px;
        }
        .foundation-flow-arrow {
            align-items: center;
            display: flex;
            justify-content: center;
            color: #39c5bb;
            font-size: 1.9rem;
            font-weight: 900;
            text-shadow: 0 0 18px rgba(57,197,187,0.55);
        }
        .foundation-flow-block b {
            color: #f8fdff;
            display: block;
            margin-bottom: 12px;
            font-size: 1.02rem;
        }
        .foundation-flow-block span {
            display: block;
            color: #c8dce6;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .foundation-flow-items {
            display: grid;
            gap: 8px;
        }
        .foundation-flow-item {
            border: 1px solid rgba(159,178,195,0.20);
            background: rgba(255,255,255,0.035);
            border-radius: 6px;
            color: #d6e7ee;
            font-size: 0.92rem;
            padding: 8px 9px;
        }
        .foundation-flow-block.hot {
            border-color: rgba(255,176,0,0.36);
        }
        .foundation-flow-block.ai {
            border-color: rgba(183,148,244,0.38);
        }
        .foundation-quality-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 14px;
        }
        .foundation-quality-card {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(9, 20, 30, 0.96));
            border: 1px solid rgba(57,197,187,0.25);
            border-top: 4px solid #39c5bb;
            border-radius: 8px;
            padding: 16px;
            min-height: 150px;
        }
        .foundation-quality-card b {
            color: #f8fdff;
            display: block;
            font-size: 1.02rem;
            margin-bottom: 8px;
        }
        .foundation-quality-card span {
            color: #c8dce6;
            display: block;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .foundation-quality-card em {
            color: #39c5bb;
            display: block;
            font-size: 0.78rem;
            font-style: normal;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .proposal-slide {
            min-height: calc(100vh - 92px);
            padding: clamp(26px, 2.55vw, 42px);
            background:
                linear-gradient(135deg, rgba(9, 18, 29, 0.99), rgba(5, 12, 20, 0.99) 58%, rgba(18, 26, 37, 0.99)),
                repeating-linear-gradient(90deg, rgba(57,197,187,0.07) 0 1px, transparent 1px 112px);
        }
        .proposal-slide::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 4px;
            background: linear-gradient(90deg, #39c5bb, #60a5fa, #ffb000, #ff647c);
        }
        .proposal-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 14px;
        }
        .proposal-kicker {
            color: #9fb2c3;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .proposal-kicker b {
            color: #39c5bb;
            margin-right: 8px;
        }
        .proposal-dots {
            display: flex;
            gap: 6px;
        }
        .proposal-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: rgba(159, 178, 195, 0.32);
        }
        .proposal-dot.active {
            width: 24px;
            background: #39c5bb;
            box-shadow: 0 0 18px rgba(57,197,187,0.55);
        }
        .proposal-slide h2 {
            max-width: 1180px;
            font-size: clamp(2.1rem, 3.05vw, 3.35rem);
            line-height: 1.08;
            margin-bottom: 11px;
        }
        .proposal-lead {
            color: #d6e7ee;
            font-size: clamp(1.02rem, 1.12vw, 1.2rem);
            line-height: 1.62;
            max-width: 1120px;
        }
        .proposal-grid-2 {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(360px, 0.88fr);
            gap: 22px;
            align-items: stretch;
            margin-top: 22px;
        }
        .proposal-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 20px;
        }
        .proposal-panel {
            background: linear-gradient(180deg, rgba(14,29,42,0.90), rgba(7,18,28,0.96));
            border: 1px solid rgba(159,178,195,0.20);
            border-radius: 8px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            padding: 18px;
        }
        .proposal-panel.accent-cyan { border-left: 4px solid #39c5bb; }
        .proposal-panel.accent-amber { border-left: 4px solid #ffb000; }
        .proposal-panel.accent-red { border-left: 4px solid #ff647c; }
        .proposal-panel.accent-blue { border-left: 4px solid #60a5fa; }
        .proposal-panel b {
            color: #f8fdff;
            display: block;
            font-size: 1.02rem;
            line-height: 1.35;
            margin-bottom: 7px;
        }
        .proposal-panel span,
        .proposal-panel li {
            color: #c8dce6;
            font-size: 0.95rem;
            line-height: 1.56;
        }
        .proposal-panel ul {
            margin: 8px 0 0 1.05rem;
            padding: 0;
        }
        .fragment-stage {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 22px;
        }
        .fragment-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(159,178,195,0.20);
            border-radius: 8px;
            min-height: 132px;
            padding: 15px;
            position: relative;
        }
        .fragment-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            border-radius: 8px 0 0 8px;
            background: var(--accent, #39c5bb);
        }
        .fragment-card em {
            color: var(--accent, #39c5bb);
            display: block;
            font-size: 0.72rem;
            font-style: normal;
            font-weight: 850;
            margin-bottom: 7px;
            text-transform: uppercase;
        }
        .fragment-card b {
            color: #f8fdff;
            display: block;
            font-size: 1rem;
            margin-bottom: 6px;
        }
        .fragment-card span {
            color: #c8dce6;
            font-size: 0.88rem;
            line-height: 1.48;
        }
        .problem-logic {
            display: grid;
            gap: 10px;
            margin-top: 16px;
        }
        .problem-map {
            align-items: stretch;
            display: grid;
            gap: 12px;
            grid-template-columns: minmax(250px, 0.88fr) 82px minmax(0, 1.32fr);
        }
        .fpna-expectation {
            background: linear-gradient(180deg, rgba(255,176,0,0.11), rgba(255,176,0,0.055));
            border: 1px solid rgba(255,176,0,0.30);
            border-left: 5px solid #ffb000;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 14px 15px;
        }
        .fpna-expectation em {
            color: #ffcf66;
            display: block;
            font-size: 0.72rem;
            font-style: normal;
            font-weight: 850;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .fpna-expectation b {
            color: #f8fdff;
            display: block;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .expectation-items {
            display: grid;
            grid-template-columns: repeat(1, minmax(0, 1fr));
            gap: 8px;
        }
        .expectation-items div {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,176,0,0.18);
            border-radius: 7px;
            color: #fff2c2;
            font-size: 0.84rem;
            font-weight: 760;
            line-height: 1.34;
            padding: 8px 10px;
        }
        .problem-connector {
            align-items: center;
            color: #9ff7ef;
            display: flex;
            flex-direction: column;
            font-size: 0.82rem;
            font-weight: 850;
            gap: 12px;
            justify-content: center;
            letter-spacing: 0;
            line-height: 1.35;
            text-align: center;
        }
        .problem-connector span {
            display: block;
            width: 76px;
        }
        .problem-connector::before,
        .problem-connector::after {
            background: rgba(57,197,187,0.28);
            content: "";
            flex: 1;
            min-height: 46px;
            width: 1px;
        }
        .fragment-stage.problem-blockers {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 0;
        }
        .fragment-stage.problem-blockers .fragment-card {
            min-height: 132px;
            padding: 11px 11px 11px 13px;
        }
        .fragment-stage.problem-blockers .fragment-card b {
            font-size: 0.92rem;
            line-height: 1.34;
        }
        .fragment-stage.problem-blockers .fragment-card span {
            font-size: 0.79rem;
            line-height: 1.42;
        }
        .executive-question {
            align-items: center;
            background: linear-gradient(90deg, rgba(57,197,187,0.13), rgba(96,165,250,0.10));
            border: 1px solid rgba(57,197,187,0.36);
            border-radius: 8px;
            color: #e5fbff;
            display: flex;
            font-size: 0.98rem;
            font-weight: 750;
            justify-content: center;
            line-height: 1.55;
            margin-top: 0;
            padding: 10px 14px;
            text-align: center;
        }
        .flow-ribbon {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 30px;
        }
        .flow-node {
            background: linear-gradient(180deg, rgba(14,29,42,0.95), rgba(8,20,31,0.98));
            border: 1px solid rgba(57,197,187,0.26);
            border-radius: 8px;
            min-height: 158px;
            padding: 16px;
            position: relative;
        }
        .flow-node:not(:last-child)::after {
            content: "\\2192";
            color: #39c5bb;
            font-size: 1.45rem;
            font-weight: 900;
            position: absolute;
            right: -19px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 3;
        }
        .flow-node em {
            color: #39c5bb;
            display: block;
            font-size: 0.72rem;
            font-style: normal;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .flow-node b {
            color: #f8fdff;
            display: block;
            font-size: 1.04rem;
            margin-bottom: 8px;
        }
        .flow-node span {
            color: #c8dce6;
            font-size: 0.9rem;
            line-height: 1.48;
        }
        .foundation-quadrant {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-top: 28px;
        }
        .foundation-core {
            align-items: center;
            background: linear-gradient(180deg, rgba(57,197,187,0.17), rgba(57,197,187,0.09));
            border: 1px solid rgba(57,197,187,0.42);
            border-radius: 8px;
            color: #d9fffb;
            display: flex;
            font-size: 1.28rem;
            font-weight: 850;
            justify-content: center;
            line-height: 1.45;
            min-height: 88px;
            margin: 22px auto 0 auto;
            max-width: 860px;
            padding: 0 28px;
            text-align: center;
        }
        .tech-architecture {
            display: grid;
            grid-template-columns: 1.1fr 0.28fr 1.1fr 0.28fr 1.1fr 0.28fr 1.05fr 0.28fr 1.05fr;
            gap: 8px;
            align-items: stretch;
            margin-top: 18px;
        }
        .tech-layer {
            background: linear-gradient(180deg, rgba(14,29,42,0.94), rgba(8,20,31,0.98));
            border: 1px solid rgba(159,178,195,0.22);
            border-top: 4px solid var(--accent, #39c5bb);
            border-radius: 8px;
            min-height: 254px;
            padding: 10px;
        }
        .tech-layer em {
            color: var(--accent, #39c5bb);
            display: block;
            font-size: 0.68rem;
            font-style: normal;
            font-weight: 850;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .tech-layer b {
            color: #f8fdff;
            display: block;
            font-size: 1rem;
            margin-bottom: 8px;
        }
        .tech-pill {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(159,178,195,0.18);
            border-radius: 6px;
            color: #d6e7ee;
            font-size: 0.72rem;
            line-height: 1.18;
            margin-top: 5px;
            padding: 5px 6px;
        }
        .tech-arrow {
            align-items: center;
            color: #39c5bb;
            display: flex;
            font-size: 1.35rem;
            font-weight: 900;
            justify-content: center;
        }
        .tech-footnote {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }
        .tech-footnote div {
            background: rgba(57,197,187,0.07);
            border: 1px solid rgba(57,197,187,0.20);
            border-radius: 8px;
            color: #c8dce6;
            font-size: 0.71rem;
            line-height: 1.24;
            padding: 6px 7px;
        }
        .option-card,
        .roadmap-phase,
        .assessment-item {
            background: linear-gradient(180deg, rgba(14,29,42,0.95), rgba(8,20,31,0.98));
            border: 1px solid rgba(159,178,195,0.22);
            border-radius: 8px;
            position: relative;
        }
        .option-card {
            min-height: 292px;
            padding: 18px;
        }
        .option-card::before,
        .roadmap-phase::after {
            content: "";
            position: absolute;
            left: 18px;
            right: 18px;
            top: 0;
            height: 4px;
            background: var(--accent, #39c5bb);
            border-radius: 0 0 6px 6px;
        }
        .option-card em,
        .roadmap-phase em,
        .assessment-item em {
            color: var(--accent, #39c5bb);
            display: block;
            font-size: 0.72rem;
            font-style: normal;
            font-weight: 850;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .option-card b,
        .roadmap-phase b,
        .assessment-item b {
            color: #f8fdff;
            display: block;
            font-size: 1.08rem;
            margin-bottom: 10px;
        }
        .option-card span,
        .roadmap-phase span,
        .assessment-item span {
            color: #c8dce6;
            display: block;
            font-size: 0.91rem;
            line-height: 1.5;
            margin-bottom: 12px;
        }
        .option-meta {
            border-top: 1px solid rgba(159,178,195,0.18);
            color: #d6e7ee;
            display: grid;
            gap: 7px;
            font-size: 0.85rem;
            padding-top: 12px;
        }
        .option-meta strong {
            color: #9ff7ef;
        }
        .roadmap {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
            margin-top: 28px;
        }
        .roadmap-phase {
            min-height: 318px;
            padding: 18px;
        }
        .roadmap-gate {
            background: rgba(255,176,0,0.09);
            border: 1px solid rgba(255,176,0,0.26);
            border-radius: 7px;
            color: #fff2c2;
            font-size: 0.87rem;
            line-height: 1.42;
            padding: 10px;
        }
        .assessment-canvas {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 25px;
        }
        .assessment-item {
            min-height: 145px;
            padding: 16px;
        }
        .proposal-close {
            background: rgba(57,197,187,0.10);
            border: 1px solid rgba(57,197,187,0.30);
            border-left: 4px solid #39c5bb;
            border-radius: 8px;
            color: #d9fffb;
            font-size: 1.05rem;
            font-weight: 760;
            line-height: 1.52;
            margin-top: 15px;
            padding: 13px 16px;
        }
        /* Projection readability pass: larger type with proportional spacing. */
        .stApp:has(.presentation-slide) [data-testid="stSidebar"] {
            max-width: 56px !important;
            min-width: 56px !important;
            width: 56px !important;
        }
        .stApp:has(#proposal-component-page) [data-testid="stSidebar"] {
            max-width: 56px !important;
            min-width: 56px !important;
            width: 56px !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebarContent"] {
            background: #09131b !important;
            border-right: 1px solid rgba(57,197,187,0.22);
            min-width: 56px !important;
            overflow: hidden !important;
            width: 56px !important;
        }
        .stApp:has(#proposal-component-page) [data-testid="stSidebarContent"] {
            background: #09131b !important;
            border-right: 1px solid rgba(57,197,187,0.22);
            min-width: 56px !important;
            overflow: hidden !important;
            width: 56px !important;
        }
        .stApp:has(.presentation-slide) [data-testid="stSidebarContent"] > :not([data-testid="stSidebarHeader"]),
        .stApp:has(.presentation-slide) [data-testid="stSidebarUserContent"],
        .stApp:has(#proposal-component-page) [data-testid="stSidebarContent"] > :not([data-testid="stSidebarHeader"]),
        .stApp:has(#proposal-component-page) [data-testid="stSidebarUserContent"] {
            display: none !important;
        }
        .stApp:has(.presentation-slide) .block-container,
        .stApp:has(#proposal-component-page) .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .presentation-progress {
            margin: 0.4rem 0 0.75rem 0;
        }
        .presentation-progress b {
            font-size: 1.08rem;
        }
        .presentation-progress span {
            font-size: 0.9rem;
        }
        .presentation-slide {
            min-height: calc(100vh - 44px);
            padding: clamp(17px, 1.5vw, 31px);
        }
        .presentation-eyebrow {
            font-size: 0.86rem;
            margin-bottom: 9px;
        }
        .presentation-slide h2 {
            font-size: clamp(2.5rem, 3.05vw, 3.25rem);
            line-height: 1.12;
            margin-bottom: 17px;
        }
        .presentation-lead {
            font-size: 1.18rem;
            line-height: 1.62;
            max-width: 1240px;
        }
        .presentation-card-grid,
        .presentation-flow {
            gap: 16px;
            margin-top: 22px;
        }
        .presentation-card,
        .presentation-metric-card {
            min-height: 150px;
            padding: 18px;
        }
        .presentation-card b,
        .presentation-metric-card b,
        .presentation-step b,
        .architecture-node b {
            font-size: 1.1rem;
            line-height: 1.34;
        }
        .presentation-card span,
        .presentation-metric-card span,
        .presentation-step span,
        .architecture-node span {
            font-size: 1.06rem;
            line-height: 1.52;
        }
        .presentation-metric-card strong {
            font-size: 1.32rem;
        }
        .presentation-step {
            min-height: 166px;
            padding: 17px;
        }
        .architecture-node {
            min-height: 166px;
            padding: 17px;
        }
        .architecture-node strong {
            font-size: 0.9rem;
        }
        .presentation-table th,
        .presentation-table td {
            font-size: 0.9rem;
            line-height: 1.36;
            padding: 8px 9px;
        }
        .presentation-note {
            font-size: 1.08rem;
            line-height: 1.56;
            padding: 15px 17px;
        }
        .presentation-footer {
            font-size: 0.98rem;
        }
        .presentation-section-divider {
            font-size: 0.9rem;
        }
        .presentation-slide.architecture-slide {
            padding: clamp(28px, 2.5vw, 42px);
        }
        .architecture-slide .presentation-lead {
            font-size: 1.08rem;
            line-height: 1.56;
        }
        .architecture-lane-map {
            grid-template-columns: minmax(0, 1fr) 28px minmax(0, 1fr) 28px minmax(0, 1fr);
            gap: 9px;
        }
        .architecture-lane {
            min-height: 350px;
            padding: 14px;
        }
        .architecture-lane-title em {
            font-size: 0.8rem;
        }
        .architecture-lane-title b {
            font-size: 1.08rem;
        }
        .architecture-node-chip {
            min-height: 86px;
            padding: 10px 11px;
        }
        .architecture-node-chip strong {
            font-size: 0.78rem;
        }
        .architecture-node-chip b {
            font-size: 0.98rem;
        }
        .architecture-node-chip span {
            font-size: 0.86rem;
            -webkit-line-clamp: 3;
        }
        .architecture-route-chip {
            font-size: 0.86rem;
            line-height: 1.4;
            padding: 9px 10px;
        }
        .architecture-route-chip b {
            font-size: 0.9rem;
        }
        .briefing-hero h2 {
            font-size: 1.95rem;
        }
        .briefing-hero p,
        .briefing-card span,
        .briefing-step span,
        .briefing-table td {
            font-size: 1.04rem;
            line-height: 1.56;
        }
        .briefing-card b,
        .briefing-step b,
        .briefing-table th {
            font-size: 1.08rem;
        }
        .briefing-step {
            min-height: 148px;
            padding: 15px;
        }
        .briefing-card {
            padding: 18px;
        }
        .foundation-flow-map {
            grid-template-columns:
                minmax(200px, 1fr) 32px
                minmax(200px, 1fr) 32px
                minmax(200px, 1fr) 32px
                minmax(200px, 1fr);
        }
        .foundation-flow-block {
            min-height: 330px;
            padding: 16px;
        }
        .foundation-flow-block b,
        .foundation-quality-card b {
            font-size: 1.14rem;
        }
        .foundation-flow-block span,
        .foundation-flow-item,
        .foundation-quality-card span {
            font-size: 1.04rem;
            line-height: 1.48;
        }
        .foundation-quality-card em {
            font-size: 0.9rem;
        }
        .proposal-slide {
            padding: clamp(17px, 1.5vw, 31px);
        }
        .proposal-topline {
            margin-bottom: 10px;
        }
        .proposal-kicker {
            font-size: 0.86rem;
        }
        .proposal-dot {
            width: 8px;
            height: 8px;
        }
        .proposal-dot.active {
            width: 28px;
        }
        .proposal-slide h2 {
            font-size: clamp(2.45rem, 3.1vw, 3.5rem);
            line-height: 1.07;
            margin-bottom: 10px;
        }
        .proposal-lead {
            font-size: clamp(1.2rem, 1.28vw, 1.36rem);
            line-height: 1.43;
            max-width: 1220px;
        }
        .proposal-panel {
            padding: 20px;
        }
        .proposal-panel b {
            font-size: 1.18rem;
        }
        .proposal-panel span,
        .proposal-panel li {
            font-size: 1.14rem;
            line-height: 1.48;
        }
        .fragment-card {
            min-height: 148px;
            padding: 17px;
        }
        .fragment-card em,
        .fpna-expectation em,
        .flow-node em,
        .tech-layer em,
        .option-card em,
        .roadmap-phase em,
        .assessment-item em {
            font-size: 0.84rem;
        }
        .fragment-card b,
        .fpna-expectation b {
            font-size: 1.12rem;
        }
        .fragment-card span {
            font-size: 1.08rem;
            line-height: 1.46;
        }
        .expectation-items div {
            font-size: 1rem;
            padding: 8px 10px;
        }
        .problem-connector {
            font-size: 0.95rem;
        }
        .fragment-stage.problem-blockers .fragment-card {
            min-height: 128px;
            padding: 12px 12px 12px 14px;
        }
        .fragment-stage.problem-blockers .fragment-card b {
            font-size: 1.05rem;
        }
        .fragment-stage.problem-blockers .fragment-card span {
            font-size: 0.98rem;
            line-height: 1.36;
        }
        .executive-question {
            font-size: 1.06rem;
            padding: 8px 12px;
        }
        .flow-node {
            min-height: 182px;
            padding: 18px;
        }
        .flow-node b {
            font-size: 1.16rem;
        }
        .flow-node span {
            font-size: 1rem;
            line-height: 1.44;
        }
        .foundation-core {
            font-size: 1.42rem;
            min-height: 100px;
        }
        .tech-architecture {
            grid-template-columns: 1.12fr 0.18fr 1.12fr 0.18fr 1.12fr 0.18fr 1.08fr 0.18fr 1.08fr;
            gap: 7px;
        }
        .tech-layer {
            min-height: 286px;
            padding: 12px;
        }
        .tech-layer em {
            font-size: 0.95rem;
        }
        .tech-layer b {
            font-size: 1.08rem;
            line-height: 1.24;
        }
        .tech-pill {
            font-size: 0.94rem;
            line-height: 1.24;
            padding: 6px 7px;
        }
        .tech-footnote div {
            font-size: 0.95rem;
            line-height: 1.32;
            padding: 8px 9px;
        }
        .option-card {
            min-height: 316px;
        }
        .option-card b,
        .roadmap-phase b,
        .assessment-item b {
            font-size: 1.18rem;
            line-height: 1.32;
        }
        .option-card span,
        .roadmap-phase span,
        .assessment-item span {
            font-size: 1.02rem;
            line-height: 1.46;
        }
        .option-meta {
            font-size: 0.96rem;
        }
        .roadmap-phase {
            min-height: 340px;
            padding: 20px;
        }
        .roadmap-gate {
            font-size: 0.98rem;
        }
        .assessment-item {
            min-height: 162px;
            padding: 18px;
        }
        .proposal-close {
            font-size: 1.18rem;
            line-height: 1.46;
            padding: 15px 18px;
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide {
            min-height: calc(100vh - 84px);
            padding: clamp(34px, 3.2vw, 60px);
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide h2 {
            font-size: clamp(2.65rem, 3.2vw, 3.45rem);
        }
        @media (max-width: 900px) {
            .cockpit-title {
                align-items: flex-start;
                flex-direction: column;
            }
            .pill-row {
                justify-content: flex-start;
            }
            .status-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .snapshot-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .story-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .briefing-grid,
            .briefing-flow {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .presentation-progress {
                align-items: flex-start;
                flex-direction: column;
            }
            .presentation-slide {
                min-height: auto;
                padding: 24px 18px;
            }
            .presentation-slide h2 {
                font-size: 1.8rem;
            }
            .presentation-lead,
            .presentation-card span,
            .presentation-metric-card span,
            .presentation-step span,
            .proposal-panel span,
            .proposal-panel li {
                font-size: 1rem;
            }
            .presentation-card-grid.is-2,
            .presentation-card-grid.is-3,
            .presentation-card-grid.is-4,
            .presentation-card-grid.is-5,
            .presentation-flow {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .architecture-diagram {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .architecture-lane-map,
            .architecture-lane-grid,
            .architecture-key-routes {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .architecture-lane {
                min-height: auto;
            }
            .architecture-lane-arrow {
                min-height: 26px;
                transform: rotate(90deg);
            }
            .architecture-node:not(:last-child)::after {
                content: "\\2193";
                right: 16px;
                top: auto;
                bottom: -19px;
                transform: none;
            }
            .foundation-summary-grid,
            .foundation-thesis-row,
            .foundation-split,
            .foundation-detail-grid,
            .lineage-strip,
            .foundation-domain-grid,
            .foundation-gate-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .foundation-hero {
                padding: 22px 18px;
            }
            .fragment-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .foundation-flow-map,
            .foundation-quality-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .architecture-object-grid,
            .architecture-edge-row {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .foundation-flow-arrow {
                min-height: 28px;
                transform: rotate(90deg);
            }
            .metric-value {
                white-space: normal;
            }
            .snapshot-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .proposal-topline {
                align-items: flex-start;
                flex-direction: column;
            }
            .proposal-grid-2,
            .proposal-grid-3,
            .problem-map,
            .expectation-items,
            .fragment-stage,
            .flow-ribbon,
            .foundation-quadrant,
            .roadmap,
            .assessment-canvas,
            .tech-footnote {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .proposal-slide {
                min-height: auto;
                padding: 24px 18px;
            }
            .proposal-slide h2 {
                font-size: 2rem;
            }
            .fpna-expectation {
                display: block;
            }
            .proposal-lead {
                font-size: 1.06rem;
            }
            .flow-node:not(:last-child)::after {
                bottom: -22px;
                right: 18px;
                top: auto;
                transform: rotate(90deg);
            }
            .tech-architecture {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
            .tech-arrow {
                min-height: 24px;
                transform: rotate(90deg);
            }
            .tech-layer {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="cockpit-title">
            <div>
                <h1>{escape(title)}</h1>
                <div class="subtitle">{escape(COMPANY_NAME)} | {escape(subtitle)}</div>
            </div>
            <div class="pill-row">
                <span class="pill">Fictional FY2026 data</span>
                <span class="pill">No real company data</span>
                <span class="pill">Rule-based AI comments</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_guide(title: str, body: str, points: list[str], takeaway: str) -> None:
    point_items = "".join(f"<li>{escape(point)}</li>" for point in points)
    st.markdown(
        f"""
        <div class="guide-panel">
            <div class="guide-title">{escape(title)}</div>
            <p>{escape(body)}</p>
            <ul>{point_items}</ul>
            <div class="guide-takeaway">{escape(takeaway)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_segment_story_cards() -> None:
    stories = [
        (
            "航空・防衛",
            "売上は堅調。ただし外注費、為替、長納期部品が利益率とキャッシュフローを圧迫。",
        ),
        (
            "エネルギーシステム",
            "高採算案件の前倒しで売上・営業利益が上振れ。全社のプラス要因。",
        ),
        (
            "船舶・海洋",
            "EAC悪化、設計変更、納期遅延により営業利益が大幅悪化。赤字化リスク案件あり。",
        ),
        (
            "産業機械・ロボット",
            "需要鈍化で売上が下振れ。固定費負担により利益率も悪化。",
        ),
    ]
    cols = st.columns(4, gap="small")
    for col, (title, text) in zip(cols, stories):
        with col:
            st.markdown(
                f"""
                <div class="story-card">
                    <b>{escape(title)}</b>
                    <span>{escape(text)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_presentation_controls(deck_key: str, titles: list[str]) -> int:
    index_key = f"{deck_key}_slide_index"
    focus_key = f"{deck_key}_focus_mode"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    if focus_key not in st.session_state:
        st.session_state[focus_key] = False

    index = int(st.session_state.get(index_key, 0))
    index = max(0, min(index, len(titles) - 1))
    st.session_state[index_key] = index
    focus_mode = bool(st.session_state.get(focus_key, False))

    def set_slide(next_index: int) -> None:
        st.session_state[index_key] = max(0, min(next_index, len(titles) - 1))

    def toggle_focus_mode() -> None:
        st.session_state[focus_key] = not bool(st.session_state.get(focus_key, False))

    controls = st.columns([1.15, *([0.48] * len(titles)), 1.15, 0.98], gap="small")
    with controls[0]:
        st.button(
            "← 前へ",
            key=f"{deck_key}_prev",
            disabled=index == 0,
            width="stretch",
            on_click=set_slide,
            args=(index - 1,),
        )
    for slide_index, _title in enumerate(titles):
        with controls[slide_index + 1]:
            st.button(
                str(slide_index + 1),
                key=f"{deck_key}_jump_{slide_index}",
                type="primary" if slide_index == index else "secondary",
                width="stretch",
                on_click=set_slide,
                args=(slide_index,),
            )
    with controls[-2]:
        st.button(
            "次へ →",
            key=f"{deck_key}_next",
            disabled=index == len(titles) - 1,
            width="stretch",
            on_click=set_slide,
            args=(index + 1,),
        )
    with controls[-1]:
        focus_label = "通常表示" if focus_mode else "大きく表示"
        st.button(focus_label, key=f"{deck_key}_focus_toggle", width="stretch", on_click=toggle_focus_mode)

    if focus_mode:
        st.markdown('<div id="presentation-focus-mode"></div>', unsafe_allow_html=True)

    dots = "".join(
        f'<span class="presentation-dot{" active" if i == index else ""}"></span>'
        for i in range(len(titles))
    )
    st.markdown(
        f"""
        <div class="presentation-progress">
            <span>Slide {index + 1} / {len(titles)}</span>
            <b>{escape(titles[index])}</b>
            <div class="presentation-dots">{dots}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return index


def presentation_cards_html(cards: list[tuple[str, str]], columns: int = 3) -> str:
    items = "".join(
        (
            '<div class="presentation-card">'
            f"<b>{escape(title)}</b>"
            f"<span>{escape(text)}</span>"
            "</div>"
        )
        for title, text in cards
    )
    return f'<div class="presentation-card-grid is-{columns}">{items}</div>'


def presentation_metric_cards_html(cards: list[tuple[str, str, str]], columns: int = 4) -> str:
    items = "".join(
        (
            '<div class="presentation-metric-card">'
            f"<b>{escape(label)}</b>"
            f"<strong>{escape(value)}</strong>"
            f"<span>{escape(note)}</span>"
            "</div>"
        )
        for label, value, note in cards
    )
    return f'<div class="presentation-card-grid is-{columns}">{items}</div>'


def presentation_flow_html(steps: list[tuple[str, str]], columns: int = 5) -> str:
    items = "".join(
        (
            '<div class="presentation-step">'
            f"<b>{escape(title)}</b>"
            f"<span>{escape(text)}</span>"
            "</div>"
        )
        for title, text in steps
    )
    return f'<div class="presentation-flow is-{columns}">{items}</div>'


def architecture_diagram_html(nodes: list[tuple[str, str, str]]) -> str:
    items = "".join(
        (
            '<div class="architecture-node">'
            f"<strong>{escape(layer)}</strong>"
            f"<b>{escape(title)}</b>"
            f"<span>{escape(text)}</span>"
            "</div>"
        )
        for layer, title, text in nodes
    )
    return f'<div class="architecture-diagram">{items}</div>'


def architecture_graph_system_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "id": "erp",
            "label": "ERP\n実績会計",
            "layer": "Source",
            "detail": "売上、原価、営業利益、キャッシュフローの正本。",
            "color": "#1f77b4",
            "size": (150, 64),
        },
        {
            "id": "epm",
            "label": "EPM\n予算・見込",
            "layer": "Source",
            "detail": "予算、前回見込、最新見込、中期計画の比較軸。",
            "color": "#1f77b4",
            "size": (150, 64),
        },
        {
            "id": "project",
            "label": "案件管理\nEAC・工程",
            "layer": "Source",
            "detail": "案件EAC、設計変更、納期、PMO管理情報。",
            "color": "#1f77b4",
            "size": (165, 66),
        },
        {
            "id": "procurement",
            "label": "調達・外注\n契約/価格",
            "layer": "Source",
            "detail": "材料費、外注費、長納期部品、サプライヤ情報。",
            "color": "#1f77b4",
            "size": (165, 66),
        },
        {
            "id": "master",
            "label": "マスタ\n案件/勘定",
            "layer": "Governance",
            "detail": "案件ID、セグメント、勘定科目、顧客、組織の共通定義。",
            "color": "#64748b",
            "size": (150, 64),
        },
        {
            "id": "integration",
            "label": "データ連携\n品質ゲート",
            "layer": "Data",
            "detail": "照合、版管理、ID統合、粒度固定、リネージを確認。",
            "color": "#f59e0b",
            "size": (170, 70),
        },
        {
            "id": "dwh",
            "label": "DWH / Lakehouse",
            "layer": "Data",
            "detail": "既存の全社データ基盤。履歴、権限、リネージを保持。",
            "color": "#0f766e",
            "size": (170, 66),
        },
        {
            "id": "fpa_mart",
            "label": "FP&A\nData Mart",
            "layer": "Data",
            "detail": "実績、予算、見込、差異、案件リスクを経営説明粒度で統合。",
            "color": "#0f766e",
            "size": (170, 70),
        },
        {
            "id": "semantic",
            "label": "KPI / 差異\nSemantic Layer",
            "layer": "Logic",
            "detail": "売上、利益、CF、EAC、差異要因を同じ定義で計算。",
            "color": "#14b8a6",
            "size": (180, 72),
        },
        {
            "id": "ai_service",
            "label": "AI説明\nService",
            "layer": "AI",
            "detail": "根拠引用、プロンプト管理、出力評価、監査ログを持つ説明サービス。",
            "color": "#8b5cf6",
            "size": (170, 70),
        },
        {
            "id": "ai_platform",
            "label": "AI共通基盤\nLLM Gateway",
            "layer": "AI",
            "detail": "モデルゲートウェイ、RAG、ガードレール、利用ログを共通管理。",
            "color": "#7c3aed",
            "size": (185, 70),
        },
        {
            "id": "cockpit",
            "label": "FP&A\nCockpit",
            "layer": "Experience",
            "detail": "ダッシュボード、差異分析、案件リスク、AIコメントを表示。",
            "color": "#06b6d4",
            "size": (165, 70),
        },
        {
            "id": "meeting",
            "label": "経営会議\n意思決定",
            "layer": "Business",
            "detail": "会議資料、確認事項、承認、次アクションへ接続。",
            "color": "#ef4444",
            "size": (170, 70),
        },
        {
            "id": "workflow",
            "label": "承認・通知\nAction",
            "layer": "Workflow",
            "detail": "タスク、承認、通知、案件アクション管理に接続。",
            "color": "#ec4899",
            "size": (165, 66),
        },
        {
            "id": "control",
            "label": "統制\n権限・監査",
            "layer": "Governance",
            "detail": "権限、承認、監査ログ、品質ゲート、モデル変更管理。",
            "color": "#64748b",
            "size": (165, 66),
        },
    ]
    edges = [
        ("erp", "integration", "実績"),
        ("epm", "integration", "予算/見込"),
        ("project", "integration", "EAC/工程"),
        ("procurement", "integration", "調達影響"),
        ("master", "integration", "共通ID"),
        ("integration", "dwh", "標準化"),
        ("dwh", "fpa_mart", "FP&A粒度"),
        ("fpa_mart", "semantic", "KPI/差異"),
        ("semantic", "cockpit", "可視化"),
        ("semantic", "ai_service", "根拠データ"),
        ("ai_platform", "ai_service", "LLM/RAG"),
        ("ai_service", "cockpit", "AIコメント"),
        ("cockpit", "meeting", "会議説明"),
        ("meeting", "workflow", "アクション"),
        ("workflow", "project", "案件更新"),
        ("control", "integration", "品質管理"),
        ("control", "ai_service", "出力統制"),
        ("control", "workflow", "承認ログ"),
    ]
    return nodes, [
        {
            "start": start,
            "end": end,
            "label": label,
            "color": "#39c5bb" if start != "control" else "#94a3b8",
            "thickness": 2.0 if start != "control" else 1.4,
        }
        for start, end, label in edges
    ]


def architecture_graph_current_demo_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "id": "parquet",
            "label": "Parquet\n架空データ",
            "layer": "Data",
            "detail": "dim_projects、fact_finance、variance_drivers、project_risk。",
            "color": "#0f766e",
            "size": (170, 70),
        },
        {
            "id": "metadata",
            "label": "JSON\nmetadata",
            "layer": "Data",
            "detail": "生成日時、行数、デモ前提を保持。",
            "color": "#0f766e",
            "size": (150, 64),
        },
        {
            "id": "load",
            "label": "load_data()\ncache",
            "layer": "Process",
            "detail": "Parquet/JSONを読み、Streamlit cacheで再利用。",
            "color": "#f59e0b",
            "size": (160, 68),
        },
        {
            "id": "semantic",
            "label": "KPI生成\npandas",
            "layer": "Logic",
            "detail": "売上、営業利益、利益率、CF、差異要因を計算。",
            "color": "#14b8a6",
            "size": (170, 70),
        },
        {
            "id": "comment",
            "label": "AIコメント\n決定的生成",
            "layer": "AI",
            "detail": "外部APIを使わず、集計済みの差異要因を日本語テンプレート化。",
            "color": "#8b5cf6",
            "size": (180, 72),
        },
        {
            "id": "client",
            "label": "client_app.py\n本体デモURL",
            "layer": "Experience",
            "detail": "クライアントが触る4画面のみを表示。",
            "color": "#06b6d4",
            "size": (180, 72),
        },
        {
            "id": "presenter",
            "label": "presenter_app.py\n説明者URL",
            "layer": "Experience",
            "detail": "社外向け説明資料と社内確認ページを表示。",
            "color": "#06b6d4",
            "size": (190, 72),
        },
        {
            "id": "github",
            "label": "GitHub\nmain",
            "layer": "Ops",
            "detail": "変更履歴と公開対象ファイルの正本。",
            "color": "#64748b",
            "size": (145, 64),
        },
        {
            "id": "streamlit",
            "label": "Streamlit Cloud\nDeploy",
            "layer": "Ops",
            "detail": "入口ファイルごとにURLを分けて公開。",
            "color": "#ef4444",
            "size": (185, 70),
        },
    ]
    edges = [
        ("parquet", "load", "read"),
        ("metadata", "load", "read"),
        ("load", "semantic", "DataFrame"),
        ("semantic", "comment", "集計値"),
        ("semantic", "client", "KPI/リスク"),
        ("comment", "client", "コメント"),
        ("semantic", "presenter", "説明材料"),
        ("github", "streamlit", "push"),
        ("streamlit", "client", "client URL"),
        ("streamlit", "presenter", "presenter URL"),
    ]
    return nodes, [
        {"start": start, "end": end, "label": label, "color": "#39c5bb", "thickness": 2.0}
        for start, end, label in edges
    ]


def yfiles_node_label(item: dict[str, Any]) -> Any:
    text = item.get("properties", {}).get("label", item.get("label", ""))
    if LabelStyle is None:
        return text
    return LabelStyle(
        text=text,
        font_size=16,
        font_weight=FontWeight.BOLD if FontWeight is not None else None,
        color="#f8fdff",
        maximum_width=150,
        wrapping=TextWrapping.WORD if TextWrapping is not None else None,
        text_alignment=TextAlignment.CENTER if TextAlignment is not None else None,
    )


def yfiles_edge_label(item: dict[str, Any]) -> Any:
    text = item.get("properties", {}).get("label", item.get("label", ""))
    if LabelStyle is None:
        return text
    return LabelStyle(
        text=text,
        font_size=12,
        color="#d9fffb",
        background_color="rgba(6, 15, 24, 0.82)",
        maximum_width=120,
        wrapping=TextWrapping.WORD if TextWrapping is not None else None,
        text_alignment=TextAlignment.CENTER if TextAlignment is not None else None,
    )


def html_fragment(html: str) -> str:
    return "\n".join(line.strip() for line in dedent(html).strip().splitlines())


def safe_hex_color(value: Any, fallback: str = "#39c5bb") -> str:
    text = str(value)
    if len(text) in {4, 7} and text.startswith("#") and all(char in "0123456789abcdefABCDEF" for char in text[1:]):
        return text
    return fallback


def architecture_lane_map_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    lanes = [
        ("Data sources", "根拠データ"),
        ("Foundation + AI", "説明基盤・AI"),
        ("Business use", "業務出口"),
    ]
    lane_for_layer = {
        "Source": 0,
        "Governance": 0,
        "Data": 1,
        "Process": 1,
        "Logic": 1,
        "AI": 1,
        "Experience": 2,
        "Business": 2,
        "Workflow": 2,
        "Ops": 2,
    }
    buckets: list[list[dict[str, Any]]] = [[], [], []]
    for node in nodes:
        layer = str(node.get("layer", ""))
        buckets[lane_for_layer.get(layer, 1)].append(node)

    lane_parts = []
    for index, (eyebrow, title) in enumerate(lanes):
        card_html = "".join(
            (
                f'<div class="architecture-node-chip" style="--accent:{safe_hex_color(node.get("color"))}">'
                f'<strong>{escape(str(node.get("layer", "")))}</strong>'
                f'<b>{escape(str(node.get("label", "")).replace(chr(10), " / "))}</b>'
                f'<span>{escape(str(node.get("detail", "")))}</span>'
                "</div>"
            )
            for node in buckets[index]
        )
        lane_parts.append(
            f"""
            <div class="architecture-lane">
                <div class="architecture-lane-title">
                    <em>{escape(eyebrow)}</em>
                    <b>{escape(title)}</b>
                </div>
                <div class="architecture-lane-grid">{card_html}</div>
            </div>
            """
        )
        if index < len(lanes) - 1:
            lane_parts.append('<div class="architecture-lane-arrow">→</div>')

    node_lookup = {node["id"]: node for node in nodes}
    route_html = "".join(
        (
            '<div class="architecture-route-chip">'
            f'<b>{escape(str(node_lookup.get(edge["start"], {}).get("label", edge["start"])).replace(chr(10), " / "))}</b>'
            f'<span>{escape(str(edge.get("label", "")))}</span> '
            f'{escape(str(node_lookup.get(edge["end"], {}).get("label", edge["end"])).replace(chr(10), " / "))}'
            "</div>"
        )
        for edge in edges[:4]
    )
    return html_fragment(
        f"""
        <div class="architecture-lane-map">{"".join(lane_parts)}</div>
        <div class="architecture-key-routes">{route_html}</div>
        """
    )


def architecture_graph_fallback_html(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    node_lookup = {node["id"]: node for node in nodes}
    node_html = "".join(
        (
            '<div class="architecture-object-card">'
            f'<strong>{escape(str(node.get("layer", "")))}</strong>'
            f'<b>{escape(str(node.get("label", "")).replace(chr(10), " / "))}</b>'
            f'<span>{escape(str(node.get("detail", "")))}</span>'
            "</div>"
        )
        for node in nodes
    )
    edge_html = "".join(
        (
            '<div class="architecture-edge-row">'
            f'<b>{escape(str(node_lookup.get(edge["start"], {}).get("label", edge["start"])).replace(chr(10), " / "))}</b>'
            f'<span>{escape(str(edge.get("label", "")))}</span>'
            f'<b>{escape(str(node_lookup.get(edge["end"], {}).get("label", edge["end"])).replace(chr(10), " / "))}</b>'
            "</div>"
        )
        for edge in edges
    )
    return html_fragment(
        f"""
    <div class="architecture-fallback">
        <div class="architecture-object-grid">{node_html}</div>
        <div class="presentation-section-divider"><span>Connections</span></div>
        <div class="architecture-edge-list">{edge_html}</div>
    </div>
    """
    )


def render_yfiles_architecture_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    key: str,
) -> None:
    if StreamlitGraphWidget is None or Node is None or Edge is None:
        st.markdown(architecture_graph_fallback_html(nodes, edges), unsafe_allow_html=True)
        return

    graph_nodes = [
        Node(
            id=node["id"],
            properties={
                "label": node["label"],
                "layer": node.get("layer", ""),
                "detail": node.get("detail", ""),
                "color": node.get("color", "#39c5bb"),
                "size": node.get("size", (160, 68)),
            },
        )
        for node in nodes
    ]
    graph_edges = [
        Edge(
            start=edge["start"],
            end=edge["end"],
            properties={
                "label": edge.get("label", ""),
                "color": edge.get("color", "#39c5bb"),
                "thickness": edge.get("thickness", 2.0),
            },
        )
        for edge in edges
    ]

    try:
        widget = StreamlitGraphWidget(
            graph_nodes,
            graph_edges,
            node_label_mapping=yfiles_node_label,
            edge_label_mapping=yfiles_edge_label,
            node_color_mapping=lambda item: item.get("properties", {}).get("color", "#39c5bb"),
            edge_color_mapping=lambda item: item.get("properties", {}).get("color", "#39c5bb"),
            node_size_mapping=lambda item: tuple(item.get("properties", {}).get("size", (160, 68))),
            node_type_mapping=lambda item: item.get("properties", {}).get("layer", ""),
            edge_thickness_factor_mapping=lambda item: item.get("properties", {}).get("thickness", 2.0),
        )
        layout = Layout.HIERARCHIC if Layout is not None else None
        widget.show(
            directed=True,
            graph_layout=layout,
            sidebar={"enabled": True, "start_with": "Data"},
            overview=True,
            key=key,
        )
    except Exception as exc:  # pragma: no cover - depends on optional Streamlit component runtime.
        print(f"Architecture graph component failed; rendering static fallback ({type(exc).__name__}).")
        st.markdown(architecture_graph_fallback_html(nodes, edges), unsafe_allow_html=True)


def render_architecture_graph_slide(
    eyebrow: str,
    title: str,
    lead_html: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    key: str,
    notes_html: str,
    footer: str,
) -> None:
    lead = html_fragment(lead_html)
    notes = html_fragment(notes_html)
    diagram = architecture_lane_map_html(nodes, edges)

    st.markdown(
        html_fragment(
            f"""
        <div class="presentation-slide architecture-slide">
            <div class="presentation-eyebrow">{escape(eyebrow)}</div>
            <h2>{escape(title)}</h2>
            {lead}
            {presentation_divider_html("構成オブジェクトと接続")}
            {diagram}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        html_fragment(
            f"""
        <div class="presentation-graph-notes">
            {notes}
            <div class="presentation-footer">{escape(footer)}</div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def foundation_flow_map_html() -> str:
    columns = [
        (
            "Source data",
            "業務データ",
            ["ERP実績", "EPM予算・見込", "案件EAC", "調達・外注", "工程・納期", "マスタ"],
            "",
        ),
        (
            "Quality gates",
            "品質ゲート",
            ["照合", "版管理", "案件ID統合", "勘定分類", "時間粒度", "リネージ"],
            "hot",
        ),
        (
            "Trusted foundation",
            "FP&Aデータ基盤",
            ["共通KPI定義", "差異要因レイヤー", "案件リスクレイヤー", "根拠リンク", "更新・監査ログ"],
            "",
        ),
        (
            "AI-driven FP&A",
            "経営説明",
            ["根拠付きAIコメント", "案件アクション", "会議資料", "意思決定", "次回更新"],
            "ai",
        ),
    ]
    flow_parts = []
    for index, (eyebrow, title, items, variant) in enumerate(columns):
        item_html = "".join(f'<div class="foundation-flow-item">{escape(item)}</div>' for item in items)
        flow_parts.append(
            f"""
            <div class="foundation-flow-block {variant}">
                <span>{escape(eyebrow)}</span>
                <b>{escape(title)}</b>
                <div class="foundation-flow-items">{item_html}</div>
            </div>
            """
        )
        if index < len(columns) - 1:
            flow_parts.append('<div class="foundation-flow-arrow">→</div>')
    return f'<div class="foundation-flow-map">{"".join(flow_parts)}</div>'


def presentation_divider_html(label: str) -> str:
    return f'<div class="presentation-section-divider"><span>{escape(label)}</span></div>'


def foundation_quality_gates_html() -> str:
    gates = [
        (
            "RECONCILE",
            "数字が戻れる",
            "AIコメントの数字が、ERP実績や案件実績まで戻れる状態にする。",
        ),
        (
            "CONTEXT",
            "比較軸が揃っている",
            "予算、最新見込、前回見込、実績の締め時点と粒度を固定する。",
        ),
        (
            "LINEAGE",
            "案件に落ちる",
            "財務、EAC、調達、工程を案件単位で結び、打ち手へつなげる。",
        ),
    ]
    cards = "".join(
        (
            '<div class="foundation-quality-card">'
            f"<em>{escape(eyebrow)}</em>"
            f"<b>{escape(title)}</b>"
            f"<span>{escape(text)}</span>"
            "</div>"
        )
        for eyebrow, title, text in gates
    )
    return f'<div class="foundation-quality-grid">{cards}</div>'


def presentation_table_html(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
    <table class="presentation-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{row_html}</tbody>
    </table>
    """


def render_presentation_slide(eyebrow: str, title: str, body_html: str, footer: str = "") -> None:
    body = "\n".join(line.strip() for line in body_html.strip().splitlines())
    footer_html = f'<div class="presentation-footer">{escape(footer)}</div>' if footer else ""
    html = (
        '<div class="presentation-slide">\n'
        "<div>\n"
        f'<div class="presentation-eyebrow">{escape(eyebrow)}</div>\n'
        f"<h2>{escape(title)}</h2>\n"
        f"{body}\n"
        "</div>\n"
        f"{footer_html}\n"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def proposal_dots_html(active: int, total: int = 8) -> str:
    return "".join(
        f'<span class="proposal-dot{" active" if index == active else ""}"></span>'
        for index in range(1, total + 1)
    )


def render_proposal_slide(page_no: int, label: str, title: str, lead_html: str, body_html: str) -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    st.markdown(
        html_fragment(
            f"""
            <div class="presentation-slide proposal-slide">
                <div class="proposal-topline">
                    <div class="proposal-kicker"><b>{page_no:02d}/08</b>AI時代のFP&amp;Aデータ基盤リファレンス構成 / {escape(label)}</div>
                    <div class="proposal-dots">{proposal_dots_html(page_no)}</div>
                </div>
                <h2>{escape(title)}</h2>
                <div class="proposal-lead">{lead_html}</div>
                {body_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_proposal_problem_statement() -> None:
    render_proposal_slide(
        1,
        "導入・問題提起",
        "数字が出ていても、「なぜそうなのか」・「次に何をするか」が分からないと意味がない",
        """
        重工業では、売上が上振れていても、EAC悪化、設計変更、調達費高騰、工程遅延によって
        利益・キャッシュフローが悪化します。財務、案件、調達、工程が分断されていると、
        経営会議で「なぜ悪化したか」「どの案件に手を打つか」を即答しづらくなります。
        """,
        """
        <div class="problem-logic">
            <div class="problem-map">
                <div class="fpna-expectation">
                    <em>What FP&amp;A should provide</em>
                    <b>本来FP&amp;Aに求められること</b>
                    <div class="expectation-items">
                        <div>「なぜそうなのか」を説明する</div>
                        <div>「次に何をするか」を示す</div>
                        <div>誰が、いつまでに動くかを決める</div>
                    </div>
                </div>
                <div class="problem-connector"><span>4つの課題が<br>阻害</span></div>
                <div class="fragment-stage problem-blockers">
                    <div class="fragment-card" style="--accent:#60a5fa">
                        <em>Financials</em>
                        <b>数字がばらばらに散らばっている</b>
                        <span>売上、営業利益、利益率、CF、案件情報が別々の資料・システムに散在し、同じ粒度で見られない。</span>
                    </div>
                    <div class="fragment-card" style="--accent:#ff647c">
                        <em>Projects</em>
                        <b>案件と財務KPIのつながりが不明</b>
                        <span>EAC悪化、設計変更、追加原価が売上・利益・CFにどう影響するか追いにくい。</span>
                    </div>
                    <div class="fragment-card" style="--accent:#ffb000">
                        <em>Supply chain</em>
                        <b>調達コストの増減把握が遅い</b>
                        <span>材料費、外注費、長納期部品の変化が月次説明に反映されるまで時間がかかる。</span>
                    </div>
                    <div class="fragment-card" style="--accent:#39c5bb">
                        <em>Commentary</em>
                        <b>差異説明が属人的/職人芸</b>
                        <span>差異要因の特定、根拠確認、説明文作成が担当者の経験と暗黙知に依存している。</span>
                    </div>
                </div>
            </div>
            <div class="executive-question">
                AI活用の前提は、財務KPIと案件・調達・工程データがつながり、増減要因を確認できることです。
            </div>
        </div>
        """,
    )


def render_proposal_target_operating_model() -> None:
    render_proposal_slide(
        2,
        "目指す業務像",
        "KPI変動から原因・影響・打ち手まで一気通貫で説明する",
        """
        目指すのは、AIが文章を書くことではありません。経営会議前に、KPI変動、差異要因、
        重点案件、説明コメント案、意思決定・アクションが同じ根拠で揃っている状態です。
        """,
        """
        <div class="flow-ribbon">
            <div class="flow-node">
                <em>01</em>
                <b>KPI変動</b>
                <span>売上、営業利益、利益率、CFの変化を、期間・事業・案件粒度で把握する。</span>
            </div>
            <div class="flow-node">
                <em>02</em>
                <b>差異要因</b>
                <span>価格、数量、EAC、調達費、工程遅延など、説明すべき主因を絞る。</span>
            </div>
            <div class="flow-node">
                <em>03</em>
                <b>重点案件</b>
                <span>財務影響が大きい案件、赤字化リスク、追加原価の発生源へ落とす。</span>
            </div>
            <div class="flow-node">
                <em>04</em>
                <b>AIコメント案</b>
                <span>根拠データに戻れる形で、経営会議向けの説明文を生成する。</span>
            </div>
            <div class="flow-node">
                <em>05</em>
                <b>判断・アクション</b>
                <span>承認、追加分析、案件アクション、次回フォローへ接続する。</span>
            </div>
        </div>
        <div class="proposal-close">
            経営会議の前に「何が起きたか」「なぜ起きたか」「どこに手を打つか」が揃う業務状態を作ります。
        </div>
        """,
    )


def render_proposal_data_foundation() -> None:
    render_proposal_slide(
        3,
        "FP&Aデータ基盤",
        "説明に耐えるには、定義・バージョン・粒度・根拠を揃える必要がある",
        """
        データを集めるだけでは、経営会議で使える説明にはなりません。
        コメントから差異、案件、元データへ戻れるように、説明責任を支える4つの能力を揃えます。
        """,
        """
        <div class="foundation-quadrant">
            <div class="proposal-panel accent-cyan">
                <b>定義</b>
                <span>KPI、勘定科目、案件、セグメントの意味を揃え、会議ごとに数字の解釈が変わらない状態にする。</span>
            </div>
            <div class="proposal-panel accent-blue">
                <b>バージョン</b>
                <span>予算、前回見込、最新見込、実績の締め時点を管理し、比較軸を固定する。</span>
            </div>
            <div class="proposal-panel accent-amber">
                <b>粒度</b>
                <span>月次、事業、案件、顧客、勘定科目を接続し、全社KPIから案件まで降りられる状態にする。</span>
            </div>
            <div class="proposal-panel accent-red">
                <b>根拠</b>
                <span>コメント、差異、案件、元データまでの追跡性を保持し、説明内容を検証できるようにする。</span>
            </div>
        </div>
        <div class="foundation-core">
            データは品質が担保されて初めて価値を持ちます。
        </div>
        """,
    )


def ai_app_architecture_component_html() -> str:
    def node(
        node_id: str,
        label: str,
        x: int,
        y: int,
        type_: str,
        detail: str,
        color: str,
        w: int = 132,
        h: int = 58,
        classes: str = "",
        products: str = "",
    ) -> dict[str, Any]:
        return {
            "data": {
                "id": node_id,
                "label": label,
                "type": type_,
                "detail": detail,
                "color": color,
                "w": w,
                "h": h,
                "products": products,
            },
            "position": {"x": x, "y": y},
            "classes": f"{type_} {classes}".strip(),
        }

    def edge(edge_id: str, source: str, target: str, label: str, classes: str = "") -> dict[str, Any]:
        return {
            "data": {"id": edge_id, "source": source, "target": target, "label": label},
            "classes": classes,
        }

    patterns = {
        "embedded": {
            "nav": "1. FP&A / EPM製品内蔵AI型",
            "title": "FP&A / EPM製品内蔵AI型",
            "subtitle": "計画・予測・連結・開示の業務プロセス内でAIを使う構成",
            "products": "Anaplan / Pigment / Workday Adaptive Planning / Oracle Cloud EPM / CCH Tagetik",
            "fit": "予算策定、ローリングフォーキャスト、差異説明、連結、経営レポート",
            "strength": "FP&Aモデル、承認、権限、計画サイクルに近く、初期導入しやすい。",
            "watch": "AI活用範囲は製品標準機能に依存するため、横断データ分析や独自エージェントは別設計にする。",
            "focus": {
                "label": "FP&A / EPM Platform",
                "detail": "中心に置くのは計画・予測・承認・レポーティングを担うFP&A/EPM製品です。",
            },
            "nodes": [
                node("erp", "ERP / GL\nActuals", 80, 80, "source", "実績会計、売上、原価、利益の正本。", "#2563eb", products="SAP S/4HANA / Oracle Fusion Cloud ERP / Microsoft Dynamics 365 Finance"),
                node("crm", "CRM\nPipeline", 80, 165, "source", "受注見込、商談、顧客別売上の先行指標。", "#2563eb", products="Salesforce Sales Cloud / Microsoft Dynamics 365 Sales / HubSpot Enterprise"),
                node("hcm", "HCM\nWorkforce", 80, 250, "source", "人員、労務費、組織、要員計画。", "#2563eb", products="Workday HCM / SAP SuccessFactors / Oracle HCM Cloud"),
                node("scm", "SCM / Project\nEAC", 80, 335, "source", "調達、工程、案件EAC、原価見通し。", "#2563eb", products="Oracle Primavera P6 / SAP Project System / EcoSys / IFS Cloud"),
                node("excel", "Local Inputs\nManual Adjustments", 80, 420, "source", "部門別入力、補正、管理表。", "#2563eb", products="Microsoft Excel / Google Sheets / Smartsheet / Airtable"),
                node("etl", "API / ETL\nData Sync", 255, 256, "data", "各システムから計画粒度へデータを同期。", "#0f766e", 142, 62, products="Informatica / Fivetran / dbt / Azure Data Factory / MuleSoft"),
                node("datahub", "DWH / Data Hub\nReference Data", 440, 256, "data", "必要な実績、マスタ、履歴だけをFP&A製品へ供給。", "#0f766e", 160, 64, products="Snowflake / Databricks / Microsoft Fabric / BigQuery"),
                node(
                    "fpna",
                    "FP&A / EPM Platform\nPlanning / Forecast\nConsolidation / Reporting",
                    705,
                    256,
                    "fpna",
                    "予算、見込、シナリオ、承認、レポートの業務プロセスを保持。",
                    "#7c3aed",
                    230,
                    100,
                    "primary",
                    products="Anaplan / Pigment / Workday Adaptive Planning / Oracle Cloud EPM / CCH Tagetik",
                ),
                node("embedded_ai", "Built-in AI\nForecast / Narrative", 705, 104, "ai", "製品内の予測、差異説明、レポート文案生成。", "#a855f7", 190, 68, products="Anaplan Intelligence / Pigment AI / Workday AI / Oracle AI for EPM / CCH Tagetik AI"),
                node("workflow", "Workflow\nApproval", 705, 430, "control", "計画提出、レビュー、承認、差戻し、通知。", "#64748b", 176, 64, products="ServiceNow / Jira / Microsoft Teams / Power Automate"),
                node("governance", "Governance\nIAM / Audit", 420, 430, "control", "権限、監査ログ、版管理、変更履歴。", "#64748b", 168, 64, products="Microsoft Entra ID / Okta / Microsoft Purview / Collibra / Atlan"),
                node("variance", "Variance\nExplanation", 985, 116, "usecase", "予実差異・見込差異の説明。", "#dc2626", products="Anaplan / Pigment / Oracle Cloud EPM / CCH Tagetik"),
                node("forecast", "Rolling\nForecast", 985, 216, "usecase", "月次・週次の見込更新。", "#dc2626", products="Anaplan / Pigment / Workday Adaptive Planning / Oracle Cloud EPM"),
                node("scenario", "Scenario\nPlanning", 985, 316, "usecase", "価格、為替、案件EACのシナリオ。", "#dc2626", products="Anaplan / Pigment / Oracle Crystal Ball / CCH Tagetik"),
                node("reporting", "Narrative\nReporting", 985, 416, "usecase", "経営会議、取締役会資料の文案。", "#dc2626", products="Workiva / Microsoft Power BI / Tableau / Oracle Cloud EPM Narrative Reporting"),
            ],
            "edges": [
                edge("e1", "erp", "etl", "actuals"),
                edge("e2", "crm", "etl", "pipeline"),
                edge("e3", "hcm", "etl", "workforce"),
                edge("e4", "scm", "etl", "EAC"),
                edge("e5", "excel", "etl", "inputs"),
                edge("e6", "etl", "datahub", "standardize"),
                edge("e7", "datahub", "fpna", "load"),
                edge("e8", "embedded_ai", "fpna", "AI in process", "ai-edge"),
                edge("e9", "fpna", "variance", "explain"),
                edge("e10", "fpna", "forecast", "forecast"),
                edge("e11", "fpna", "scenario", "simulate"),
                edge("e12", "fpna", "reporting", "draft"),
                edge("e13", "governance", "datahub", "lineage", "control-edge"),
                edge("e14", "governance", "fpna", "access", "control-edge"),
                edge("e15", "workflow", "fpna", "approval", "control-edge"),
            ],
        },
        "data_ai": {
            "nav": "2. データ基盤AI型",
            "title": "データ基盤AI型",
            "subtitle": "全社データ基盤上でAI分析・予測・RAGを行い、FP&Aへ結果連携する構成",
            "products": "Snowflake Cortex / Databricks Mosaic AI / Microsoft Fabric Copilot",
            "fit": "全社データ分析、異常検知、KPI要因分析、予測モデル、RAG",
            "strength": "ERP、CRM、SCM、HCMなど横断データに強く、独自ロジックや予測モデルを作りやすい。",
            "watch": "FP&A製品への書き戻し、承認フロー、数値の正本管理を明確にする必要がある。",
            "focus": {
                "label": "DWH / Lakehouse AI Foundation",
                "detail": "中心に置くのは全社データ基盤です。AIはデータの近くで動かし、結果をFP&A業務へ返します。",
            },
            "nodes": [
                node("systems", "ERP / CRM / HCM\nSCM / Project", 80, 180, "source", "実績、商談、人員、調達、案件EACを横断的に取得。", "#2563eb", 150, 74, products="SAP S/4HANA / Oracle ERP / Salesforce / Workday / MES・SCM systems"),
                node("external", "External Drivers\nFX / Market / Macro", 80, 330, "source", "為替、金利、市況、サプライヤ情報など外部説明変数。", "#2563eb", 150, 70, products="LSEG Workspace / Bloomberg / S&P Global / internal market feeds"),
                node("elt", "ELT / Streaming\nData Quality", 295, 255, "data", "取り込み、標準化、品質チェック、更新管理。", "#0f766e", 150, 64, products="Fivetran / Informatica / dbt / Airflow / Azure Data Factory"),
                node(
                    "lakehouse",
                    "DWH / Lakehouse\nAI Foundation",
                    535,
                    255,
                    "data",
                    "全社データ、履歴、権限、リネージを保持するAI活用の中心。",
                    "#0f766e",
                    180,
                    76,
                    "primary",
                    products="Snowflake / Databricks / Microsoft Fabric / BigQuery",
                ),
                node("catalog", "Data Catalog\nLineage / Policy", 535, 455, "control", "データ定義、リネージ、ポリシー、利用状況を管理。", "#64748b", 180, 64, products="Microsoft Purview / Collibra / Atlan / Unity Catalog / Snowflake Horizon"),
                node("ai_sql", "Native AI\nSQL / Notebook\nAssist", 790, 112, "ai", "SQL、ノートブック、対話型支援でAI分析を実行。", "#a855f7", 170, 78, products="Snowflake Cortex / Databricks Mosaic AI / Microsoft Fabric Copilot / BigQuery ML"),
                node("forecast_model", "Forecast Model\nML / AutoML", 790, 242, "ai", "ローリングフォーキャストやリスク予測をモデル化。", "#a855f7", 158, 64, products="Databricks AutoML / Azure Machine Learning / Vertex AI / Amazon SageMaker"),
                node("rag", "RAG / AI Search\nPolicy + Docs", 790, 365, "ai", "レポート、議事録、ルール、注記を検索拡張で参照。", "#a855f7", 158, 64, products="Azure AI Search / Databricks Vector Search / Snowflake Cortex Search / Vertex AI Search"),
                node("fpna_tools", "FP&A / EPM Platform\nPlanning Workflow", 1010, 255, "fpna", "計画、承認、予算、見込の業務運用を担う。", "#7c3aed", 170, 72, products="Anaplan / Pigment / Workday Adaptive Planning / Oracle Cloud EPM / CCH Tagetik"),
                node("variance", "KPI Driver\nAnalysis", 1225, 135, "usecase", "KPI変動の主因を横断データで分解。", "#dc2626", products="Snowflake Cortex / Databricks Mosaic AI / Microsoft Fabric Copilot / Power BI"),
                node("anomaly", "Anomaly\nDetection", 1225, 255, "usecase", "案件、原価、CFの異常を早期検知。", "#dc2626", products="Databricks / Azure Machine Learning / Vertex AI / Amazon SageMaker"),
                node("writeback", "Planning\nWrite-back", 1225, 375, "usecase", "予測結果やアラートを計画業務へ戻す。", "#dc2626", products="Anaplan APIs / Pigment API / Oracle EPM REST API / CCH Tagetik connectors"),
            ],
            "edges": [
                edge("d1", "systems", "elt", "internal data"),
                edge("d2", "external", "elt", "drivers"),
                edge("d3", "elt", "lakehouse", "curate"),
                edge("d4", "lakehouse", "ai_sql", "AI SQL/RAG", "ai-edge"),
                edge("d5", "lakehouse", "forecast_model", "features", "ai-edge"),
                edge("d6", "lakehouse", "rag", "documents", "ai-edge"),
                edge("d7", "ai_sql", "variance", "explain"),
                edge("d8", "forecast_model", "anomaly", "predict"),
                edge("d9", "forecast_model", "fpna_tools", "forecast"),
                edge("d10", "rag", "fpna_tools", "context"),
                edge("d11", "fpna_tools", "writeback", "workflow"),
                edge("d12", "catalog", "lakehouse", "policy", "control-edge"),
                edge("d13", "catalog", "fpna_tools", "lineage", "control-edge"),
            ],
        },
        "agent": {
            "nav": "3. 横断AIエージェント型",
            "title": "横断AIエージェント型",
            "subtitle": "業務エージェントと開発・運用エージェントが、複数システムをAPI・コネクタ経由で横断する構成",
            "products": "Azure OpenAI / Amazon Bedrock / Google Vertex AI",
            "fit": "経営問答、レポート自動作成、複数システム横断の業務支援、AIアプリ実装、データ連携、テスト、運用改善",
            "strength": "ベンダー横断で柔軟に設計でき、BI、FP&A、ERP、文書、会議情報をまたいだ業務体験と開発・運用支援を作れる。",
            "watch": "権限継承、監査、データ境界、コスト制御、Human-in-the-loopを最初から設計する。開発・運用エージェントは本番データや権限境界に直接触れさせず、リポジトリ、CI/CD、承認フローで統制する。",
            "focus": {
                "label": "Business AI Agent\nOrchestration Layer",
                "detail": "中心に置くのは業務横断のエージェント層です。各システムの正本を持たず、権限付きで参照・提案します。",
            },
            "nodes": [
                node("channels", "Users / Channels\nCFO + FP&A\nChat / Approval", 95, 260, "usecase", "経営層とFP&A担当が自然言語で問い、確認・承認・指示を行う入口。", "#dc2626", 184, 94, products="Microsoft 365 Copilot / Gemini Enterprise / Microsoft Teams / Slack / ServiceNow"),
                node("llm", "LLM Gateway\nModel Routing\nPrompt / Logs", 430, 95, "ai", "モデル選択、プロンプト、利用ログ、コスト制御。", "#a855f7", 210, 84, products="Azure OpenAI / Amazon Bedrock / Google Vertex AI"),
                node(
                    "agent",
                    "Business AI Agent\nPlanning + Tool Use\nAnswer + Draft",
                    430,
                    260,
                    "ai",
                    "ユーザー意図を解釈し、必要なシステムを呼び出して回答・提案を生成。",
                    "#a855f7",
                    230,
                    104,
                    "primary",
                    products="Microsoft Copilot Studio / LangGraph / Semantic Kernel / Vertex AI Agent Builder / Amazon Bedrock Agents",
                ),
                node("systems_api", "Business System APIs\nERP / FP&A / Docs", 735, 140, "source", "実績、計画、文書、会議情報をAPI・コネクタ経由で参照する接続面。", "#2563eb", 210, 86, products="SAP BTP APIs / Oracle Integration Cloud / MuleSoft / Boomi / Anaplan APIs / Pigment API / Oracle EPM REST API / CCH Tagetik connectors / SharePoint / Confluence / Google Drive"),
                node("data_access", "Data & BI Access\nDWH / Metrics / Search", 735, 295, "data", "横断分析、KPI定義、検索、特徴量、履歴データをまとめて参照する層。", "#0f766e", 210, 86, products="Snowflake / Databricks / Microsoft Fabric / BigQuery / Power BI / Tableau / Looker / Azure AI Search / Vertex AI Search"),
                node("governance", "Governance & Workflow\nIAM / Approval / Audit", 735, 455, "control", "権限継承、PII制御、監査ログ、承認、チケット、案件アクションを統制。", "#64748b", 220, 86, products="Microsoft Entra ID / Okta / Microsoft Purview / AWS IAM / Google Cloud IAM / ServiceNow / Jira / Power Automate"),
                node(
                    "engineering_agent",
                    "Engineering Agent\nAI SDLC / DevOps\nBuild + Operate",
                    430,
                    455,
                    "ai",
                    "データ連携、API、RAG、テスト、監視、ドキュメント更新を支援する開発・運用エージェント。",
                    "#a855f7",
                    220,
                    78,
                    products="Codex / Claude Code / Cursor / Devin / GitHub Copilot",
                ),
                node("outputs", "Management Outputs\nQ&A / Board Pack\nActions", 1020, 280, "usecase", "経営問答、会議資料草案、打ち手候補、担当、期限を出力する。", "#dc2626", 190, 100, products="Microsoft 365 Copilot / Gemini Enterprise / Microsoft PowerPoint / Workiva / Google Slides / Oracle Narrative Reporting / ServiceNow / Jira / Microsoft Planner / Asana"),
            ],
            "edges": [
                edge("a1", "channels", "agent", ""),
                edge("a2", "llm", "agent", "", "ai-edge"),
                edge("a3", "agent", "systems_api", ""),
                edge("a4", "agent", "data_access", ""),
                edge("a5", "agent", "outputs", ""),
                edge("a6", "governance", "agent", "", "control-edge"),
                edge("a7", "engineering_agent", "agent", "", "ai-edge"),
                edge("a8", "engineering_agent", "systems_api", "", "ai-edge"),
                edge("a9", "engineering_agent", "data_access", "", "ai-edge"),
                edge("a10", "governance", "engineering_agent", "", "control-edge"),
            ],
        },
    }
    patterns_json = json.dumps(patterns, ensure_ascii=False).replace("</", "<\\/")
    html = """
    <!doctype html>
    <html lang="ja">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
      <style>
        :root {
          --panel: #0b1722;
          --line: rgba(57,197,187,0.28);
          --cyan: #39c5bb;
          --text: #f8fdff;
          --muted: #9fb2c3;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          background: transparent;
          color: var(--text);
          font-family: "Inter", "Noto Sans JP", "Yu Gothic", "Meiryo", sans-serif;
          letter-spacing: 0;
        }
        .slide {
          position: relative;
          min-height: 800px;
          padding: 22px 24px;
          overflow: hidden;
          background:
            linear-gradient(135deg, rgba(9,18,29,0.99), rgba(5,12,20,0.99) 58%, rgba(18,26,37,0.99)),
            repeating-linear-gradient(90deg, rgba(57,197,187,0.07) 0 1px, transparent 1px 112px);
          border: 1px solid rgba(57,197,187,0.30);
          border-radius: 8px;
          box-shadow: 0 22px 70px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .slide::after {
          content: "";
          position: absolute;
          left: 0;
          right: 0;
          top: 0;
          height: 4px;
          background: linear-gradient(90deg, #39c5bb, #60a5fa, #ffb000, #ff647c);
        }
        .topline {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 18px;
          margin-bottom: 8px;
        }
        .kicker {
          color: var(--muted);
          font-size: 0.86rem;
          font-weight: 800;
          text-transform: uppercase;
        }
        .kicker b {
          color: var(--cyan);
          margin-right: 8px;
        }
        .dots { display: flex; gap: 6px; }
        .dot {
          width: 7px;
          height: 7px;
          border-radius: 999px;
          background: rgba(159,178,195,0.32);
        }
        .dot.active {
          width: 24px;
          background: var(--cyan);
          box-shadow: 0 0 18px rgba(57,197,187,0.55);
        }
        h1 {
          margin: 0 0 8px 0;
          color: var(--text);
          font-size: clamp(2.15rem, 2.85vw, 3.2rem);
          line-height: 1.08;
          letter-spacing: 0;
        }
        .lead {
          color: #d6e7ee;
          max-width: 1160px;
          margin: 0 0 10px 0;
          font-size: 1.08rem;
          line-height: 1.42;
        }
        .pattern-tabs {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
          margin: 12px 0 10px 0;
        }
        .pattern-tab {
          background: rgba(14,29,42,0.90);
          border: 1px solid rgba(159,178,195,0.20);
          border-radius: 8px;
          color: #c8dce6;
          cursor: pointer;
          font: inherit;
          font-size: 1rem;
          font-weight: 750;
          min-height: 42px;
          padding: 8px 10px;
          text-align: left;
        }
        .pattern-tab.active {
          background: rgba(57,197,187,0.14);
          border-color: rgba(57,197,187,0.58);
          color: #f8fdff;
          box-shadow: 0 0 0 1px rgba(57,197,187,0.18) inset;
        }
        .main-grid {
          display: grid;
          grid-template-columns: minmax(760px, 1fr) 340px;
          gap: 12px;
          align-items: stretch;
        }
        .graph-card,
        .side-card {
          background: linear-gradient(180deg, rgba(14,29,42,0.92), rgba(7,18,28,0.96));
          border: 1px solid rgba(57,197,187,0.25);
          border-radius: 8px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .graph-card {
          display: flex;
          flex-direction: column;
          padding: 10px;
          min-height: 580px;
        }
        .graph-title {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
          min-height: 38px;
          padding: 0 2px 8px 2px;
          border-bottom: 1px solid rgba(57,197,187,0.18);
        }
        .graph-title b {
          color: #f8fdff;
          display: block;
          font-size: 1.1rem;
        }
        .graph-title span {
          color: #9fb2c3;
          font-size: 0.9rem;
          white-space: nowrap;
        }
        .graph-title-meta {
          align-items: center;
          display: inline-flex;
          gap: 12px;
        }
        .click-cue {
          align-items: center;
          color: #d9fffb !important;
          display: inline-flex;
          font-size: 0.9rem !important;
          font-weight: 800;
          gap: 6px;
        }
        .click-cue i {
          animation: cuePulse 1.7s ease-in-out infinite;
          background: #ffb000;
          border-radius: 50%;
          box-shadow: 0 0 0 0 rgba(255,176,0,0.38);
          display: inline-block;
          height: 8px;
          width: 8px;
        }
        @keyframes cuePulse {
          0% { box-shadow: 0 0 0 0 rgba(255,176,0,0.38); }
          70% { box-shadow: 0 0 0 7px rgba(255,176,0,0); }
          100% { box-shadow: 0 0 0 0 rgba(255,176,0,0); }
        }
        #cy {
          flex: 1 1 auto;
          width: 100%;
          min-height: 460px;
          background:
            linear-gradient(90deg, rgba(57,197,187,0.04) 0 1px, transparent 1px 80px),
            linear-gradient(0deg, rgba(96,165,250,0.04) 0 1px, transparent 1px 70px),
            rgba(4, 10, 17, 0.42);
          border-radius: 6px;
          margin-top: 8px;
        }
        .side-card {
          max-height: 580px;
          overflow-x: hidden;
          overflow-y: auto;
          padding: 14px;
          min-height: 580px;
          scrollbar-color: rgba(57,197,187,0.62) rgba(7,18,28,0.92);
          scrollbar-gutter: stable;
        }
        .side-card::-webkit-scrollbar {
          width: 8px;
        }
        .side-card::-webkit-scrollbar-track {
          background: rgba(7,18,28,0.92);
          border-radius: 999px;
        }
        .side-card::-webkit-scrollbar-thumb {
          background: rgba(57,197,187,0.62);
          border-radius: 999px;
        }
        .side-group + .side-group {
          border-top: 1px solid rgba(57,197,187,0.24);
          margin-top: 14px;
          padding-top: 14px;
        }
        .section-label {
          color: #9fb2c3;
          display: block;
          font-size: 0.84rem;
          font-weight: 850;
          margin-bottom: 7px;
          text-transform: uppercase;
        }
        .side-eyebrow {
          color: var(--cyan);
          display: block;
          font-size: 0.86rem;
          font-weight: 850;
          margin-bottom: 5px;
          text-transform: uppercase;
        }
        .dynamic-head {
          align-items: center;
          display: flex;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 8px;
        }
        .change-note {
          color: #ffdf86;
          font-size: 0.84rem;
          font-weight: 850;
          white-space: nowrap;
        }
        .side-card h2 {
          color: #f8fdff;
          font-size: 1.32rem;
          line-height: 1.24;
          margin: 0 0 8px 0;
        }
        .side-card p {
          color: #c8dce6;
          font-size: 1rem;
          line-height: 1.45;
          margin: 0;
        }
        .info-block {
          border-top: 1px solid rgba(57,197,187,0.18);
          margin-top: 14px;
          padding-top: 12px;
        }
        .info-block.first {
          border-top: 0;
          margin-top: 10px;
          padding-top: 0;
        }
        .info-block b {
          color: #d9fffb;
          display: block;
          font-size: 0.92rem;
          margin-bottom: 5px;
        }
        .info-block span {
          color: #c8dce6;
          display: block;
          font-size: 0.96rem;
          line-height: 1.42;
        }
        .node-detail {
          background: rgba(57,197,187,0.09);
          border: 1px solid rgba(57,197,187,0.26);
          border-left: 4px solid #39c5bb;
          border-radius: 8px;
          margin-top: 10px;
          padding: 12px;
        }
        .node-detail strong {
          color: #f8fdff;
          display: block;
          font-size: 1.02rem;
          line-height: 1.32;
          margin-bottom: 5px;
          white-space: pre-line;
        }
        .node-detail span {
          color: #d9fffb;
          font-size: 0.96rem;
          line-height: 1.42;
        }
        .legend {
          display: flex;
          flex: 0 0 auto;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 8px;
        }
        .legend span {
          align-items: center;
          color: #9fb2c3;
          display: inline-flex;
          font-size: 0.9rem;
          gap: 5px;
        }
        .legend i {
          border-radius: 3px;
          display: inline-block;
          height: 10px;
          width: 10px;
        }
        .fallback {
          color: #d9fffb;
          padding: 24px;
          line-height: 1.6;
        }
        @media (max-width: 980px) {
          .slide { min-height: auto; padding: 22px; }
          .pattern-tabs { grid-template-columns: 1fr; }
          .main-grid { grid-template-columns: 1fr; }
          .graph-card, .side-card { min-height: auto; }
          .side-card { max-height: none; }
          #cy { height: 560px; }
          .graph-title { align-items: flex-start; flex-direction: column; }
          .graph-title-meta { align-items: flex-start; flex-direction: column; gap: 4px; }
        }
      </style>
    </head>
    <body>
      <div class="slide">
        <div class="topline">
          <div class="kicker"><b>04/08</b>AI時代のFP&amp;Aデータ基盤リファレンス構成 / AI App Architecture</div>
          <div class="dots" aria-hidden="true">
            <span class="dot"></span><span class="dot"></span><span class="dot"></span><span class="dot active"></span>
            <span class="dot"></span><span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
        </div>
        <h1>FP&amp;AプラットフォームにAIを組み込む3つの実装パターン</h1>
        <p class="lead">
          AIを単一製品の機能としてではなく、FP&amp;A業務、全社データ基盤、横断AIエージェントのどこに置くかで設計を分ける。
        </p>
        <div id="tabBar" class="pattern-tabs"></div>
        <div class="main-grid">
          <section class="graph-card">
            <div class="graph-title">
              <b id="patternTitle">Architecture</b>
              <div class="graph-title-meta">
                <span>Technical architecture view</span>
                <span class="click-cue"><i></i>オブジェクトをクリック</span>
              </div>
            </div>
            <div id="cy"></div>
            <div class="legend">
              <span><i style="background:#2563eb"></i>Source</span>
              <span><i style="background:#0f766e"></i>Data</span>
              <span><i style="background:#7c3aed"></i>FP&amp;A</span>
              <span><i style="background:#a855f7"></i>AI</span>
              <span><i style="background:#64748b"></i>Control</span>
              <span><i style="background:#dc2626"></i>Use case</span>
            </div>
          </section>
          <aside class="side-card">
            <section class="side-group">
              <div class="dynamic-head">
                <span class="section-label">選択中のオブジェクト</span>
                <span class="change-note">クリックで切替</span>
              </div>
              <div class="node-detail">
                <strong id="nodeLabel"></strong>
                <span id="nodeDetail"></span>
              </div>
              <div class="info-block">
                <b>代表製品</b>
                <span id="sideProducts"></span>
              </div>
            </section>
            <section class="side-group">
              <span class="section-label">固定情報</span>
              <span class="side-eyebrow">Implementation Pattern</span>
              <h2 id="sideTitle"></h2>
              <p id="sideSubtitle"></p>
              <div class="info-block first">
                <b>向いている用途</b>
                <span id="sideFit"></span>
              </div>
              <div class="info-block">
                <b>強み</b>
                <span id="sideStrength"></span>
              </div>
              <div class="info-block">
                <b>設計上の注意点</b>
                <span id="sideWatch"></span>
              </div>
            </section>
          </aside>
        </div>
      </div>
      <script>
        const patterns = __PATTERNS__;
        let cy = null;
        let activeKey = Object.keys(patterns)[0];

        function setText(id, value) {
          document.getElementById(id).textContent = value || "";
        }

        function showNodeDetail(data, fallbackProducts) {
          setText("nodeLabel", data.label || "");
          setText("nodeDetail", data.detail || "");
          setText("sideProducts", data.products || fallbackProducts || "");
        }

        function renderTabs() {
          const tabBar = document.getElementById("tabBar");
          tabBar.innerHTML = "";
          Object.entries(patterns).forEach(([key, pattern]) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "pattern-tab" + (key === activeKey ? " active" : "");
            button.textContent = pattern.nav;
            button.addEventListener("click", () => renderPattern(key));
            tabBar.appendChild(button);
          });
        }

        function renderPattern(key) {
          activeKey = key;
          const pattern = patterns[key];
          renderTabs();
          setText("patternTitle", pattern.title);
          setText("sideTitle", pattern.title);
          setText("sideSubtitle", pattern.subtitle);
          setText("sideProducts", pattern.products);
          setText("sideFit", pattern.fit);
          setText("sideStrength", pattern.strength);
          setText("sideWatch", pattern.watch);
          showNodeDetail(pattern.focus, pattern.products);

          if (!window.cytoscape) {
            document.getElementById("cy").innerHTML = '<div class="fallback">Cytoscape.jsを読み込めませんでした。ネットワーク接続またはCDN許可設定を確認してください。</div>';
            return;
          }
          if (cy) {
            cy.destroy();
          }
          document.getElementById("cy").innerHTML = "";
          cy = cytoscape({
            container: document.getElementById("cy"),
            elements: [...pattern.nodes, ...pattern.edges],
            layout: { name: "preset", fit: true, padding: 26 },
            minZoom: 0.42,
            maxZoom: 1.8,
            wheelSensitivity: 0.18,
            boxSelectionEnabled: false,
            autoungrabify: true,
            style: [
              {
                selector: "node",
                style: {
                  "shape": "round-rectangle",
                  "width": "data(w)",
                  "height": "data(h)",
                  "background-color": "data(color)",
                  "background-opacity": 0.95,
                  "border-width": 1.2,
                  "border-color": "rgba(248,253,255,0.30)",
                  "label": "data(label)",
                  "color": "#f8fdff",
                  "font-family": "Inter, Noto Sans JP, Yu Gothic, Meiryo, sans-serif",
                  "font-size": 12,
                  "font-weight": 700,
                  "line-height": 1.18,
                  "text-wrap": "wrap",
                  "text-max-width": 150,
                  "text-valign": "center",
                  "text-halign": "center",
                  "padding": "8px",
                  "overlay-opacity": 0
                }
              },
              {
                selector: "node.primary",
                style: {
                  "border-width": 3,
                  "border-color": "#f8fdff",
                  "font-size": 14,
                  "text-max-width": 225
                }
              },
              {
                selector: "node.control",
                style: {
                  "border-style": "dashed",
                  "border-color": "rgba(203,213,225,0.58)"
                }
              },
              {
                selector: "edge",
                style: {
                  "curve-style": "bezier",
                  "width": 2.1,
                  "line-color": "rgba(57,197,187,0.72)",
                  "target-arrow-shape": "triangle",
                  "target-arrow-color": "rgba(57,197,187,0.82)",
                  "label": "data(label)",
                  "color": "#d9fffb",
                  "font-family": "Inter, Noto Sans JP, Yu Gothic, Meiryo, sans-serif",
                  "font-size": 9.5,
                  "font-weight": 700,
                  "text-background-color": "#071016",
                  "text-background-opacity": 0.86,
                  "text-background-padding": "3px",
                  "text-rotation": "autorotate",
                  "z-index": 1
                }
              },
              {
                selector: "edge.ai-edge",
                style: {
                  "line-color": "rgba(168,85,247,0.86)",
                  "target-arrow-color": "rgba(168,85,247,0.95)",
                  "width": 2.6
                }
              },
              {
                selector: "edge.control-edge",
                style: {
                  "line-style": "dashed",
                  "line-color": "rgba(148,163,184,0.70)",
                  "target-arrow-color": "rgba(148,163,184,0.80)",
                  "width": 1.7
                }
              },
              {
                selector: "node:selected",
                style: {
                  "border-color": "#ffb000",
                  "border-width": 3
                }
              }
            ]
          });
          cy.on("tap", "node", (event) => {
            cy.elements().unselect();
            event.target.select();
            showNodeDetail(event.target.data(), pattern.products);
          });
          cy.on("mouseover", "node", () => {
            cy.container().style.cursor = "pointer";
          });
          cy.on("mouseout", "node", () => {
            cy.container().style.cursor = "default";
          });
          window.architectureCy = cy;
          window.architecturePatterns = patterns;
          window.selectArchitectureNode = function(id) {
            const node = cy.getElementById(id);
            if (!node || node.empty()) {
              return false;
            }
            cy.elements().unselect();
            node.select();
            showNodeDetail(node.data(), pattern.products);
            return true;
          };
          cy.ready(() => setTimeout(() => {
            const primaryNode = cy.nodes(".primary").first();
            if (primaryNode && primaryNode.length) {
              primaryNode.select();
            }
            cy.fit(cy.elements(), 28);
          }, 20));
        }

        window.addEventListener("resize", () => {
          if (cy) {
            cy.resize();
            cy.fit(cy.elements(), 28);
          }
        });
        renderPattern(activeKey);
      </script>
    </body>
    </html>
    """
    return html.replace("__PATTERNS__", patterns_json)


def render_proposal_ai_app_architecture() -> None:
    st.markdown('<div id="demo-briefing-page"></div><div id="proposal-component-page"></div>', unsafe_allow_html=True)
    components.html(ai_app_architecture_component_html(), height=802, scrolling=False)


def render_proposal_system_architecture() -> None:
    render_proposal_slide(
        5,
        "システムアーキテクチャ",
        "既存システムの上に、FP&AデータマートとAI説明サービスを配置する",
        """
        既存のERP、EPM、案件管理、調達、工程、マスタを置き換えるのではなく、
        上位にFP&A説明用のデータマート、KPI・差異ロジック、AI説明サービスを重ねます。
        このページだけは技術構成を明示し、本番化に耐える設計論点を見せます。
        """,
        """
        <div class="tech-architecture">
            <div class="tech-layer" style="--accent:#60a5fa">
                <em>Source systems</em>
                <b>業務システム群</b>
                <div class="tech-pill">ERP / GL actuals</div>
                <div class="tech-pill">EPM budget / forecast</div>
                <div class="tech-pill">Project EAC / PMO</div>
                <div class="tech-pill">Procurement / schedule</div>
                <div class="tech-pill">Master data</div>
            </div>
            <div class="tech-arrow">&rarr;</div>
            <div class="tech-layer" style="--accent:#ffb000">
                <em>Integration</em>
                <b>連携・品質管理</b>
                <div class="tech-pill">照合 / reconciliation</div>
                <div class="tech-pill">シナリオバージョン管理</div>
                <div class="tech-pill">案件ID・勘定マッピング</div>
                <div class="tech-pill">履歴・追跡性</div>
            </div>
            <div class="tech-arrow">&rarr;</div>
            <div class="tech-layer" style="--accent:#39c5bb">
                <em>FP&amp;A mart</em>
                <b>説明用データ層</b>
                <div class="tech-pill">実績 / 予算 / 見込</div>
                <div class="tech-pill">差異要因テーブル</div>
                <div class="tech-pill">案件リスクレイヤー</div>
                <div class="tech-pill">KPI semantic layer</div>
            </div>
            <div class="tech-arrow">&rarr;</div>
            <div class="tech-layer" style="--accent:#b794f4">
                <em>AI service</em>
                <b>AI説明サービス</b>
                <div class="tech-pill">根拠引用 / RAG</div>
                <div class="tech-pill">プロンプト管理</div>
                <div class="tech-pill">出力評価・監査ログ</div>
                <div class="tech-pill">権限・ガードレール</div>
            </div>
            <div class="tech-arrow">&rarr;</div>
            <div class="tech-layer" style="--accent:#ff647c">
                <em>Experience</em>
                <b>業務利用</b>
                <div class="tech-pill">FP&amp;A Cockpit</div>
                <div class="tech-pill">会議資料 / コメント</div>
                <div class="tech-pill">承認・通知</div>
                <div class="tech-pill">案件アクション管理</div>
            </div>
        </div>
        <div class="tech-footnote">
            <div>既存DWHやLakehouseがある場合は、その上にFP&amp;Aデータマートを設計します。</div>
            <div>AIは自由回答ではなく、根拠引用・出力評価・監査ログを持つ説明サービスとして扱います。</div>
            <div>KPI・差異ロジックはBI画面とAIコメントで共通化し、数字のズレを防ぎます。</div>
            <div>権限、承認、プロンプト変更、モデル変更は本番運用時の統制対象です。</div>
        </div>
        """,
    )


def render_proposal_approach_options() -> None:
    render_proposal_slide(
        6,
        "進め方の選択肢",
        "先に解消すべき不確実性から、アプローチを選ぶ",
        """
        進め方は作業メニューではありません。経営層の構想、実データでの効果、統制・承認運用のうち、
        どの不確実性を先に潰すかで選びます。
        """,
        """
        <div class="proposal-grid-3">
            <div class="option-card" style="--accent:#39c5bb">
                <em>Option 1</em>
                <b>構想具体化型</b>
                <span>経営層・関係者が、目指す業務像と利用イメージを共有できるかを先に確認する。</span>
                <div class="option-meta">
                    <div><strong>成果物:</strong> 画面イメージ、説明ストーリー、AIコメント例</div>
                    <div><strong>負荷:</strong> 低〜中</div>
                    <div><strong>向く状況:</strong> まず構想を具体化したい</div>
                </div>
            </div>
            <div class="option-card" style="--accent:#ffb000">
                <em>Option 2</em>
                <b>重点KPI実証型</b>
                <span>代表KPI・重点案件に絞り、実データで業務効果が出るかを確認する。</span>
                <div class="option-meta">
                    <div><strong>成果物:</strong> KPI連携PoC、差異分析、案件リスク分析</div>
                    <div><strong>負荷:</strong> 中</div>
                    <div><strong>向く状況:</strong> 実現性を確認したい</div>
                </div>
            </div>
            <div class="option-card" style="--accent:#ff647c">
                <em>Option 3</em>
                <b>本番運用設計型</b>
                <span>統制、承認、運用体制まで含めて、本番導入判断に進めるかを確認する。</span>
                <div class="option-meta">
                    <div><strong>成果物:</strong> 接続方針、品質ゲート、承認プロセス</div>
                    <div><strong>負荷:</strong> 高</div>
                    <div><strong>向く状況:</strong> 導入判断に進みたい</div>
                </div>
            </div>
        </div>
        <div class="proposal-close">
            初期段階では、構想具体化・効果検証・本番運用のどれを先に確かめるかを明確にすることが重要です。
        </div>
        """,
    )


def render_proposal_recommended_roadmap() -> None:
    render_proposal_slide(
        7,
        "推奨ロードマップ",
        "構想の具体化から始め、実証結果を本番化判断につなげる",
        """
        いきなり本番設計に入ると重く、PoCだけでは経営価値が曖昧になりやすい。
        まず経営会議での利用シーンと目指す業務像を具体化し、その後に重点KPIで有効性を確認し、実証結果をもとに本番化判断へ進めます。
        """,
        """
        <div class="roadmap">
            <div class="roadmap-phase" style="--accent:#39c5bb">
                <em>Phase 1 / 2-4 weeks</em>
                <b>構想具体化</b>
                <span>対象会議を一つ選び、画面イメージ、説明ストーリー、AIコメント例を作る。</span>
                <div class="roadmap-gate">判断ポイント: 経営層・FP&amp;A・ITが目指す業務像を共有できるか</div>
            </div>
            <div class="roadmap-phase" style="--accent:#ffb000">
                <em>Phase 2 / 6-8 weeks</em>
                <b>重点KPI実証</b>
                <span>営業利益、CF、重点案件などに絞り、実データに近い形で差異分析とコメント生成を検証する。</span>
                <div class="roadmap-gate">判断ポイント: 業務効果、データ接続難度、説明品質が見えるか</div>
            </div>
            <div class="roadmap-phase" style="--accent:#ff647c">
                <em>Phase 3 / 8-12 weeks</em>
                <b>本番運用設計</b>
                <span>品質ゲート、レビュー、承認、権限、監査ログ、運用体制を設計する。</span>
                <div class="roadmap-gate">判断ポイント: 月次FP&amp;Aプロセスへ組み込めるか</div>
            </div>
        </div>
        <div class="proposal-close">
            推奨は、構想具体化型 → 重点KPI実証型 → 本番運用設計型の段階導入です。
        </div>
        """,
    )


def render_proposal_assessment() -> None:
    render_proposal_slide(
        8,
        "アセスメント提案",
        "次フェーズで、PoCの対象と判断材料を具体化する",
        """
        次フェーズは、現状診断だけで終わらせません。短期に何を作り、何を検証し、
        本番化判断に何が必要かを決める場として設計します。
        """,
        """
        <div class="assessment-canvas">
            <div class="assessment-item">
                <em>Agenda 1</em>
                <b>対象会議</b>
                <span>月次経営会議、予実会議、見込更新会議のどこから変えるか。</span>
            </div>
            <div class="assessment-item">
                <em>Agenda 2</em>
                <b>対象KPI</b>
                <span>売上、営業利益、利益率、CFのどこを説明可能にするか。</span>
            </div>
            <div class="assessment-item">
                <em>Agenda 3</em>
                <b>利用可能データ</b>
                <span>ERP、EPM、案件EAC、調達、工程、マスタの所在と品質を確認する。</span>
            </div>
            <div class="assessment-item">
                <em>Agenda 4</em>
                <b>重点案件</b>
                <span>赤字化リスク、EAC悪化、調達影響が大きい案件群を選ぶ。</span>
            </div>
            <div class="assessment-item">
                <em>Agenda 5</em>
                <b>品質課題</b>
                <span>定義、バージョン、粒度、ID、元データまでの追跡性、承認のどこが詰まるかを見極める。</span>
            </div>
            <div class="assessment-item">
                <em>Agenda 6</em>
                <b>PoC範囲</b>
                <span>構想、接続範囲、成果物、判断基準を具体化する。</span>
            </div>
        </div>
        <div class="proposal-close">
            目的は「診断」ではなく、PoC対象と本番化判断に必要な材料を短期で揃えることです。
        </div>
        """,
    )


def presentation_snapshot_html(
    kpis: pd.DataFrame,
    risk: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    period: str = "FY2026 Total",
) -> str:
    metadata = metadata or {}
    base = aggregate_kpis(kpis, "Budget", period)
    actual = aggregate_kpis(kpis, "Actual", period)
    revenue_delta = actual["revenue_jpy_mn"] - base["revenue_jpy_mn"]
    op_delta = actual["operating_profit_jpy_mn"] - base["operating_profit_jpy_mn"]
    cf_delta = actual["cash_flow_jpy_mn"] - base["cash_flow_jpy_mn"]
    margin_delta = actual["op_margin_pct"] - base["op_margin_pct"]
    critical_count = int((risk["risk_level"] == "Critical").sum())
    high_count = int((risk["risk_level"] == "High").sum())
    loss_count = int(risk["loss_risk_flag"].sum())
    return presentation_metric_cards_html(
        [
            (
                "Data refresh",
                format_generated_at(metadata),
                f"{row_count_text(metadata)} / all figures are fictional",
            ),
            (
                "Executive readout",
                f"売上 {format_kpi_delta('Revenue', revenue_delta)}",
                f"営業利益 {format_kpi_delta('Operating Profit', op_delta)} / CF {format_kpi_delta('Cash Flow', cf_delta)}",
            ),
            (
                "Margin pressure",
                format_kpi_value("Operating Profit Margin", actual["op_margin_pct"]),
                f"vs Budget {format_kpi_delta('Operating Profit Margin', margin_delta)}",
            ),
            (
                "Risk queue",
                f"Critical {critical_count} / High {high_count}",
                f"赤字化リスク {loss_count}件を重点確認",
            ),
        ],
        columns=4,
    )


def render_client_pre_demo(kpis: pd.DataFrame, risk: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    render_header(
        "AI活用の前に整えるべきFP&Aデータ基盤",
        "経営KPIから案件リスク・打ち手まで説明できる状態をつくる",
    )

    slide = render_presentation_controls(
        "client_pre_demo",
        ["本日の論点", "経営会議の問い", "AI導入の落とし穴", "デモで見ること", "進め方"],
    )

    if slide == 0:
        render_presentation_slide(
            "Opening",
            "本日のテーマは、AIツールの紹介ではありません",
            """
            <div class="presentation-lead">
            CFO、経営企画、FP&amp;Aが必要としているのは、AIが文章を書くことそのものではありません。
            経営会議で「なぜ悪化したのか」「どの案件に手を打つべきか」を、同じ根拠データで説明できる状態です。
            </div>
            <div class="presentation-note">
            今日確認したいのは、AI活用の前に、経営KPI、差異要因、案件リスク、打ち手をつなぐFP&amp;Aデータ基盤をどう整えるかです。
            </div>
            """,
            "途中でCockpitを操作し、最後にデータ基盤アセスメントの論点へ戻します。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Management Question",
            "経営会議で問われるのは、数字ではなく原因と打ち手です",
            """
            <div class="presentation-lead">
            重工業のFP&amp;Aでは、売上が上振れていても、営業利益、利益率、キャッシュフローが悪化することがあります。
            このとき経営会議で必要なのは、グラフの説明ではなく、悪化要因と手を打つべき案件をすぐ示せることです。
            </div>
            """
            + presentation_snapshot_html(kpis, risk, metadata),
            "数値はすべて架空データです。実在企業の財務・案件情報は含みません。",
        )
    elif slide == 2:
        render_presentation_slide(
            "AI Trap",
            "データが分断されたままAIを入れても、説明責任は強くなりません",
            presentation_cards_html(
                [
                    (
                        "定義が揃わない",
                        "売上、営業利益、EAC、案件ステータスの定義が部門ごとに違うと、AIの出力を検証できません。",
                    ),
                    (
                        "版が揃わない",
                        "予算、前回見込、最新見込、実績の版が管理されていないと、差異説明が毎回作り直しになります。",
                    ),
                    (
                        "案件と財務がつながらない",
                        "EAC悪化、調達費、工程遅延が財務KPIへつながらないと、原因は見えても打ち手に落ちません。",
                    ),
                ],
                columns=3,
            ),
            "AIは最後の文章化を助けますが、経営説明の品質はその前のデータ基盤で決まります。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Demo Flow",
            "デモは、データがつながった場合の経営説明体験として見ます",
            """
            <div class="presentation-lead">
            この後、一度Cockpitをしっかり操作します。ただし見る対象は画面機能ではありません。
            経営KPIから差異要因、案件リスク、経営会議コメントまでが、同じ根拠でつながる流れを確認します。
            </div>
            """
            + presentation_flow_html(
                [
                    ("Dashboard", "売上、利益、利益率、CFの矛盾を捉える。"),
                    ("Variance", "差異を要因別に分解し、説明の主因を絞る。"),
                    ("Project Risk", "財務影響を案件リスクと打ち手へつなげる。"),
                    ("AI Commentary", "会議向けコメントを根拠付きの文章案にする。"),
                ],
                columns=4,
            ),
            "デモ後に、この体験を本番で成立させるためのデータ基盤論点へ戻ります。",
        )
    else:
        render_presentation_slide(
            "Assessment Lens",
            "本日のゴールは、データ基盤アセスメントの論点を揃えることです",
            """
            <div class="presentation-lead">
            デモ画面は完成製品の紹介ではなく、FP&amp;Aデータ基盤が整うと経営説明がどう変わるかを示す到達イメージです。
            操作後は、どの会議テーマ、KPI、データ、品質ゲートから棚卸しすべきかを確認します。
            </div>
            """
            + presentation_cards_html(
                [
                    ("会議テーマ", "月次経営会議、予実会議、見込更新会議のどこを最初に変えるか。"),
                    ("KPIと粒度", "売上、営業利益、利益率、CFをどの期間・セグメント・案件粒度で説明するか。"),
                    ("必要データ", "ERP、EPM、案件EAC、調達、工程、マスタのどこにギャップがあるか。"),
                ],
                columns=3,
            ),
            "この前提を置いたうえで、Cockpitの操作に入ります。",
        )


def render_internal_demo_guide() -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    render_header("デモ解説用 / Presenter Guide", "Internal talk track, not for client handout")

    st.markdown(
        """
        <div class="briefing-hero">
            <h2>このデモで伝える芯</h2>
            <p>
            クライアントに残したい印象は「AIでコメントが出る」ではなく、
            「経営会議で原因、影響、打ち手を説明するには、予実・見込・案件・調達・工程・マスタがつながる必要がある」です。
            Cockpitは到達イメージとして見せ、最後はデータ基盤アセスメントへ会話を戻します。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">ページ別の話法</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <table class="briefing-table">
            <thead>
                <tr>
                    <th>画面</th>
                    <th>一言メッセージ</th>
                    <th>話すポイント</th>
                    <th>避けたい言い方</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Client Preview</td>
                    <td>AI活用前のデータ基盤論点を置く</td>
                    <td>CFO・経営企画・FP&amp;A向けに、経営会議で説明すべき原因と打ち手から話を始める。</td>
                    <td>AIツールや画面機能の紹介から入る。</td>
                </tr>
                <tr>
                    <td>Dashboard</td>
                    <td>経営会議の問いを作る</td>
                    <td>売上は上振れ、利益とCFは悪化という矛盾を提示し、原因説明の必要性を作る。</td>
                    <td>グラフ仕様の説明に長く入る。</td>
                </tr>
                <tr>
                    <td>Variance Analysis</td>
                    <td>差異を要因へ分解し、説明を標準化する</td>
                    <td>Budget vs Actual、Latest Forecastなど比較軸を切り替え、ウォーターフォールで要因を確認する。</td>
                    <td>AIが原因を推測しているように見せる。</td>
                </tr>
                <tr>
                    <td>Project Risk</td>
                    <td>KPIから案件アクションへ落とす</td>
                    <td>財務影響をEAC悪化、調達、工程遅延などの案件論点へつなげる。</td>
                    <td>案件リスクを財務KPIと別物として扱う。</td>
                </tr>
                <tr>
                    <td>AI Commentary</td>
                    <td>AIは最後の文章化の出口として扱う</td>
                    <td>金額、要因、案件、推奨アクションが同じ根拠から文章化される点を見せる。</td>
                    <td>AIが最終判断まで自動化すると受け取られる言い方。</td>
                </tr>
                <tr>
                    <td>Data Foundation</td>
                    <td>本番化の主戦場はデータ基盤アセスメント</td>
                    <td>ERP、EPM、EAC、調達、工程、マスタ、品質ゲートを棚卸し対象として説明する。</td>
                    <td>LLM連携や画面構築だけを次アクションにする。</td>
                </tr>
                <tr>
                    <td>Client Follow-up</td>
                    <td>次はデータ基盤アセスメントを始める</td>
                    <td>会議テーマ、KPI定義、必要データ、版管理、粒度、リネージの棚卸しへ落とす。</td>
                    <td>「PoCしましょう」だけで終える。</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">5分版トークトラック</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="briefing-flow">
            <div class="briefing-step">
                <b>0:00 Opening</b>
                <span>「本日はAIツールではなく、経営説明に必要なFP&amp;Aデータ基盤の話です。」</span>
            </div>
            <div class="briefing-step">
                <b>0:45 Problem</b>
                <span>売上は上振れ、利益とCFは悪化という経営会議の問いを置く。</span>
            </div>
            <div class="briefing-step">
                <b>1:30 Demo</b>
                <span>Dashboard、Variance、Risk、AI Commentaryを経営説明の流れとして操作する。</span>
            </div>
            <div class="briefing-step">
                <b>3:45 Foundation</b>
                <span>今の体験は、ERP、EPM、案件EAC、調達、工程、マスタがつながることが前提だと戻す。</span>
            </div>
            <div class="briefing-step">
                <b>4:30 Close</b>
                <span>「次はデータ基盤アセスメントから始めましょう」で締める。</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">想定Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <table class="briefing-table">
            <thead>
                <tr>
                    <th>質問</th>
                    <th>答え方</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>これは実データですか</td>
                    <td>いいえ、完全な架空データです。実装時はERP、EPM、案件EAC、調達、工程、マスタを接続します。</td>
                </tr>
                <tr>
                    <td>これはAI製品のデモですか</td>
                    <td>主目的は製品紹介ではありません。経営説明に必要なデータ接続と品質ゲートを具体化するための到達イメージです。</td>
                </tr>
                <tr>
                    <td>最初に何から始めるべきですか</td>
                    <td>代表会議テーマを1つ選び、KPI定義、比較軸、必要データ、品質ゲートを棚卸しするのが現実的です。</td>
                </tr>
                <tr>
                    <td>どのくらいでPoCできますか</td>
                    <td>代表セグメント・代表KPIに絞れば短期PoCは可能ですが、実運用化にはデータ定義と版管理の整備が必要です。</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_client_post_demo() -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    render_header("データ基盤アセスメント / Client Follow-up", "デモ体験を自社適用の論点へ戻す")

    slide = render_presentation_controls(
        "client_post_demo",
        ["回収メッセージ", "主な示唆", "アセスメント観点", "次の進め方", "次回アジェンダ"],
    )

    if slide == 0:
        render_presentation_slide(
            "Client Follow-up",
            "今見た体験は、AI単体では成立しません",
            """
            <div class="presentation-lead">
            Cockpitで見た「KPIから差異要因、案件リスク、会議コメントまでつながる」体験は、
            ERP、EPM、案件EAC、調達、工程、マスタが同じ定義・版・粒度でつながって初めて成立します。
            </div>
            <div class="presentation-note">
            次に行うべきことはAIツール選定ではなく、自社のFP&amp;Aデータ基盤が経営説明に耐えられるかを確認するアセスメントです。
            </div>
            """,
            "ここから自社データで何を確認すべきかに切り替えます。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Takeaways",
            "主な示唆",
            presentation_cards_html(
                [
                    (
                        "1. 経営説明はデータのつながりで決まる",
                        "売上、利益、CF、案件リスクが同じ根拠でつながると、会議で説明すべき原因と打ち手が見えやすくなります。",
                    ),
                    (
                        "2. 重工業では案件粒度が重要",
                        "売上・利益・CFの変化は、案件EAC、設計変更、調達、工程遅延とつながっています。全社KPIだけでは打ち手に届きません。",
                    ),
                    (
                        "3. AI導入はアセスメントから始める",
                        "LLM連携より先に、データソース、KPI定義、シナリオ版管理、案件マスタ、品質ゲートを棚卸しする必要があります。",
                    ),
                ],
                columns=3,
            ),
            "この3点が、データ基盤アセスメントと後続PoCの優先順位を決めます。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Assessment",
            "データ基盤アセスメントで確認する観点",
            """
            <div class="presentation-lead">
            最初から全社データをすべて調べる必要はありません。経営説明の改善効果が大きい会議テーマを一つ選び、
            その説明に必要なKPI、データ、品質ゲートを棚卸しします。
            </div>
            """
            + presentation_cards_html(
                [
                    ("会議テーマとKPI", "月次経営会議、予実会議、見込更新会議のどこを対象にし、どのKPIを説明可能にするか。"),
                    ("根拠データとマスタ", "ERP、EPM、案件EAC、調達、工程、セグメント・案件・勘定マスタの所在と責任部門。"),
                    ("品質ゲート", "照合、版管理、粒度合わせ、ID統合、リネージ、承認プロセスをどこまで持てているか。"),
                ],
                columns=3,
            ),
            "詳細なチェックリストは、この3点が決まってから作る方が会話が進みます。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Next Steps",
            "データ基盤アセスメントの進め方",
            """
            <div class="presentation-lead">
            進め方は、最初から大きな基盤を作るのではなく、説明責任が高い会議テーマから小さく始めます。
            そのテーマで必要なデータとKPIを固め、ギャップを見える化し、短期PoCの範囲へ落とします。
            </div>
            """
            + presentation_flow_html(
                [
                    ("対象を決める", "最初に変えたい会議とKPIを決めます。"),
                    ("根拠を棚卸し", "必要データ、比較軸、案件粒度、責任部門を確認します。"),
                    ("ギャップを整理", "定義、版、粒度、マスタ、リネージの不足を明確にします。"),
                    ("PoCへ落とす", "代表セグメントと重点案件に絞って検証範囲を決めます。"),
                ],
                columns=4,
            ),
            "最初から全社展開を狙わず、説明責任が高い会議テーマに絞って検証します。",
        )
    else:
        render_presentation_slide(
            "Recommended Agenda",
            "推奨する次回アジェンダ",
            presentation_cards_html(
                [
                    ("会議テーマ", "経営会議、予実会議、見込更新会議のどこを最初の対象にするかを決めます。"),
                    ("データとKPI", "必要データの所在、粒度、比較軸、責任部門、品質ゲートを確認します。"),
                    ("アセスメント範囲", "どのセグメント、案件、KPIを対象に棚卸しし、短期PoCへつなげるかを決めます。"),
                ],
                columns=3,
            ),
            "AIコメント生成より前に、整えるべきFP&Aデータ基盤の範囲を明確にします。",
        )


@st.cache_data(show_spinner="デモデータを読み込み中...")
def load_data() -> dict[str, Any]:
    required = {
        "dim_projects": DATA_DIR / "dim_projects.parquet",
        "fact_finance": DATA_DIR / "fact_finance.parquet",
        "fact_variance_drivers": DATA_DIR / "fact_variance_drivers.parquet",
        "project_risk": DATA_DIR / "project_risk.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        return {"missing": missing}

    data = {name: pd.read_parquet(path) for name, path in required.items()}
    metadata_path = DATA_DIR / "demo_metadata.json"
    if metadata_path.exists():
        data["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        data["metadata"] = {}
    return data


@st.cache_data(show_spinner=False)
def build_project_month_kpis(finance: pd.DataFrame) -> pd.DataFrame:
    dims = [
        "company",
        "fiscal_year",
        "period",
        "period_start",
        "fiscal_month",
        "scenario",
        "scenario_ja",
        "segment_en",
        "segment_ja",
        "segment_code",
        "business_unit",
        "project_id",
        "project_name",
    ]
    revenue = (
        finance.loc[finance["account"] == "Revenue"]
        .groupby(dims, observed=True)["amount_jpy_mn"]
        .sum()
        .rename("revenue_jpy_mn")
    )
    operating_profit = (
        finance.loc[finance["account"] != "Cash Flow"]
        .groupby(dims, observed=True)["amount_jpy_mn"]
        .sum()
        .rename("operating_profit_jpy_mn")
    )
    cash_flow = (
        finance.loc[finance["account"] == "Cash Flow"]
        .groupby(dims, observed=True)["amount_jpy_mn"]
        .sum()
        .rename("cash_flow_jpy_mn")
    )
    kpis = pd.concat([revenue, operating_profit, cash_flow], axis=1).fillna(0).reset_index()
    kpis["op_margin_pct"] = np.where(
        kpis["revenue_jpy_mn"].abs() > 0,
        kpis["operating_profit_jpy_mn"] / kpis["revenue_jpy_mn"] * 100,
        np.nan,
    )
    return kpis


def format_amount(value_jpy_mn: float) -> str:
    sign = "-" if value_jpy_mn < 0 else ""
    value_bn = abs(value_jpy_mn) / 1_000
    if abs(value_bn) >= 1_000:
        return f"{sign}¥{value_bn:,.0f}bn"
    return f"{sign}¥{value_bn:,.1f}bn"


def format_signed_amount(value_jpy_mn: float) -> str:
    sign = "+" if value_jpy_mn >= 0 else ""
    return f"{sign}{format_amount(value_jpy_mn)}"


def format_kpi_value(kpi: str, value: float) -> str:
    if KPI_META[kpi]["unit"] == "margin":
        return f"{value:,.1f}%"
    return format_amount(value)


def format_kpi_delta(kpi: str, value: float) -> str:
    if KPI_META[kpi]["unit"] == "margin":
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:,.1f}pt"
    return format_signed_amount(value)


def pct_change(current: float, base: float) -> float:
    if abs(base) < 1e-9:
        return 0.0
    return (current - base) / abs(base) * 100


def period_options(kpis: pd.DataFrame) -> list[str]:
    months = sorted(kpis["period"].dropna().unique().tolist())
    return [
        "FY2026 Total",
        "Q1 Apr-Jun",
        "Q2 Jul-Sep",
        "Q3 Oct-Dec",
        "Q4 Jan-Mar",
    ] + months


def apply_period_filter(df: pd.DataFrame, selected_period: str) -> pd.DataFrame:
    if selected_period == "FY2026 Total":
        return df
    quarter_months = {
        "Q1 Apr-Jun": [1, 2, 3],
        "Q2 Jul-Sep": [4, 5, 6],
        "Q3 Oct-Dec": [7, 8, 9],
        "Q4 Jan-Mar": [10, 11, 12],
    }
    if selected_period in quarter_months:
        return df[df["fiscal_month"].isin(quarter_months[selected_period])]
    return df[df["period"] == selected_period]


def segment_options(kpis: pd.DataFrame) -> list[str]:
    segments = kpis[["segment_en", "segment_ja"]].drop_duplicates().sort_values("segment_en")
    return ["全社 / Total"] + [f"{row.segment_ja} / {row.segment_en}" for row in segments.itertuples()]


def parse_segment(option: str) -> str | None:
    if option.startswith("全社"):
        return None
    return option.split(" / ", 1)[1]


def filter_kpis(kpis: pd.DataFrame, period: str, segment_option: str | None = None) -> pd.DataFrame:
    scoped = apply_period_filter(kpis, period)
    segment_en = parse_segment(segment_option) if segment_option else None
    if segment_en:
        scoped = scoped[scoped["segment_en"] == segment_en]
    return scoped


def filter_drivers(drivers: pd.DataFrame, period: str, comparison: str, segment_option: str | None = None) -> pd.DataFrame:
    scoped = apply_period_filter(drivers, period)
    scoped = scoped[scoped["comparison_type"] == comparison]
    segment_en = parse_segment(segment_option) if segment_option else None
    if segment_en:
        scoped = scoped[scoped["segment_en"] == segment_en]
    return scoped


def aggregate_kpis(kpis: pd.DataFrame, scenario: str, period: str, segment_option: str | None = None) -> dict[str, float]:
    scoped = filter_kpis(kpis, period, segment_option)
    scoped = scoped[scoped["scenario"] == scenario]
    revenue = float(scoped["revenue_jpy_mn"].sum())
    op = float(scoped["operating_profit_jpy_mn"].sum())
    cf = float(scoped["cash_flow_jpy_mn"].sum())
    margin = op / revenue * 100 if abs(revenue) > 1e-9 else 0.0
    return {
        "revenue_jpy_mn": revenue,
        "operating_profit_jpy_mn": op,
        "cash_flow_jpy_mn": cf,
        "op_margin_pct": margin,
    }


def metric_card(label: str, value: str, delta: str, delta_pct_text: str = "", favorable: bool | None = None) -> None:
    if favorable is True:
        cls = "delta-good"
    elif favorable is False:
        cls = "delta-bad"
    else:
        cls = "delta-neutral"
    pct_html = f" <span class='small-note'>({escape(delta_pct_text)})</span>" if delta_pct_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-delta {cls}">{escape(delta)}{pct_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_generated_at(metadata: dict[str, Any]) -> str:
    generated_at = metadata.get("generated_at")
    if not generated_at:
        return "未生成"
    try:
        return datetime.fromisoformat(str(generated_at)).strftime("%Y-%m-%d %H:%M JST")
    except ValueError:
        return str(generated_at)


def row_count_text(metadata: dict[str, Any]) -> str:
    total = metadata.get("row_counts", {}).get("total")
    if total is None:
        return "行数未取得"
    return f"{int(total):,} rows"


def render_latest_snapshot(
    kpis: pd.DataFrame,
    risk: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    period: str = "FY2026 Total",
) -> None:
    metadata = metadata or {}
    base = aggregate_kpis(kpis, "Budget", period)
    actual = aggregate_kpis(kpis, "Actual", period)
    revenue_delta = actual["revenue_jpy_mn"] - base["revenue_jpy_mn"]
    op_delta = actual["operating_profit_jpy_mn"] - base["operating_profit_jpy_mn"]
    cf_delta = actual["cash_flow_jpy_mn"] - base["cash_flow_jpy_mn"]
    margin_delta = actual["op_margin_pct"] - base["op_margin_pct"]
    critical_count = int((risk["risk_level"] == "Critical").sum())
    high_count = int((risk["risk_level"] == "High").sum())
    loss_count = int(risk["loss_risk_flag"].sum())

    cards = [
        (
            "Data refresh",
            format_generated_at(metadata),
            f"{row_count_text(metadata)} / all figures are fictional",
        ),
        (
            "Executive readout",
            f"売上 {format_kpi_delta('Revenue', revenue_delta)}",
            f"営業利益 {format_kpi_delta('Operating Profit', op_delta)} / CF {format_kpi_delta('Cash Flow', cf_delta)}",
        ),
        (
            "Margin pressure",
            format_kpi_value("Operating Profit Margin", actual["op_margin_pct"]),
            f"vs Budget {format_kpi_delta('Operating Profit Margin', margin_delta)}",
        ),
        (
            "Risk queue",
            f"Critical {critical_count} / High {high_count}",
            f"赤字化リスク {loss_count}件を重点確認",
        ),
    ]
    card_html = "".join(
        (
            '<div class="snapshot-cell">'
            f"<b>{escape(label)}</b>"
            f"<strong>{escape(value)}</strong>"
            f"<span>{escape(note)}</span>"
            "</div>"
        )
        for label, value, note in cards
    )
    st.markdown(f'<div class="snapshot-grid">{card_html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,16,22,0.65)",
        font={"color": "#edf6f9"},
        margin={"l": 20, "r": 20, "t": 44, "b": 28},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.28)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.28)")
    return fig


def kpi_variance(base: dict[str, float], current: dict[str, float], kpi: str) -> tuple[float, float, float]:
    key = KPI_META[kpi]["key"]
    base_value = float(base[key])
    current_value = float(current[key])
    variance = current_value - base_value
    return base_value, current_value, variance


def format_heatmap_amount(value_jpy_mn: float) -> str:
    return f"{value_jpy_mn / 1_000:+,.1f}bn"


def normalize_for_heatmap(values: pd.Series) -> pd.Series:
    max_abs = float(values.abs().max()) if not values.empty else 0.0
    if max_abs < 1e-9:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values / max_abs).clip(-1.0, 1.0)


def render_segment_kpi_heatmap(kpis: pd.DataFrame, risk: pd.DataFrame, selected_period: str) -> go.Figure:
    scoped = apply_period_filter(kpis, selected_period)
    segment_kpis = (
        scoped[scoped["scenario"].isin(["Budget", "Actual"])]
        .groupby(["segment_code", "segment_en", "segment_ja", "scenario"], observed=True)[
            ["revenue_jpy_mn", "operating_profit_jpy_mn", "cash_flow_jpy_mn"]
        ]
        .sum()
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for (segment_code, segment_en, segment_ja), segment_rows in segment_kpis.groupby(
        ["segment_code", "segment_en", "segment_ja"],
        observed=True,
        sort=True,
    ):
        budget_rows = segment_rows[segment_rows["scenario"] == "Budget"]
        actual_rows = segment_rows[segment_rows["scenario"] == "Actual"]
        if budget_rows.empty or actual_rows.empty:
            continue

        budget = budget_rows.iloc[0]
        actual = actual_rows.iloc[0]
        budget_revenue = float(budget["revenue_jpy_mn"])
        actual_revenue = float(actual["revenue_jpy_mn"])
        budget_op = float(budget["operating_profit_jpy_mn"])
        actual_op = float(actual["operating_profit_jpy_mn"])
        budget_cf = float(budget["cash_flow_jpy_mn"])
        actual_cf = float(actual["cash_flow_jpy_mn"])
        budget_margin = budget_op / budget_revenue * 100 if abs(budget_revenue) > 1e-9 else 0.0
        actual_margin = actual_op / actual_revenue * 100 if abs(actual_revenue) > 1e-9 else 0.0
        scoped_risk = risk[risk["segment_en"] == segment_en]

        rows.append(
            {
                "segment": f"{segment_code} | {segment_ja}",
                "segment_en": segment_en,
                "revenue_delta": actual_revenue - budget_revenue,
                "revenue_pct": pct_change(actual_revenue, budget_revenue),
                "op_delta": actual_op - budget_op,
                "op_pct": pct_change(actual_op, budget_op),
                "margin_delta": actual_margin - budget_margin,
                "actual_margin": actual_margin,
                "cf_delta": actual_cf - budget_cf,
                "cf_pct": pct_change(actual_cf, budget_cf),
                "loss_risk_count": int(scoped_risk["loss_risk_flag"].sum()),
                "critical_count": int((scoped_risk["risk_level"] == "Critical").sum()),
                "high_count": int((scoped_risk["risk_level"] == "High").sum()),
            }
        )

    heatmap = pd.DataFrame(rows).sort_values("segment_en")
    columns = ["売上", "営業利益", "利益率", "CF", "赤字リスク"]
    if heatmap.empty:
        fig = go.Figure()
        fig.update_layout(title="セグメントKPIヒートマップ / 予算差異・赤字化リスク集中度")
        return style_fig(fig, 380)

    heatmap["revenue_z"] = normalize_for_heatmap(heatmap["revenue_delta"])
    heatmap["op_z"] = normalize_for_heatmap(heatmap["op_delta"])
    heatmap["margin_z"] = normalize_for_heatmap(heatmap["margin_delta"])
    heatmap["cf_z"] = normalize_for_heatmap(heatmap["cf_delta"])
    max_loss_count = int(heatmap["loss_risk_count"].max())
    heatmap["loss_risk_z"] = (
        -(heatmap["loss_risk_count"] / max_loss_count).clip(0.0, 1.0)
        if max_loss_count > 0
        else 0.0
    )

    z_values: list[list[float]] = []
    text_values: list[list[str]] = []
    for row in heatmap.itertuples(index=False):
        z_values.append(
            [
                float(row.revenue_z),
                float(row.op_z),
                float(row.margin_z),
                float(row.cf_z),
                float(row.loss_risk_z),
            ]
        )
        text_values.append(
            [
                f"{format_heatmap_amount(row.revenue_delta)}<br>{row.revenue_pct:+.1f}%",
                f"{format_heatmap_amount(row.op_delta)}<br>{row.op_pct:+.1f}%",
                f"{row.margin_delta:+.1f}pt<br>{row.actual_margin:.1f}%",
                f"{format_heatmap_amount(row.cf_delta)}<br>{row.cf_pct:+.1f}%",
                f"赤字 {row.loss_risk_count}件<br>Critical {row.critical_count}件",
            ]
        )

    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=columns,
            y=heatmap["segment"].tolist(),
            text=text_values,
            texttemplate="%{text}",
            textfont={"color": "#edf6f9", "size": 12},
            colorscale=[
                [0.0, "#7a2330"],
                [0.42, "#2a3640"],
                [0.50, "#3a4a55"],
                [0.58, "#274b42"],
                [1.0, "#3f7f55"],
            ],
            zmin=-1,
            zmax=1,
            xgap=3,
            ygap=3,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(title="セグメントKPIヒートマップ / 予算差異・赤字化リスク集中度")
    fig.update_xaxes(side="top", tickfont={"size": 12})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11})
    fig = style_fig(fig, 420)
    fig.update_layout(margin={"l": 165, "r": 20, "t": 60, "b": 20})
    return fig


def render_dashboard(
    kpis: pd.DataFrame,
    drivers: pd.DataFrame,
    risk: pd.DataFrame,
    show_guide: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    render_header("Dashboard", "Command Center | FY2026")
    if show_guide:
        render_page_guide(
            "この画面の見方 / 全社の異常値を最初に掴む",
            "まず、売上・営業利益・営業利益率・キャッシュフローの方向がそろっているかを確認します。重工業では売上が伸びても、EAC悪化、外注費、納期遅延、運転資本で利益とCFが悪化することがあります。",
            [
                "KPIカードでは、ActualがBudgetに対して上振れか下振れかを確認します。",
                "月次推移では、売上と営業利益の動きが同じ方向か、乖離しているかを見ます。",
                "セグメント別差異とTop Driversで、どの事業が全社悪化を作っているかを特定します。",
            ],
            "このデモの結論: 全社売上は上振れしている一方、営業利益・利益率・キャッシュフローは悪化しています。",
        )

    col_filter, col_status = st.columns([1.1, 2.9])
    with col_filter:
        selected_period = st.selectbox("分析期間", period_options(kpis), index=0)
    with col_status:
        st.markdown(
            f"""
            <div class="status-strip">
                <div class="status-cell"><b>Scope</b>Consolidated FY2026</div>
                <div class="status-cell"><b>Comparison</b>Budget vs Actual</div>
                <div class="status-cell"><b>Data refresh</b>{escape(format_generated_at(metadata or {}))}</div>
                <div class="status-cell"><b>Display</b>JPY bn / fictional facts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    base = aggregate_kpis(kpis, "Budget", selected_period)
    actual = aggregate_kpis(kpis, "Actual", selected_period)

    render_latest_snapshot(kpis, risk, metadata or {}, selected_period)

    cols = st.columns(4)
    for col, kpi in zip(cols, KPI_META.keys()):
        base_value, current_value, variance = kpi_variance(base, actual, kpi)
        favorable = variance >= 0
        if kpi == "Operating Profit Margin":
            pct_text = ""
        else:
            pct_text = f"{pct_change(current_value, base_value):+.1f}%"
        with col:
            metric_card(
                f"{KPI_META[kpi]['ja']} / {kpi}",
                format_kpi_value(kpi, current_value),
                f"vs Budget {format_kpi_delta(kpi, variance)}",
                pct_text,
                favorable=favorable,
            )

    if show_guide:
        st.markdown('<div class="section-label">セグメント別ストーリー / Business Story</div>', unsafe_allow_html=True)
        render_segment_story_cards()

    trend_df = apply_period_filter(kpis, selected_period)
    monthly = (
        trend_df[trend_df["scenario"].isin(["Budget", "Actual"])]
        .groupby(["period", "scenario"], observed=True)[["revenue_jpy_mn", "operating_profit_jpy_mn", "cash_flow_jpy_mn"]]
        .sum()
        .reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {"Actual": "#39c5bb", "Budget": "#9fb2c3"}
    for scenario in ["Actual", "Budget"]:
        scenario_df = monthly[monthly["scenario"] == scenario]
        fig.add_trace(
            go.Scatter(
                x=scenario_df["period"],
                y=scenario_df["revenue_jpy_mn"] / 1_000,
                mode="lines+markers",
                name=f"{SCENARIO_JA[scenario]} Revenue",
                line={"color": colors[scenario], "width": 3 if scenario == "Actual" else 2},
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=scenario_df["period"],
                y=scenario_df["operating_profit_jpy_mn"] / 1_000,
                mode="lines+markers",
                name=f"{SCENARIO_JA[scenario]} OP",
                line={"color": colors[scenario], "width": 2, "dash": "dash"},
            ),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="Revenue (JPY bn)", secondary_y=False)
    fig.update_yaxes(title_text="Operating Profit (JPY bn)", secondary_y=True)
    fig.update_layout(title="月次推移 / Monthly Trend")
    st.plotly_chart(style_fig(fig, 430), width="stretch")

    left, right = st.columns([1.25, 1.0])
    with left:
        st.plotly_chart(render_segment_kpi_heatmap(kpis, risk, selected_period), width="stretch")

    with right:
        drv = filter_drivers(drivers, selected_period, "Budget vs Actual")
        top = (
            drv.groupby(["variance_driver_ja", "variance_driver"], observed=True)[["impact_op_jpy_mn"]]
            .sum()
            .reset_index()
        )
        top["abs_impact"] = top["impact_op_jpy_mn"].abs()
        top = top.nlargest(8, "abs_impact").sort_values("impact_op_jpy_mn")
        fig = px.bar(
            top,
            x="impact_op_jpy_mn",
            y="variance_driver_ja",
            orientation="h",
            color="impact_op_jpy_mn",
            color_continuous_scale=["#ff647c", "#9fb2c3", "#76d275"],
            title="Top Variance Drivers / 営業利益影響",
            labels={"impact_op_jpy_mn": "OP Impact (JPY mn)", "variance_driver_ja": ""},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    risk_top = risk.sort_values("risk_score", ascending=False).head(6)
    st.markdown('<div class="section-label">Project Risk Snapshot / 重点モニタリング案件</div>', unsafe_allow_html=True)
    st.dataframe(
        risk_top[
            [
                "risk_level",
                "risk_score",
                "segment_ja",
                "project_id",
                "project_name",
                "eac_deterioration_jpy_mn",
                "forecast_margin_pct",
                "primary_driver_ja",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "eac_deterioration_jpy_mn": st.column_config.NumberColumn("EAC悪化 (JPY mn)", format="%.0f"),
            "forecast_margin_pct": st.column_config.NumberColumn("見込利益率", format="%.1f%%"),
        },
    )


def summarize_driver_impacts(drivers: pd.DataFrame, kpi: str, base_revenue: float) -> pd.DataFrame:
    driver_col = KPI_META[kpi]["driver_col"]
    summary = (
        drivers.groupby(["variance_driver", "variance_driver_ja"], observed=True)[driver_col]
        .sum()
        .reset_index()
        .rename(columns={driver_col: "impact"})
    )
    if KPI_META[kpi]["unit"] == "margin":
        denominator = base_revenue if abs(base_revenue) > 1e-9 else 1.0
        summary["impact_display"] = summary["impact"] / denominator * 100
    else:
        summary["impact_display"] = summary["impact"]
    summary["abs_impact"] = summary["impact_display"].abs()
    return summary.sort_values("abs_impact", ascending=False)


def render_waterfall(base_value: float, current_value: float, driver_summary: pd.DataFrame, kpi: str) -> go.Figure:
    top = driver_summary.head(6).copy()
    driver_sum = float(top["impact_display"].sum()) if not top.empty else 0.0
    residual = current_value - base_value - driver_sum
    x = ["Base"] + top["variance_driver_ja"].tolist() + ["Residual / Mix", "Current"]
    y = [base_value] + top["impact_display"].tolist() + [residual, 0]
    measure = ["absolute"] + ["relative"] * len(top) + ["relative", "total"]
    fig = go.Figure(
        go.Waterfall(
            x=x,
            y=y,
            measure=measure,
            connector={"line": {"color": "rgba(159,178,195,0.45)"}},
            increasing={"marker": {"color": "#76d275"}},
            decreasing={"marker": {"color": "#ff647c"}},
            totals={"marker": {"color": "#39c5bb"}},
        )
    )
    unit_label = "pt" if KPI_META[kpi]["unit"] == "margin" else "JPY mn"
    fig.update_layout(title=f"ウォーターフォール / {KPI_META[kpi]['ja']} ({unit_label})")
    return style_fig(fig, 430)


def build_context(
    kpis: pd.DataFrame,
    drivers: pd.DataFrame,
    risk: pd.DataFrame,
    period: str,
    segment_option: str,
    comparison: str,
    kpi: str,
) -> dict[str, Any]:
    base_scenario, current_scenario = COMPARISON_MAP[comparison]
    base = aggregate_kpis(kpis, base_scenario, period, segment_option)
    current = aggregate_kpis(kpis, current_scenario, period, segment_option)
    key = KPI_META[kpi]["key"]
    variance = current[key] - base[key]
    scoped_drivers = filter_drivers(drivers, period, comparison, segment_option)
    driver_summary = summarize_driver_impacts(scoped_drivers, kpi, base["revenue_jpy_mn"])
    top_drivers = driver_summary.head(5).to_dict("records")

    segment_en = parse_segment(segment_option)
    scoped_risk = risk if segment_en is None else risk[risk["segment_en"] == segment_en]
    top_projects = scoped_risk.sort_values("risk_score", ascending=False).head(5).to_dict("records")

    return {
        "period": period,
        "segment": segment_option,
        "comparison": comparison,
        "base_scenario": base_scenario,
        "current_scenario": current_scenario,
        "kpi": kpi,
        "kpi_ja": KPI_META[kpi]["ja"],
        "base_value": base[key],
        "current_value": current[key],
        "variance": variance,
        "variance_pct": pct_change(current[key], base[key]) if KPI_META[kpi]["unit"] != "margin" else None,
        "base_all": base,
        "current_all": current,
        "top_drivers": top_drivers,
        "top_projects": top_projects,
        "critical_count": int((scoped_risk["risk_level"] == "Critical").sum()),
        "high_count": int((scoped_risk["risk_level"] == "High").sum()),
    }


def driver_phrase(driver: dict[str, Any], kpi: str) -> str:
    amount = float(driver.get("impact_display", 0.0))
    name = str(driver.get("variance_driver_ja", driver.get("variance_driver", "")))
    if KPI_META[kpi]["unit"] == "margin":
        return f"{name}（{amount:+.1f}pt）"
    return f"{name}（{format_signed_amount(amount)}）"


def context_signature(task: str, context: dict[str, Any]) -> str:
    payload = {
        "task": task,
        "period": context["period"],
        "segment": context["segment"],
        "comparison": context["comparison"],
        "kpi": context["kpi"],
        "variance": round(float(context["variance"]), 3),
        "drivers": [d.get("variance_driver") for d in context["top_drivers"][:3]],
        "projects": [p.get("project_id") for p in context["top_projects"][:3]],
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def variant_pick(task: str, context: dict[str, Any], options: list[str]) -> str:
    signature = context_signature(task, context)
    return options[int(signature[:8], 16) % len(options)]


def segment_focus(segment: str) -> str:
    if "Aerospace & Defense" in segment:
        return "航空・防衛では、外注費、為替、長納期部品の3点が利益率とキャッシュフローを圧迫しやすい構造です。"
    if "Energy Systems" in segment:
        return "エネルギーシステムでは、高採算案件の前倒しが売上と営業利益を押し上げる一方、継続確度の確認が必要です。"
    if "Marine & Offshore" in segment:
        return "船舶・海洋では、EAC悪化、設計変更、納期遅延が同時に発生し、赤字化リスク案件へ直結しやすい点が論点です。"
    if "Industrial Machinery & Robotics" in segment:
        return "産業機械・ロボットでは、需要鈍化と固定費吸収不足により、売上下振れが利益率悪化へ波及しやすい状況です。"
    return "全社では、セグメント間のプラス・マイナスが相殺されるため、売上、利益、CFを分けて見る必要があります。"


def kpi_focus(kpi: str) -> str:
    if kpi == "Revenue":
        return "売上は計上時期と案件前倒しの影響を受けるため、利益・CFと同じ方向に動くとは限りません。"
    if kpi == "Operating Profit":
        return "営業利益は、売上増減だけでなく材料費、外注費、EAC悪化、固定費吸収の影響を強く受けます。"
    if kpi == "Operating Profit Margin":
        return "営業利益率は、売上規模よりも案件ミックスとコスト上振れの影響を端的に示します。"
    return "キャッシュフローは、利益悪化に加えて納期遅延、検収遅れ、運転資本負担の影響を受けます。"


def impact_tone(value: float, kpi: str) -> str:
    unit = "pt" if KPI_META[kpi]["unit"] == "margin" else "JPY mn"
    if value < 0:
        return f"不利差異。影響は{value:,.1f}{unit}で、是正アクションの優先度が高い。"
    return f"有利差異。影響は{value:,.1f}{unit}で、継続確度と再現性の確認が必要。"


def rule_based_commentary(task: str, context: dict[str, Any]) -> str:
    kpi = context["kpi"]
    delta = float(context["variance"])
    direction = "上振れ" if delta >= 0 else "下振れ"
    if kpi in {"Operating Profit", "Operating Profit Margin", "Cash Flow"} and delta < 0:
        direction = "悪化"

    value_text = format_kpi_delta(kpi, delta)
    top_drivers = context["top_drivers"]
    driver_text = "、".join(driver_phrase(driver, kpi) for driver in top_drivers[:3]) or "主要差異は限定的"
    lens = variant_pick(
        task,
        context,
        ["収益性優先", "キャッシュ創出優先", "案件リスク優先", "見込精度優先", "実行アクション優先"],
    )
    opening = variant_pick(
        task + "-opening",
        context,
        [
            "今回の分析では",
            "当該条件で見ると",
            "経営会議向けには",
            "FP&Aレビューでは",
            "事業部長レビューの観点では",
        ],
    )
    period_note = f"対象期間は{context['period']}、比較軸は{context['comparison']}です。"
    parameter_note = (
        f"分析条件: {context['segment']} / {context['kpi_ja']} / {context['comparison']} / {context['period']} "
        f"（観点: {lens}）"
    )

    project_lines = []
    for project in context["top_projects"][:3]:
        project_lines.append(
            f"{project['project_id']} {project['project_name']}（{project['segment_ja']}、"
            f"Risk {project['risk_score']}、主因: {project['primary_driver_ja']}）"
        )
    projects = "；".join(project_lines) if project_lines else "該当する重点案件なし"

    if task == "rank":
        lines = [
            f"{idx + 1}. {driver_phrase(driver, kpi)}: {impact_tone(float(driver.get('impact_display', 0.0)), kpi)}"
            for idx, driver in enumerate(top_drivers[:5])
        ]
        closing = variant_pick(
            task + "-closing",
            context,
            [
                "次回レビューでは、上位2要因の金額根拠とアクション進捗を確認してください。",
                "金額影響の大きい順に、責任部署と回復可能性を確認する進め方が妥当です。",
                "ランキング上位は、経営会議で論点化し、期限付きの確認事項に落とすべきです。",
            ],
        )
        return f"{parameter_note}\n\n重要要因ランキング:\n" + "\n".join(lines) + f"\n\n{closing}"

    if task == "recommend":
        action_angle = variant_pick(
            task + "-action",
            context,
            [
                "EAC再見積、変更契約、価格転嫁、検収時期の4点を確認してください。",
                "案件オーナー、調達責任者、工程管理、FP&Aが同じ前提で回復計画を確認してください。",
                "単月差異として処理せず、次回見込に織り込むべき構造差異かを確認してください。",
            ],
        )
        return (
            f"{parameter_note}\n\n確認すべき案件・部署は、{projects}です。\n\n"
            f"{action_angle} {segment_focus(context['segment'])} "
            "Critical/High案件は、アクションオーナー、期限、金額インパクトをセットで管理する必要があります。"
        )

    if task == "board":
        board_angle = variant_pick(
            task + "-board",
            context,
            [
                "経営会議では、売上の見かけの改善と利益・CFの悪化を分けて説明する必要があります。",
                "資料上は、全社KPI、セグメント差異、案件アクションの順に示すと論点が通りやすくなります。",
                "意思決定事項は、追加コストの回収、見込修正、CF改善策の3点に絞るのが有効です。",
            ],
        )
        return (
            f"{opening}、{context['segment']}の{context['kpi_ja']}は{value_text}となり、{direction}しています。"
            f"{period_note} 主因は{driver_text}です。\n\n"
            f"{segment_focus(context['segment'])} {kpi_focus(kpi)} {board_angle}\n\n"
            "経営会議では、差異金額だけでなく、回復可能性、責任部署、次回見込への反映要否を確認することを提案します。"
        )

    explain_angle = variant_pick(
        task + "-explain",
        context,
        [
            "差異は単一要因ではなく、事業ミックスと案件進捗の組み合わせとして説明するのが自然です。",
            "この条件では、まず大口要因を押さえ、その後に案件単位の確認へ進むのが効率的です。",
            "コメントでは、数値差、主要因、確認アクションの順で整理すると実務資料に転用しやすくなります。",
        ],
    )
    return (
        f"{parameter_note}\n\n{opening}、{context['kpi_ja']}は{value_text}となり、{direction}しています。"
        f"主要因は{driver_text}です。{segment_focus(context['segment'])} {kpi_focus(kpi)}\n\n"
        f"{explain_angle}"
    )


def render_variance_analysis(kpis: pd.DataFrame, drivers: pd.DataFrame, risk: pd.DataFrame, show_guide: bool = True) -> None:
    render_header("Variance Analysis", "差異サマリ / Waterfall / AI comment")
    if show_guide:
        render_page_guide(
            "この画面の見方 / 差異を金額から要因へ分解する",
            "期間、セグメント、KPI、比較タイプを変えることで、予実差、見込更新差、前年差、中計差を同じ構造で確認できます。",
            [
                "BaseとTargetの差額を最初に確認し、差異が有利か不利かを判断します。",
                "ウォーターフォールでは、差異要因がどの順番でKPIを押し上げたか、または押し下げたかを見ます。",
                "AIコメントは、表やチャートを経営会議で説明するための文章案として使います。",
            ],
            "読み解きのコツ: 金額差だけでなく、EAC、外注費、為替、売上計上時期のどれが説明責任の中心かを見極めます。",
        )

    c1, c2, c3, c4 = st.columns([1, 1.25, 1, 1.35])
    with c1:
        selected_period = st.selectbox("期間", period_options(kpis), index=0, key="variance_period")
    with c2:
        selected_segment = st.selectbox("セグメント", segment_options(kpis), index=0, key="variance_segment")
    with c3:
        selected_kpi = st.selectbox("KPI", list(KPI_META.keys()), format_func=lambda x: f"{KPI_META[x]['ja']} / {x}")
    with c4:
        comparison = st.selectbox("比較タイプ", list(COMPARISON_MAP.keys()))

    base_scenario, current_scenario = COMPARISON_MAP[comparison]
    base = aggregate_kpis(kpis, base_scenario, selected_period, selected_segment)
    current = aggregate_kpis(kpis, current_scenario, selected_period, selected_segment)
    base_value, current_value, variance = kpi_variance(base, current, selected_kpi)

    cols = st.columns(3)
    with cols[0]:
        metric_card(
            f"Base: {SCENARIO_JA[base_scenario]}",
            format_kpi_value(selected_kpi, base_value),
            base_scenario,
            favorable=None,
        )
    with cols[1]:
        metric_card(
            f"Target: {SCENARIO_JA[current_scenario]}",
            format_kpi_value(selected_kpi, current_value),
            current_scenario,
            favorable=None,
        )
    with cols[2]:
        pct_text = "" if KPI_META[selected_kpi]["unit"] == "margin" else f"{pct_change(current_value, base_value):+.1f}%"
        metric_card(
            f"Variance: {KPI_META[selected_kpi]['ja']}",
            format_kpi_delta(selected_kpi, variance),
            "Favorable" if variance >= 0 else "Unfavorable",
            pct_text,
            favorable=variance >= 0,
        )

    scoped_drivers = filter_drivers(drivers, selected_period, comparison, selected_segment)
    driver_summary = summarize_driver_impacts(scoped_drivers, selected_kpi, base["revenue_jpy_mn"])
    display_base = base_value
    display_current = current_value
    if KPI_META[selected_kpi]["unit"] == "amount":
        table = driver_summary.copy()
        table["impact_display"] = table["impact_display"] / 1_000
        impact_label = "影響額 (JPY bn)"
    else:
        table = driver_summary.copy()
        impact_label = "影響 (pt)"

    left, right = st.columns([1.25, 1.0])
    with left:
        st.plotly_chart(render_waterfall(display_base, display_current, driver_summary, selected_kpi), width="stretch")
    with right:
        st.markdown('<div class="section-label">Top差異要因 / Driver Table</div>', unsafe_allow_html=True)
        table = table.sort_values("abs_impact", ascending=False).head(8)
        st.dataframe(
            table[["variance_driver_ja", "variance_driver", "impact_display"]].rename(
                columns={
                    "variance_driver_ja": "差異要因",
                    "variance_driver": "Driver",
                    "impact_display": impact_label,
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={impact_label: st.column_config.NumberColumn(impact_label, format="%.1f")},
        )

    context = build_context(kpis, drivers, risk, selected_period, selected_segment, comparison, selected_kpi)
    st.markdown('<div class="section-label">AIコメント / Rule-based commentary</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-box'>{escape(rule_based_commentary('explain', context)).replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )


ACTION_STATUS_OPTIONS = ["未着手", "確認中", "対応中", "経過観察"]
FORECAST_REFLECTION_OPTIONS = ["必要", "判断中", "不要"]
ACTION_REFERENCE_DATE = pd.Timestamp("2026-06-23")


def project_sequence(project_id: Any) -> int:
    try:
        return int(str(project_id).split("-")[-1])
    except (TypeError, ValueError):
        return 0


def action_due_date(row: pd.Series) -> pd.Timestamp:
    risk_level = row.get("risk_level")
    sequence = project_sequence(row.get("project_id"))
    base_days = {
        "Critical": 7,
        "High": 14,
        "Medium": 28,
        "Low": 45,
    }.get(str(risk_level), 30)
    return ACTION_REFERENCE_DATE + pd.Timedelta(days=base_days + sequence % 7)


def action_status(row: pd.Series) -> str:
    sequence = project_sequence(row.get("project_id"))
    risk_level = row.get("risk_level")
    if risk_level == "Critical":
        return ["対応中", "未着手", "確認中"][sequence % 3]
    if risk_level == "High":
        return ["確認中", "対応中", "未着手"][sequence % 3]
    if risk_level == "Medium":
        return ["確認中", "経過観察"][sequence % 2]
    return "経過観察"


def action_recovery_jpy_mn(row: pd.Series) -> float:
    deterioration = max(float(row.get("eac_deterioration_jpy_mn", 0.0)), 0.0)
    risk_level = row.get("risk_level")
    if risk_level == "Critical":
        ratio = 0.28
    elif risk_level == "High":
        ratio = 0.22
    elif risk_level == "Medium":
        ratio = 0.12
    else:
        ratio = 0.06
    if bool(row.get("loss_risk_flag", False)):
        ratio += 0.04
    return deterioration * ratio


def forecast_reflection(row: pd.Series, recovery_jpy_mn: float) -> str:
    if row.get("risk_level") == "Critical" or bool(row.get("loss_risk_flag", False)):
        return "必要"
    if row.get("risk_level") == "High" or recovery_jpy_mn >= 1_000:
        return "判断中"
    return "不要"


def simple_issue(row: pd.Series) -> str:
    deterioration = float(row.get("eac_deterioration_jpy_mn", 0.0))
    margin = float(row.get("forecast_margin_pct", 0.0))
    if margin < 0:
        return f"コストが{format_amount(deterioration)}悪化し、利益率が赤字水準"
    return f"コストが{format_amount(deterioration)}悪化し、利益率が{margin:.1f}%まで低下"


def simple_action(row: pd.Series) -> str:
    driver = str(row.get("primary_driver_ja", ""))
    if "EAC" in driver or "設計変更" in driver:
        return "EACを再見積し、顧客と変更契約を交渉する"
    if "外注費" in driver or "為替" in driver:
        return "外注単価と為替ヘッジを確認する"
    if "需要鈍化" in driver or "固定費" in driver:
        return "受注確度と固定費吸収計画を見直す"
    if "高採算" in driver or "前倒し" in driver:
        return "前倒し計上と追加オーダーの確度を確認する"
    action = str(row.get("recommended_action_ja", "")).split("、")[0]
    return action or "担当部署で回復策と見込反映要否を確認する"


def build_action_register(risk: pd.DataFrame) -> pd.DataFrame:
    if risk.empty:
        return pd.DataFrame()

    action = risk.copy()
    action["期限"] = action.apply(action_due_date, axis=1)
    action["対応状況"] = action.apply(action_status, axis=1)
    action["recovery_jpy_mn"] = action.apply(action_recovery_jpy_mn, axis=1)
    action["業績見通し修正"] = action.apply(
        lambda row: forecast_reflection(row, float(row["recovery_jpy_mn"])),
        axis=1,
    )
    action["優先度"] = action["risk_level"]
    action["案件"] = action["project_id"].astype(str) + " " + action["project_name"].astype(str)
    action["何が問題か"] = action.apply(simple_issue, axis=1)
    action["まずやること"] = action.apply(simple_action, axis=1)
    action["担当部署"] = action["owner_department"]
    action["コスト悪化額_jpy_bn"] = action["eac_deterioration_jpy_mn"] / 1_000
    action["戻せる見込_jpy_bn"] = action["recovery_jpy_mn"] / 1_000
    return action


def risk_level_from_score(score: float) -> str:
    if score >= 86:
        return "Critical"
    if score >= 70:
        return "High"
    if score >= 48:
        return "Medium"
    return "Low"


def risk_animation_periods(kpis: pd.DataFrame | None = None) -> list[str]:
    if kpis is not None and not kpis.empty and "period" in kpis.columns:
        periods = sorted(kpis["period"].dropna().astype(str).unique().tolist())
        if periods:
            return periods
    return [str(period) for period in pd.period_range("2025-04", periods=12, freq="M")]


def risk_progression(month_index: int, total_months: int, record: dict[str, Any]) -> float:
    risk_level = str(record.get("risk_level", "Low"))
    schedule_delay = max(float(record.get("schedule_delay_days", 0.0)), 0.0)
    sequence = project_sequence(record.get("project_id"))
    acceleration = {
        "Critical": 0.58,
        "High": 0.48,
        "Medium": 0.34,
        "Low": 0.22,
    }.get(risk_level, 0.30)
    month_ratio = month_index / max(total_months, 1)
    delay_ratio = min(schedule_delay / 180.0, 1.0)
    curve_power = max(0.72, 1.68 - acceleration - delay_ratio * 0.28)
    project_shift = ((sequence % 5) - 2) * 0.018 * month_ratio * (1 - month_ratio)
    progress = month_ratio**curve_power + acceleration * 0.16 * month_ratio + project_shift
    return float(np.clip(progress, 0.0, 1.0))


def build_project_risk_animation(risk: pd.DataFrame, kpis: pd.DataFrame | None = None) -> pd.DataFrame:
    if risk.empty:
        return pd.DataFrame()

    periods = risk_animation_periods(kpis)
    total_months = len(periods)
    rows: list[dict[str, Any]] = []
    for record in risk.to_dict("records"):
        annual_revenue = max(float(record.get("annual_revenue_budget_jpy_mn", 0.0)), 1.0)
        budget_eac = float(record.get("budget_eac_jpy_mn", annual_revenue * 0.905))
        final_eac_delta = float(record.get("eac_deterioration_jpy_mn", 0.0))
        final_margin = float(record.get("forecast_margin_pct", 0.0))
        start_margin = (annual_revenue - budget_eac) / annual_revenue * 100
        final_score = float(record.get("risk_score", 0.0))
        schedule_delay = float(record.get("schedule_delay_days", 0.0))
        delay_ratio = min(max(schedule_delay / 180.0, 0.0), 1.0)
        score_start = max(8.0, final_score - 34.0 - delay_ratio * 10.0)

        for month_index, period in enumerate(periods, start=1):
            progress = risk_progression(month_index, total_months, record)
            eac_delta = final_eac_delta * progress
            forecast_margin = start_margin + (final_margin - start_margin) * progress
            risk_score = score_start + (final_score - score_start) * progress
            risk_level = risk_level_from_score(risk_score)
            if month_index == total_months:
                risk_score = final_score
                risk_level = str(record.get("risk_level", risk_level))
                forecast_margin = final_margin
                eac_delta = final_eac_delta

            animated_record = dict(record)
            animated_record.update(
                {
                    "period": period,
                    "fiscal_month": month_index,
                    "eac_deterioration_jpy_mn_frame": eac_delta,
                    "eac_deterioration_jpy_bn": eac_delta / 1_000,
                    "forecast_margin_pct_frame": forecast_margin,
                    "risk_score_frame": risk_score,
                    "risk_level_frame": risk_level,
                    "schedule_delay_days_frame": schedule_delay * progress,
                    "loss_risk_flag_frame": bool(forecast_margin < 0 or risk_score >= 88),
                }
            )
            rows.append(animated_record)

    return pd.DataFrame(rows)


def risk_animation_ranges(animated: pd.DataFrame) -> tuple[list[float], list[float]]:
    x_values = animated["eac_deterioration_jpy_bn"]
    y_values = animated["forecast_margin_pct_frame"]
    x_span = max(float(x_values.max() - x_values.min()), 1.0)
    y_span = max(float(y_values.max() - y_values.min()), 4.0)
    x_pad = max(x_span * 0.12, 1.0)
    y_pad = max(y_span * 0.12, 1.5)
    range_x = [min(0.0, float(x_values.min())) - x_pad, max(0.0, float(x_values.max())) + x_pad]
    range_y = [min(0.0, float(y_values.min())) - y_pad, max(10.0, float(y_values.max())) + y_pad]
    return range_x, range_y


def render_project_risk(risk: pd.DataFrame, kpis: pd.DataFrame | None = None, show_guide: bool = True) -> None:
    render_header("Project Risk", "Risk map / Action register / Forecast impact")
    if show_guide:
        render_page_guide(
            "この画面の見方 / 危ない案件をアクションに変える",
            "重工業のFP&Aでは、危ない案件を見つけるだけでなく、誰が、いつまでに、何をして、いくら戻せる可能性があるかを経営会議で確認できる状態にすることが重要です。",
            [
                "上部のカードでは、すぐ対応が必要な件数、期限が近い件数、戻せる見込金額、業績見通し修正が必要な件数を確認します。",
                "Animated Bubbleでは、コスト悪化額、利益率、リスクスコアが期中にどう悪化したかを月次で確認します。",
                "アクション台帳では、問題、まずやること、担当部署、期限、戻せる見込、業績見通し修正の要否を案件単位で整理します。",
            ],
            "この画面のゴールは、単なるリスク一覧ではなく、経営会議後に追える是正アクション台帳を作ることです。",
        )

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        segment_filter = st.selectbox(
            "セグメント",
            ["全社 / Total"] + [f"{row.segment_ja} / {row.segment_en}" for row in risk[["segment_ja", "segment_en"]].drop_duplicates().itertuples()],
            key="risk_segment",
        )
    with c2:
        risk_filter = st.multiselect("リスクレベル", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
    with c3:
        loss_only = st.toggle("赤字化リスクのみ", value=False)

    scoped = risk.copy()
    segment_en = parse_segment(segment_filter)
    if segment_en:
        scoped = scoped[scoped["segment_en"] == segment_en]
    if risk_filter:
        scoped = scoped[scoped["risk_level"].isin(risk_filter)]
    if loss_only:
        scoped = scoped[scoped["loss_risk_flag"]]

    action_register = build_action_register(scoped)
    f1, f2 = st.columns([1.0, 1.0])
    with f1:
        status_filter = st.multiselect("対応状況", ACTION_STATUS_OPTIONS, default=ACTION_STATUS_OPTIONS)
    with f2:
        forecast_filter = st.multiselect("業績見通し修正", FORECAST_REFLECTION_OPTIONS, default=FORECAST_REFLECTION_OPTIONS)

    if not action_register.empty:
        action_register = action_register[action_register["対応状況"].isin(status_filter)]
        action_register = action_register[action_register["業績見通し修正"].isin(forecast_filter)]
        scoped = scoped[scoped["project_id"].isin(action_register["project_id"])]

    cards = st.columns(4)
    near_due_count = 0
    recovery_total = 0.0
    forecast_update_count = 0
    if not action_register.empty:
        due_dates = pd.to_datetime(action_register["期限"])
        near_due_count = int((due_dates <= ACTION_REFERENCE_DATE + pd.Timedelta(days=14)).sum())
        recovery_total = float(action_register["recovery_jpy_mn"].sum())
        forecast_update_count = int((action_register["業績見通し修正"] == "必要").sum())
    metrics = [
        ("すぐ対応が必要", int(scoped["risk_level"].isin(["Critical", "High"]).sum()), "件"),
        ("期限が近い", near_due_count, "件"),
        ("戻せる見込", recovery_total, "amount"),
        ("見通し修正が必要", forecast_update_count, "件"),
    ]
    for col, (label, value, unit) in zip(cards, metrics):
        with col:
            display = format_amount(float(value)) if unit == "amount" else f"{int(value):,}{unit}"
            metric_card(label, display, "Action Register", favorable=None)

    if action_register.empty:
        st.info("条件に一致するアクションはありません。フィルタ条件を変更してください。")
        return

    map_mode = st.radio(
        "リスクマップ表示",
        ["月次進行 / Animated Bubble", "最新スナップショット / Current Bubble"],
        horizontal=True,
        key="risk_map_mode",
    )
    if map_mode.startswith("月次進行"):
        animated = build_project_risk_animation(scoped, kpis)
        range_x, range_y = risk_animation_ranges(animated)
        fig = px.scatter(
            animated,
            x="eac_deterioration_jpy_bn",
            y="forecast_margin_pct_frame",
            animation_frame="period",
            animation_group="project_id",
            size="annual_revenue_budget_jpy_mn",
            size_max=48,
            color="risk_level_frame",
            color_discrete_map=RISK_COLORS,
            category_orders={"period": risk_animation_periods(kpis)},
            hover_name="project_name",
            hover_data={
                "project_id": True,
                "segment_ja": True,
                "risk_score_frame": ":.1f",
                "schedule_delay_days_frame": ":.0f",
                "eac_deterioration_jpy_mn_frame": ":,.0f",
                "annual_revenue_budget_jpy_mn": ":,.0f",
                "forecast_margin_pct_frame": ":.1f",
                "risk_level_frame": False,
                "period": False,
            },
            range_x=range_x,
            range_y=range_y,
            labels={
                "eac_deterioration_jpy_bn": "Cumulative EAC deterioration (JPY bn)",
                "forecast_margin_pct_frame": "Forecast Margin %",
                "risk_level_frame": "Risk level",
                "annual_revenue_budget_jpy_mn": "Annual revenue budget (JPY mn)",
            },
            title="案件リスク推移 / Plotly Animated Bubble",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#ff647c", annotation_text="Loss threshold")
        fig.add_vline(x=0, line_dash="dot", line_color="rgba(159,178,195,0.45)")
        fig.update_traces(marker={"line": {"width": 0.8, "color": "rgba(237,246,249,0.55)"}})
        slider_config = []
        for slider in fig.layout.sliders:
            slider_dict = slider.to_plotly_json()
            slider_dict["currentvalue"] = {"prefix": "Month: "}
            slider_config.append(slider_dict)
        fig.update_layout(sliders=slider_config)
        st.plotly_chart(style_fig(fig, 500), width="stretch")
    else:
        fig = px.scatter(
            scoped,
            x=scoped["eac_deterioration_jpy_mn"] / 1_000,
            y="forecast_margin_pct",
            size="annual_revenue_budget_jpy_mn",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            hover_name="project_name",
            hover_data={
                "project_id": True,
                "segment_ja": True,
                "risk_score": True,
                "schedule_delay_days": True,
                "eac_deterioration_jpy_mn": ":,.0f",
                "annual_revenue_budget_jpy_mn": ":,.0f",
                "forecast_margin_pct": ":.1f",
            },
            labels={"x": "EAC Deterioration (JPY bn)", "forecast_margin_pct": "Forecast Margin %"},
            title="案件別リスクマップ / Project Risk Map",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#ff647c", annotation_text="Loss threshold")
        st.plotly_chart(style_fig(fig, 460), width="stretch")

    st.markdown('<div class="section-label">是正アクション台帳 / Action Register</div>', unsafe_allow_html=True)
    display_cols = [
        "優先度",
        "案件",
        "何が問題か",
        "コスト悪化額_jpy_bn",
        "まずやること",
        "担当部署",
        "期限",
        "戻せる見込_jpy_bn",
        "業績見通し修正",
        "対応状況",
    ]
    st.dataframe(
        action_register.sort_values(["risk_score", "eac_deterioration_jpy_mn"], ascending=[False, False])[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "コスト悪化額_jpy_bn": st.column_config.NumberColumn("コスト悪化額 (JPY bn)", format="%.1f"),
            "戻せる見込_jpy_bn": st.column_config.NumberColumn("戻せる見込 (JPY bn)", format="%.1f"),
            "期限": st.column_config.DateColumn("期限", format="YYYY-MM-DD"),
        },
    )


def render_ai_commentary(kpis: pd.DataFrame, drivers: pd.DataFrame, risk: pd.DataFrame, show_guide: bool = True) -> None:
    render_header("AI Commentary", "Executive-ready Japanese comments")
    if show_guide:
        render_page_guide(
            "この画面の見方 / 分析結果を経営会議向けの言葉に変換する",
            "AI Commentaryは、差異金額、差異要因、案件リスクをもとに、事業部長・FP&A担当者がそのまま資料に貼れる日本語コメントを作る画面です。",
            [
                "差異要因の説明、ランキング、確認すべき案件・部署、経営会議向けコメントを目的別に切り替えます。",
                "デモでは外部APIを呼び出さず、集計済みの差異要因と案件リスクからルールベースで生成します。",
                "実運用では、AIコメントの根拠となるKPI定義とデータリネージが重要になります。",
            ],
            "AIは最後の文章化を助けますが、説得力の源泉は正しいFP&Aデータと差異要因の構造化です。",
        )

    c1, c2, c3, c4 = st.columns([1, 1.25, 1.35, 1.0])
    with c1:
        selected_period = st.selectbox("期間", period_options(kpis), index=0, key="ai_period")
    with c2:
        selected_segment = st.selectbox("セグメント", segment_options(kpis), index=0, key="ai_segment")
    with c3:
        comparison = st.selectbox("比較タイプ", list(COMPARISON_MAP.keys()), key="ai_comparison")
    with c4:
        selected_kpi = st.selectbox("KPI", list(KPI_META.keys()), index=1, format_func=lambda x: KPI_META[x]["ja"], key="ai_kpi")

    context = build_context(kpis, drivers, risk, selected_period, selected_segment, comparison, selected_kpi)

    b1, b2, b3, b4 = st.columns(4)
    task = None
    with b1:
        if st.button("差異要因を説明", width="stretch"):
            task = "explain"
    with b2:
        if st.button("重要要因をランキング", width="stretch"):
            task = "rank"
    with b3:
        if st.button("確認すべき案件・部署を提案", width="stretch"):
            task = "recommend"
    with b4:
        if st.button("経営会議向けコメントを作成", width="stretch"):
            task = "board"

    current_task = task or st.session_state.get("last_ai_task", "board")
    current_signature = context_signature(current_task, context)
    should_generate = (
        task is not None
        or "last_ai_comment" not in st.session_state
        or st.session_state.get("last_ai_signature") != current_signature
    )

    if should_generate:
        comment_text = rule_based_commentary(current_task, context)
        mode = "Rule-based commentary"
        st.session_state["last_ai_comment"] = comment_text
        st.session_state["last_ai_mode"] = mode
        st.session_state["last_ai_task"] = current_task
        st.session_state["last_ai_signature"] = current_signature

    mode = st.session_state.get("last_ai_mode", "Rule-based commentary")
    comment = st.session_state.get("last_ai_comment", rule_based_commentary(current_task, context))

    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-cell"><b>Generation</b>{escape(mode)}</div>
            <div class="status-cell"><b>Audience</b>事業部長 / FP&amp;A</div>
            <div class="status-cell"><b>Style</b>経営会議資料向け</div>
            <div class="status-cell"><b>Data</b>Fictional parquet facts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='insight-box'>{escape(comment).replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    with st.expander("Context Snapshot / AI入力サマリ", expanded=False):
        st.json(
            {
                "period": context["period"],
                "segment": context["segment"],
                "comparison": context["comparison"],
                "kpi": context["kpi_ja"],
                "variance": format_kpi_delta(context["kpi"], context["variance"]),
                "top_drivers": [
                    {
                        "driver": d["variance_driver_ja"],
                        "impact": driver_phrase(d, context["kpi"]),
                    }
                    for d in context["top_drivers"]
                ],
                "critical_count": context["critical_count"],
                "high_count": context["high_count"],
            }
        )


def render_data_explorer(data: dict[str, Any]) -> None:
    render_header("Data Explorer", "Generated fictional data inventory")
    render_page_guide(
        "この画面の見方 / デモデータの中身を確認する",
        "このアプリは架空データで動いています。Data Explorerでは、どのデータセットに何行あり、どのカラムがあり、どのような値が入っているかを確認できます。",
        [
            "fact_financeは、シナリオ、月、セグメント、案件、勘定科目別の金額データです。",
            "fact_variance_driversは、比較タイプ別の差異要因インパクトです。",
            "project_riskは、案件別のEAC悪化、赤字化リスク、推奨アクションです。",
        ],
        "実案件では、この画面に相当する確認がデータ品質レビューや要件定義の出発点になります。",
    )

    datasets = {
        "fact_finance": data["fact_finance"],
        "fact_variance_drivers": data["fact_variance_drivers"],
        "project_risk": data["project_risk"],
        "dim_projects": data["dim_projects"],
    }
    selected = st.selectbox("データセット", list(datasets.keys()))
    df = datasets[selected]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Rows", f"{len(df):,}", selected, favorable=None)
    with c2:
        metric_card("Columns", f"{len(df.columns):,}", "Schema", favorable=None)
    with c3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metric_card("Memory", f"{memory_mb:,.1f} MB", "Loaded in cache", favorable=None)

    st.markdown('<div class="section-label">Columns</div>', unsafe_allow_html=True)
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [int(df[col].notna().sum()) for col in df.columns],
        }
    )
    st.dataframe(schema, width="stretch", hide_index=True)

    st.markdown('<div class="section-label">Head</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), width="stretch", hide_index=True)

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        st.markdown('<div class="section-label">Basic Statistics</div>', unsafe_allow_html=True)
        st.dataframe(numeric.describe().T, width="stretch")

    metadata = data.get("metadata", {})
    if metadata:
        with st.expander("Metadata / 生成情報", expanded=False):
            st.json(metadata)


def render_data_foundation(data: dict[str, Any]) -> None:
    st.markdown('<div id="foundation-page"></div>', unsafe_allow_html=True)
    render_header(
        "FP&Aデータ基盤 / Data Foundation",
        "経営説明・意思決定のために、定義・版・粒度・根拠をそろえる",
    )

    slide = render_presentation_controls(
        "data_foundation",
        ["経営リスク", "3つの論点", "データの流れ", "つなぐデータ", "品質ゲート", "経営判断"],
    )

    if slide == 0:
        render_presentation_slide(
            "Executive Risk Message",
            "AI活用の前に、経営説明の根拠をつなぐ必要があります",
            """
            <div class="presentation-lead">
            LLMは文章を作ることはできます。しかし、売上・利益・キャッシュフロー・案件EAC・調達・工程・マスタが分断されたままでは、
            経営会議で「なぜ業績が悪化したのか」「どの案件に手を打つべきか」を根拠付きで説明できません。
            </div>
            <div class="presentation-note">
            速くなるのは説明ではなく、もっともらしいが検証できないコメントの生成です。
            </div>
            """,
            "AI導入の前に、経営データの定義・版・粒度・根拠をそろえる必要があります。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Management Thesis",
            "CFO・経営企画・FP&Aに伝える3つの論点",
            presentation_cards_html(
                [
                    (
                        "AI投資の落とし穴",
                        "AIツールを導入しても、根拠データが分断されていれば、生成されるのは検証しにくいコメントです。",
                    ),
                    (
                        "本当の経営リスク",
                        "経営会議で「なぜ悪化したのか」「どの案件に手を打つべきか」を説明できないことがリスクです。",
                    ),
                    (
                        "先に整えるべきもの",
                        "LLMの前に、売上・利益・CF・案件リスクの根拠をつなぐ経営データ基盤が必要です。",
                    ),
                ],
                columns=3,
            ),
            "この論点はIT施策ではなく、経営判断の品質に関わるアジェンダです。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Data Flow",
            "経営データの流れ",
            """
            <div class="presentation-lead">
            同じデータでも、手元Excelでつなぐだけでは「根拠を追えないAIコメント」に流れます。
            品質ゲートとTrusted FP&amp;A Data Foundationを通すことで、KPI、差異要因、案件リスク、AIコメント、経営判断が同じ根拠でつながります。
            </div>
            """
            + presentation_divider_html("経営データの接続順")
            + foundation_flow_map_html()
            + presentation_divider_html("この図で伝えること")
            + presentation_cards_html(
                [
                    ("Excel連結だけでは弱い", "集計はできても、版・粒度・根拠が揃わないとAIコメントの説明責任を持てません。"),
                    ("品質ゲートが中核", "照合、版管理、ID統合、リネージを通すことで、コメントから元データへ戻れます。"),
                    ("経営判断へつなげる", "KPI、差異、案件リスク、会議資料が同じ根拠でつながる状態を目指します。"),
                ],
                columns=3,
            ),
            "図は概念図です。本番では既存DWH、EPM、ERP、案件管理システムの構成に合わせて接続点を設計します。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Data Domains",
            "つなぐべきデータ",
            """
            <div class="presentation-lead">
            すべてのデータを最初から集める必要はありません。経営説明に効く起点は、財務の比較軸、案件の見通し、
            そして両者を結ぶマスタです。この3つがつながると、AIコメントの根拠が追えるようになります。
            </div>
            """
            + presentation_divider_html("最初に接続する範囲")
            + presentation_cards_html(
                [
                    ("財務の比較軸", "ERP実績とEPMの予算・見込を、同じ期間と粒度で比較できる状態にします。"),
                    ("案件の見通し", "EAC、調達、工程の変化を案件単位で見て、利益悪化の原因を説明します。"),
                    ("共通マスタ", "セグメント、案件、勘定科目、顧客の定義をそろえ、数字の意味をぶらさないようにします。"),
                ],
                columns=3,
            ),
            "AIコメントの精度は、これらのデータが同じ粒度と定義でつながるかで決まります。",
        )
    elif slide == 4:
        render_presentation_slide(
            "Quality Gates",
            "最低限の品質ゲート",
            """
            <div class="presentation-lead">
            AIコメントから元データまで戻れることが、経営会議で使う最低条件です。
            コメント、KPI、差異要因、案件、元データの線が切れていると、説明責任を果たす材料にはなりません。
            </div>
            """
            + presentation_divider_html("最低限の確認ポイント")
            + foundation_quality_gates_html(),
            "まずは代表KPIと重点案件に絞り、この3つの品質ゲートを月次運用に組み込みます。",
        )
    else:
        render_presentation_slide(
            "Executive Decisions",
            "経営層・FP&Aが決めること",
            """
            <div class="presentation-lead">
            経営層が決めるべきことは、AIツールの種類ではありません。
            どの会議で、どのKPIを、どの事業領域から説明可能にするかです。
            </div>
            """
            + presentation_divider_html("決めること")
            + presentation_cards_html(
                [
                    ("会議", "月次経営会議、予実会議、見込更新会議のどこから説明品質を変えるか。"),
                    ("KPI", "売上、営業利益、利益率、キャッシュフローのどこに説明責任を持たせるか。"),
                    ("開始範囲", "赤字化リスクやEAC悪化が大きい案件群から始め、成果を見せやすくする。"),
                ],
                columns=3,
            )
            + """
            <div class="presentation-note">
            AI活用の成否は、モデル選定だけでは決まりません。経営会議で使えるAIにするには、
            まず経営データの定義・版・粒度・根拠を揃える必要があります。
            </div>
            """,
            "ここで決めた範囲が、データ基盤アセスメントと短期PoCの対象になります。",
        )


def render_reference_architecture() -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    render_header(
        "構成・進め方 / Reference Architecture",
        "経営説明から逆算して、FP&Aデータ基盤とAI活用を設計する",
    )

    slide = render_presentation_controls(
        "reference_architecture",
        ["全体像", "推奨構成", "構成要素"],
    )

    if slide == 0:
        nodes, edges = architecture_graph_system_data()
        render_architecture_graph_slide(
            "Reference View",
            "構成は、経営会議で説明したい問いから逆算して設計する",
            """
            <div class="presentation-lead">
            最初に決めるべきことは、LLMの製品名ではありません。
            どの会議で、どのKPIを、どの根拠データに戻れる状態で説明するかを定義し、その上にAIを置きます。
            </div>
            """,
            nodes,
            edges,
            "reference_architecture_system_graph",
            presentation_divider_html("読み方")
            + presentation_cards_html(
                [
                    ("左側", "ERP、EPM、案件管理、調達、マスタなど、AIコメントの根拠になる業務データです。"),
                    ("中央", "品質ゲート、DWH、FP&Aデータマート、KPI定義が、説明可能性を作ります。"),
                    ("右側", "AI説明サービス、Cockpit、会議、承認・アクションが業務利用の出口です。"),
                ],
                columns=3,
            ),
            "このデモは画面デモですが、本番化の焦点はFP&Aデータ基盤、AI実装、統制運用をつなぐことです。",
        )
    elif slide == 1:
        nodes, edges = architecture_graph_system_data()
        render_architecture_graph_slide(
            "Recommended Architecture",
            "推奨構成: FP&Aデータ基盤を全社データ・AI基盤へつなぐ",
            """
            <div class="presentation-lead">
            FP&amp;Aだけで閉じる構成ではなく、全社データ基盤、AI共通基盤、業務ワークフローへ接続できる形で設計します。
            短期PoCでも、この将来構成に接続できるようにKPI定義、データ粒度、出力統制を先に決めます。
            </div>
            """,
            nodes,
            edges,
            "reference_architecture_recommended_graph",
            presentation_divider_html("設計上のポイント")
            + presentation_cards_html(
                [
                    ("設計の考え方", "FP&Aだけで閉じず、全社データ基盤とAI共通基盤に接続できる形で作る。"),
                    ("最初の範囲", "代表会議、代表KPI、重点案件に絞り、FP&Aデータマートの最小構成から始める。"),
                ],
                columns=2,
            ),
            "短期PoCでも、この将来構成に接続できる形でKPI定義とデータ粒度を決めます。",
        )
    elif slide == 2:
        nodes, edges = architecture_graph_system_data()
        render_architecture_graph_slide(
            "Building Blocks",
            "主要な構成要素",
            """
            <div class="presentation-lead">
            構成要素は多く見えますが、説明するときは「根拠データ」「AI説明サービス」「業務出口」の三層で十分です。
            図のノードを選択すると、各オブジェクトの役割を確認できます。
            </div>
            """,
            nodes,
            edges,
            "reference_architecture_building_blocks_graph",
            presentation_divider_html("三層で説明する")
            + presentation_cards_html(
                [
                    ("Data foundation", "ERP、EPM、案件EAC、調達、工程、マスタを、FP&Aデータマートで同じ説明粒度にそろえる。"),
                    ("AI service", "AIは自由回答ではなく、根拠引用、プロンプト管理、出力評価を持つ説明サービスとして扱う。"),
                    ("Experience", "Cockpit、BI、会議資料、通知をつなぎ、確認とアクションを同じ流れにする。"),
                ],
                columns=3,
            ),
            "構成要素を分けておくと、短期PoC、本番化、全社展開を段階的に進めやすくなります。",
        )


def render_tech_architecture(data: dict[str, Any]) -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    metadata = data.get("metadata", {})
    row_counts = metadata.get("row_counts", {})
    generated_at = format_generated_at(metadata)
    total_rows = row_count_text(metadata)

    render_header("技術構成 / Tech Architecture", "社内説明用: デモの構成、処理、公開運用、次の拡張を説明する")

    titles = [
        "全体像",
        "URLとアプリ分離",
        "データ構成",
        "処理レイヤー",
        "AIコメント設計",
        "画面構成",
        "公開・運用",
        "本番化ロードマップ",
    ]
    slide = render_presentation_controls("tech_architecture", titles)

    if slide == 0:
        render_presentation_slide(
            "Architecture / 01",
            "架空データで体験する FP&Aデータ基盤・AI活用コックピット",
            f"""
            <div class="presentation-lead">
            このデモは、実データや外部AI APIに依存せず、架空データだけでAI-driven FP&amp;Aの体験を見せる構成です。
            Python、Parquet、Streamlit Cloudを使い、短期間で見せられる一方、本番化時はデータ基盤とAI統制へ差し替える前提です。
            </div>
            <div class="presentation-note">
            現在のデータは {escape(total_rows)}。生成日時は {escape(generated_at)} です。
            </div>
            """
            + presentation_cards_html(
                [
                    ("画面", "Streamlitでダッシュボードと説明資料を一体化しています。"),
                    ("データ", "架空のParquetとJSONを読み込み、pandasでKPIとリスクを集計します。"),
                    ("公開", "GitHubの入口ファイルを分け、Streamlit CloudでURLごとの体験を出し分けます。"),
                ],
                columns=3,
            ),
            "本質は、AI単体ではなく、信頼できるFP&Aデータ基盤と説明プロセスを体験させることです。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Architecture / 02",
            "URLを分けるために、入口ファイルを3つに分けています",
            """
            <div class="presentation-lead">
            コードベースは一つですが、入口ファイルを分けることでURLごとに体験を変えています。
            クライアントには触る画面だけを見せ、当社側には説明資料と内部確認ページを残します。
            </div>
            """
            + presentation_cards_html(
                [
                    ("client_app.py", "クライアントが操作する本体デモだけを表示します。"),
                    ("presenter_app.py / app.py", "説明者用資料、技術構成、データ確認を当社側で使います。"),
                ],
                columns=2,
            ),
            "クライアント向けURLはダッシュボード専用、社内情報URLは説明資料と当デモ情報に絞っています。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Architecture / 03",
            "データは全て架空で、Parquetを中心に軽量に配布しています",
            """
            <div class="presentation-lead">
            デモではDBに接続せず、架空データをファイルとして持っています。
            この構成は公開が速く、守秘情報を含めずに商談で使える一方、本番ではDWHやFP&Aデータマートへ置き換える想定です。
            </div>
            """
            + presentation_cards_html(
                [
                    ("案件マスタ", f"案件やセグメントの軸を持ちます。現在 {escape(str(row_counts.get('dim_projects', 'N/A')))} 行。"),
                    ("財務・差異", "実績、予算、見込、差異要因を月次粒度で保持します。"),
                    ("案件リスク", f"EAC、遅延、調達圧力などを最新状態として持ちます。現在 {escape(str(row_counts.get('project_risk', 'N/A')))} 行。"),
                ],
                columns=3,
            ),
            "このデモではファイル配布を優先し、本番ではERP/EPM/案件管理システムからの連携に置き換えます。",
        )
    elif slide == 3:
        nodes, edges = architecture_graph_current_demo_data()
        render_architecture_graph_slide(
            "Architecture / 04",
            "現在のデモ実装は、ファイル、処理、入口URLが分かれています",
            """
            <div class="presentation-lead">
            現在は、架空データをParquet/JSONで読み込み、pandasでKPIを意味づけし、Streamlitの入口ファイルでURLごとの体験を分けています。
            本番化する場合は、Parquet部分をDWH/データマートへ、AIコメント部分を統制されたAIサービスへ置き換えます。
            </div>
            """,
            nodes,
            edges,
            "tech_architecture_current_demo_graph",
            presentation_divider_html("実装上の読み方")
            + presentation_cards_html(
                [
                    ("Data", "Parquet/JSONを読み、守秘情報を含まない公開デモとして安定させています。"),
                    ("Logic", "pandasでKPI、差異、案件リスク、コメント用の入力を生成しています。"),
                    ("Deploy", "client_app.pyとpresenter_app.pyで、公開URLごとの表示範囲を分けています。"),
                ],
                columns=3,
            ),
            "StreamlitはMVPの画面実装に適しており、本番ではデータ処理と権限制御を外部基盤に寄せます。",
        )
    elif slide == 4:
        render_presentation_slide(
            "Architecture / 05",
            "AIコメントは、現時点では外部APIを使わない決定的な生成です",
            """
            <div class="presentation-lead">
            デモのAIコメントは、外部APIを呼ばずに、集計済みの数値と差異要因を日本語テンプレートへ落としています。
            そのため公開デモとして安定し、APIキーや通信失敗の影響を受けません。
            </div>
            """
            + presentation_cards_html(
                [
                    ("現在", "期間、KPI、セグメント、差異要因、案件リスクを集計し、定型コメントに変換します。"),
                    ("本番", "LLMを使う場合は、Secrets管理、根拠引用、出力ログ、承認フローを追加します。"),
                ],
                columns=2,
            ),
            "クライアントには『AIの前に、説明可能なデータとロジックが必要』というメッセージを伝えます。",
        )
    elif slide == 5:
        render_presentation_slide(
            "Architecture / 06",
            "画面は、本体デモ、説明資料、当デモ情報に分けています",
            """
            <div class="presentation-lead">
            画面の分け方はシンプルです。相手に触ってもらうURLには本体デモだけを出し、
            当社側のURLには説明資料、技術構成、データ確認を残します。
            </div>
            """
            + presentation_cards_html(
                [
                    ("クライアントが触る", "Dashboard、Variance Analysis、Project Risk、AI Commentaryだけを表示します。"),
                    ("当社側が使う", "デモ前後の説明、データ基盤、技術構成、データ確認を投影・確認に使います。"),
                ],
                columns=2,
            ),
            "『相手に触ってもらう画面』と『こちらが説明する画面』を分けることで、デモ体験が混ざらないようにしています。",
        )
    elif slide == 6:
        render_presentation_slide(
            "Architecture / 07",
            "公開運用は、GitHubを正とし、Streamlit Cloudが読み込む形です",
            """
            <div class="presentation-lead">
            長期運用では、GitHubを正にして変更履歴を残します。
            ローカルで編集し、動作確認してからpushすることで、公開環境とのズレを小さくします。
            </div>
            """
            + presentation_cards_html(
                [
                    ("編集と確認", "cloneした作業フォルダで修正し、py_compileやAppTestで主要導線を確認します。"),
                    ("commit / push", "GitHub Desktopまたはgitで変更理由を残し、mainへ反映します。"),
                    ("公開後QA", "クライアントURLに本体4ページだけ出ること、説明者向けページが混ざらないことを確認します。"),
                ],
                columns=3,
            ),
            "将来APIを使う場合、APIキーはGitHubに置かずStreamlit Secretsへ入れます。",
        )
    else:
        render_presentation_slide(
            "Architecture / 08",
            "本番化では、データ接続、権限、AI統制、監査ログを追加します",
            """
            <div class="presentation-lead">
            本番化で増える論点は多いですが、提案で伝える芯は三つです。
            実データをつなぎ、AIの出力を統制し、月次FP&Aプロセスへ組み込むことです。
            </div>
            """
            + presentation_cards_html(
                [
                    ("データ接続", "ERP、EPM、案件管理、調達、工程、マスタを棚卸しし、KPI定義を合意します。"),
                    ("AI統制", "LLM API、プロンプト、根拠引用、出力レビュー、監査ログを設計します。"),
                    ("業務定着", "代表セグメントからPoCし、月次サイクルと承認プロセスへ組み込みます。"),
                ],
                columns=3,
            ),
            "このスライドは、デモ後に『では自社で実装するには何が必要か』へ会話を進めるために使います。",
        )


def render_missing_data(missing: list[str]) -> None:
    render_header(APP_NAME, "Setup required")
    st.error("デモデータが見つかりません。先にデータ生成スクリプトを実行してください。")
    st.code("python scripts/generate_demo_data.py\nstreamlit run app.py", language="bash")
    st.write("Missing files:")
    for path in missing:
        st.write(f"- {path}")


def main(app_mode: str = "internal") -> None:
    inject_css()
    data = load_data()
    if "missing" in data:
        render_missing_data(data["missing"])
        st.stop()

    kpis = build_project_month_kpis(data["fact_finance"])
    metadata = data.get("metadata", {})

    client_pages = [
        ("Dashboard", "ダッシュボード", False),
        ("Variance Analysis", "差異分析", False),
        ("Project Risk", "案件リスク", False),
        ("AI Commentary", "AIコメント", False),
    ]
    presentation_pages = [
        ("Problem Statement", "導入・問題提起", False),
        ("Target Operating Model", "目指す業務像", False),
        ("FPA Data Foundation", "FP&Aデータ基盤", False),
        ("AI App Architecture", "AIアプリアーキテクチャ", False),
        ("System Architecture", "システムアーキテクチャ", False),
        ("Approach Options", "進め方の選択肢", False),
        ("Recommended Roadmap", "推奨ロードマップ", False),
        ("Assessment Proposal", "アセスメント提案", False),
    ]
    operational_pages = [
        ("Dashboard", "全社ダッシュボード"),
        ("Variance Analysis", "差異分析"),
        ("Project Risk", "案件リスク"),
        ("AI Commentary", "AIコメント"),
    ]
    internal_pages = [
        ("Internal Demo Guide", "デモ解説用", False),
        ("Tech Architecture", "技術構成", False),
        ("Data Explorer", "データ確認", False),
    ]
    presenter_pages = presentation_pages
    client_page_keys = {key for key, _, _ in client_pages}
    client_page_guides = {key: guide for key, _, guide in client_pages}
    client_page_labels = {key: label for key, label, _ in client_pages}
    presentation_page_keys = {key for key, _, _ in presentation_pages}
    presentation_page_guides = {key: guide for key, _, guide in presentation_pages}
    presentation_page_labels = {key: label for key, label, _ in presentation_pages}
    presenter_page_keys = {key for key, _, _ in presenter_pages}
    presenter_page_guides = {key: guide for key, _, guide in presenter_pages}
    presenter_page_labels = {key: label for key, label, _ in presenter_pages}
    internal_page_labels = {key: label for key, label, _ in internal_pages}
    operational_page_labels = {key: label for key, label in operational_pages}
    internal_page_keys = {key for key, _, _ in internal_pages}
    operational_page_keys = {key for key, _ in operational_pages}
    client_only = app_mode in {"client", "dashboard"}
    presenter_only = app_mode == "presenter"

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard" if client_only else "Problem Statement"
        st.session_state["active_surface"] = "client" if client_only else "presenter"
        st.session_state["show_guide"] = False

    if st.session_state.get("active_surface") == "demo":
        st.session_state["active_surface"] = (
            "internal" if st.session_state.get("active_page") in internal_page_keys else "presenter"
        )

    if client_only:
        active_page = st.session_state.get("active_page")
        if active_page not in client_page_keys:
            active_page = "Dashboard"
        st.session_state["active_surface"] = "client"
        st.session_state["active_page"] = active_page
        st.session_state["show_guide"] = client_page_guides.get(active_page, False)
    elif presenter_only:
        active_page = st.session_state.get("active_page")
        if active_page not in presenter_page_keys:
            active_page = "Problem Statement"
        st.session_state["active_surface"] = "presenter"
        st.session_state["active_page"] = active_page
        st.session_state["show_guide"] = presenter_page_guides.get(active_page, False)
    elif st.session_state.get("active_surface") not in {"presenter", "internal"}:
        st.session_state["active_surface"] = "internal" if st.session_state.get("active_page") in internal_page_keys else "presenter"

    active_surface = st.session_state.get("active_surface")
    active_page = st.session_state.get("active_page")
    if active_surface == "client" and active_page not in client_page_keys:
        st.session_state["active_page"] = "Dashboard"
        st.session_state["show_guide"] = False
    elif active_surface == "presenter":
        valid_presenter_keys = presenter_page_keys if presenter_only else presentation_page_keys
        if active_page in internal_page_keys and not presenter_only:
            st.session_state["active_surface"] = "internal"
            st.session_state["show_guide"] = False
        elif active_page not in valid_presenter_keys:
            st.session_state["active_page"] = "Problem Statement"
            st.session_state["show_guide"] = False
    elif active_surface == "operational" and active_page not in operational_page_keys:
        st.session_state["active_page"] = "Dashboard"
        st.session_state["show_guide"] = False
    elif active_surface == "internal" and active_page not in internal_page_keys:
        st.session_state["active_page"] = "Internal Demo Guide"
        st.session_state["show_guide"] = False

    def switch_surface(surface: str) -> None:
        current_page = st.session_state.get("active_page")
        st.session_state["active_surface"] = surface
        if surface == "operational":
            st.session_state["active_page"] = current_page if current_page in operational_page_keys else "Dashboard"
            st.session_state["show_guide"] = False
        elif surface == "internal":
            st.session_state["active_page"] = current_page if current_page in internal_page_keys else "Internal Demo Guide"
            st.session_state["show_guide"] = False
        elif surface == "presenter":
            valid_pages = presenter_page_keys if presenter_only else presentation_page_keys
            st.session_state["active_page"] = current_page if current_page in valid_pages else "Problem Statement"
            st.session_state["show_guide"] = False
        else:
            st.session_state["active_page"] = current_page if current_page in client_page_keys else "Dashboard"
            st.session_state["show_guide"] = False

    def choose_page(surface: str, page_key: str) -> None:
        st.session_state["active_surface"] = surface
        st.session_state["active_page"] = page_key
        if surface == "client":
            st.session_state["show_guide"] = client_page_guides.get(page_key, False)
        elif surface == "presenter":
            st.session_state["show_guide"] = (
                presenter_page_guides if presenter_only else presentation_page_guides
            ).get(page_key, False)
        else:
            st.session_state["show_guide"] = False

    def turn_to_proposal_page(page_key: str) -> None:
        choose_page("presenter", page_key)
        st.session_state["presenter_page_selector"] = page_key

    def render_proposal_turn_controls() -> None:
        surface_pages = presenter_pages if presenter_only else presentation_pages
        page_options = [key for key, _, _ in surface_pages]
        current_page = st.session_state.get("active_page")
        if current_page not in page_options:
            return

        current_index = page_options.index(current_page)
        prev_key = page_options[current_index - 1] if current_index > 0 else None
        next_key = page_options[current_index + 1] if current_index < len(page_options) - 1 else None
        with st.container(key="proposal-turn-controls"):
            st.button(
                "<",
                key=f"proposal_prev_{current_index}",
                on_click=turn_to_proposal_page,
                args=(prev_key or current_page,),
                disabled=prev_key is None,
                help="前のページへ",
                type="secondary",
                width="stretch",
            )
            st.markdown(
                f'<div class="proposal-button-index">{current_index + 1:02d}/{len(page_options):02d}</div>',
                unsafe_allow_html=True,
            )
            st.button(
                ">",
                key=f"proposal_next_{current_index}",
                on_click=turn_to_proposal_page,
                args=(next_key or current_page,),
                disabled=next_key is None,
                help="次のページへ",
                type="secondary",
                width="stretch",
            )

    with st.sidebar:
        proposal_sidebar = presenter_only or st.session_state.get("active_surface") == "presenter"
        if proposal_sidebar:
            st.markdown("### AI時代のFP&Aデータ基盤")
            st.caption("経営向けリファレンス構成")
        else:
            st.markdown(f"### {APP_NAME}")
            st.caption("AI時代のFP&Aデータ基盤")
            row_total = metadata.get("row_counts", {}).get("total")
            row_total_text = f"{int(row_total):,}" if row_total is not None else "N/A"
            st.markdown(
                f"""
                <div class="sidebar-meta">
                    <b>Current data</b><br>
                    Generated:&nbsp;{escape(format_generated_at(metadata))}<br>
                    Records:&nbsp;{escape(row_total_text)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not client_only and not presenter_only:
            surface_options = ["presenter", "internal"]
            surface_labels = {
                "presenter": "【社外用】プレゼンテーション資料",
                "internal": "【社内用】当デモに関する情報",
            }
            active_surface = st.session_state.get("active_surface")
            selected_surface = st.radio(
                "表示範囲",
                surface_options,
                index=surface_options.index(active_surface) if active_surface in surface_options else 0,
                format_func=lambda key: surface_labels[key],
                key="surface_selector",
            )
            if selected_surface != st.session_state.get("active_surface"):
                switch_surface(selected_surface)

        active_surface = st.session_state.get("active_surface")
        if active_surface == "operational":
            page_options = [key for key, _ in operational_pages]
            active_page = st.session_state.get("active_page")
            selected_page = st.radio(
                "ページ",
                page_options,
                index=page_options.index(active_page) if active_page in page_options else 0,
                format_func=lambda key: operational_page_labels[key],
                key="operational_page_selector",
            )
            if selected_page != st.session_state.get("active_page"):
                choose_page("operational", selected_page)
        elif active_surface == "presenter":
            surface_pages = presenter_pages if presenter_only else presentation_pages
            surface_page_labels = presenter_page_labels if presenter_only else presentation_page_labels
            page_options = [key for key, _, _ in surface_pages]
            active_page = st.session_state.get("active_page")
            if st.session_state.get("presenter_page_selector") not in page_options:
                st.session_state["presenter_page_selector"] = (
                    active_page if active_page in page_options else page_options[0]
                )
            selected_page = st.radio(
                "ページ",
                page_options,
                index=None,
                format_func=lambda key: surface_page_labels[key],
                key="presenter_page_selector",
            )
            if selected_page is not None and selected_page != st.session_state.get("active_page"):
                choose_page("presenter", selected_page)
        elif active_surface == "internal":
            page_options = [key for key, _, _ in internal_pages]
            active_page = st.session_state.get("active_page")
            selected_page = st.radio(
                "ページ",
                page_options,
                index=page_options.index(active_page) if active_page in page_options else 0,
                format_func=lambda key: internal_page_labels[key],
                key="internal_page_selector",
            )
            if selected_page != st.session_state.get("active_page"):
                choose_page("internal", selected_page)
        else:
            page_options = [key for key, _, _ in client_pages]
            active_page = st.session_state.get("active_page")
            selected_page = st.radio(
                "ページ",
                page_options,
                index=page_options.index(active_page) if active_page in page_options else 0,
                format_func=lambda key: client_page_labels[key],
                key="client_page_selector",
            )
            if selected_page != st.session_state.get("active_page"):
                choose_page("client", selected_page)

        current_surface = st.session_state.get("active_surface")
        current_page = st.session_state.get("active_page")
        current_surface_label = {
            "client": "本体デモ",
            "presenter": "【社外用】プレゼンテーション資料",
            "operational": "本体デモ確認",
            "internal": "【社内用】当デモに関する情報",
        }.get(current_surface, "【社外用】プレゼンテーション資料")
        current_page_label = {
            **client_page_labels,
            **presenter_page_labels,
            **operational_page_labels,
            **internal_page_labels,
        }.get(current_page, current_page)
        st.markdown(
            f"""
            <div class="nav-current">
                <b>現在の表示</b>
                <span>{escape(current_surface_label)} / {escape(current_page_label)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.session_state["active_page"]
    show_guide = bool(st.session_state.get("show_guide", False))
    current_label = next((label for key, label in operational_pages if key == page), None)
    if st.session_state.get("active_surface") == "client":
        demo_label_map = {
            "Dashboard": "ダッシュボード",
            "Variance Analysis": "差異分析",
            "Project Risk": "案件リスク",
            "AI Commentary": "AIコメント",
        }
        current_label = demo_label_map.get(page, page)
        mode_label = "本体デモ"
    elif st.session_state.get("active_surface") == "presenter":
        current_label = presenter_page_labels.get(page, page)
        mode_label = "【社外用】プレゼンテーション資料"
    elif st.session_state.get("active_surface") == "internal":
        internal_label_map = {
            "Internal Demo Guide": "説明者向けガイド",
            "Tech Architecture": "技術構成",
            "Data Explorer": "データ確認",
        }
        current_label = internal_label_map.get(page, page)
        mode_label = "【社内用】当デモに関する情報"
    else:
        mode_label = "本体デモ確認"
    surface_note = "このページは、共有・説明の目的に合わせた補助ページです。"
    if page == "Internal Demo Guide":
        surface_note = "社内・登壇者向けの台本です。公開デモでは確認用ページとして表示しています。"
    elif page in {"Dashboard", "Variance Analysis", "Project Risk", "AI Commentary"}:
        surface_note = "分析画面は、サイドバーからデモ用ガイド付き表示にも切り替えられます。"
    if page == "Problem Statement":
        render_proposal_problem_statement()
    elif page == "Target Operating Model":
        render_proposal_target_operating_model()
    elif page == "FPA Data Foundation":
        render_proposal_data_foundation()
    elif page == "AI App Architecture":
        render_proposal_ai_app_architecture()
    elif page == "System Architecture":
        render_proposal_system_architecture()
    elif page == "Approach Options":
        render_proposal_approach_options()
    elif page == "Recommended Roadmap":
        render_proposal_recommended_roadmap()
    elif page == "Assessment Proposal":
        render_proposal_assessment()
    elif page == "Internal Demo Guide":
        render_internal_demo_guide()
    elif page == "Dashboard":
        render_dashboard(
            kpis,
            data["fact_variance_drivers"],
            data["project_risk"],
            show_guide=show_guide,
            metadata=metadata,
        )
    elif page == "Variance Analysis":
        render_variance_analysis(kpis, data["fact_variance_drivers"], data["project_risk"], show_guide=show_guide)
    elif page == "Project Risk":
        render_project_risk(data["project_risk"], kpis, show_guide=show_guide)
    elif page == "AI Commentary":
        render_ai_commentary(kpis, data["fact_variance_drivers"], data["project_risk"], show_guide=show_guide)
    elif page == "Data Explorer":
        render_data_explorer(data)
    else:
        render_tech_architecture(data)

    if st.session_state.get("active_surface") == "presenter":
        render_proposal_turn_controls()


if __name__ == "__main__":
    main()
