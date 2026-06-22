from __future__ import annotations

import json
import hashlib
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


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
            content: "→";
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
        .presentation-footer {
            border-top: 1px solid rgba(15, 23, 42, 0.12);
            padding-top: 11px;
            color: #64748b;
            font-size: 0.86rem;
        }
        .stApp:has(#presentation-focus-mode) [data-testid="stSidebar"],
        .stApp:has(#presentation-focus-mode) [data-testid="stHeader"],
        .stApp:has(#presentation-focus-mode) [data-testid="stToolbar"],
        .stApp:has(#presentation-focus-mode) [data-testid="stDecoration"],
        .stApp:has(#presentation-focus-mode) footer {
            display: none !important;
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
        .stApp:has(#foundation-page) .presentation-slide {
            background: #ffffff;
        }
        .stApp:has(#foundation-page) {
            background: #f4f7fb !important;
        }
        .stApp:has(#foundation-page) .block-container,
        .stApp:has(#foundation-page) .block-container h1,
        .stApp:has(#foundation-page) .block-container h2,
        .stApp:has(#foundation-page) .block-container h3,
        .stApp:has(#foundation-page) .block-container p,
        .stApp:has(#foundation-page) .block-container li {
            color: #102033 !important;
        }
        .stApp:has(#foundation-page) .cockpit-title {
            border-bottom: 1px solid rgba(15, 23, 42, 0.14);
        }
        .stApp:has(#foundation-page) .cockpit-title .subtitle {
            color: #475569 !important;
        }
        .stApp:has(#foundation-page) .pill {
            background: #e7f7f3;
            border-color: rgba(15, 118, 110, 0.22);
            color: #0f766e;
        }
        .stApp:has(#foundation-page) .section-label {
            color: #0f172a;
            margin-top: 18px;
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
            padding: 1.05rem 1.2rem 0.85rem 1.2rem;
        }
        .stApp:has(.presentation-slide) [data-testid="stHeader"],
        .stApp:has(.presentation-slide) [data-testid="stToolbar"],
        .stApp:has(.presentation-slide) [data-testid="stDecoration"] {
            display: none !important;
        }
        .stApp:has(.presentation-slide) .block-container {
            max-width: 100vw;
            padding-top: 0.45rem !important;
            padding-bottom: 0.55rem !important;
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
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(11, 23, 34, 0.98), rgba(6, 15, 24, 0.99)),
                linear-gradient(90deg, rgba(57,197,187,0.08), transparent 42%);
            border: 1px solid rgba(57,197,187,0.30);
            border-radius: 8px;
            box-shadow: 0 22px 70px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.06);
            min-height: min(610px, calc(100vh - 190px));
            padding: clamp(16px, 1.55vw, 24px);
            gap: 10px;
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
            font-size: 1.78rem;
            line-height: 1.12;
            margin: 0 0 7px 0;
            max-width: 1180px;
        }
        .presentation-lead {
            color: #d6e7ee;
            font-size: 0.94rem;
            line-height: 1.48;
            max-width: 1120px;
        }
        .presentation-card-grid {
            gap: 7px;
            margin-top: 9px;
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
            padding: 10px 11px;
            min-height: 88px;
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
            line-height: 1.34;
            font-size: 0.78rem;
        }
        .presentation-metric-card strong {
            color: #f8fdff;
            font-size: 1rem;
            margin-bottom: 4px;
        }
        .presentation-flow {
            gap: 7px;
            margin-top: 9px;
        }
        .presentation-step {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.92), rgba(9, 20, 30, 0.94));
            border: 1px solid rgba(57,197,187,0.25);
            border-top: 3px solid #39c5bb;
            padding: 9px;
            min-height: 94px;
        }
        .presentation-step b,
        .architecture-node b {
            color: #f8fdff;
        }
        .presentation-step span {
            color: #c8dce6;
            font-size: 0.76rem;
            line-height: 1.32;
        }
        .architecture-diagram {
            gap: 7px;
            margin-top: 9px;
        }
        .architecture-node {
            border-top: 3px solid #39c5bb;
            padding: 10px;
            min-height: 94px;
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
            padding: 9px 11px;
            line-height: 1.38;
            margin-top: 9px;
        }
        .presentation-footer {
            border-top: 1px solid rgba(57,197,187,0.22);
            color: #9fb2c3;
            font-size: 0.74rem;
            padding-top: 7px;
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
            padding: 0.55rem 0.9rem 0.75rem 0.9rem;
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide {
            min-height: calc(100vh - 96px);
            padding: clamp(22px, 2.4vw, 38px);
        }
        .stApp:has(#presentation-focus-mode) .presentation-slide h2 {
            font-size: 2.25rem;
        }
        .foundation-flow-map {
            display: grid;
            grid-template-columns: 1.05fr 0.7fr 1.1fr 1fr;
            gap: 8px;
            align-items: stretch;
            margin-top: 10px;
        }
        .foundation-flow-block {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(9, 20, 30, 0.96));
            border: 1px solid rgba(57,197,187,0.25);
            border-radius: 8px;
            padding: 10px;
            min-height: 172px;
        }
        .foundation-flow-block b {
            color: #f8fdff;
            display: block;
            margin-bottom: 8px;
        }
        .foundation-flow-block span {
            display: block;
            color: #c8dce6;
            font-size: 0.78rem;
            line-height: 1.34;
        }
        .foundation-flow-items {
            display: grid;
            gap: 6px;
        }
        .foundation-flow-item {
            border: 1px solid rgba(159,178,195,0.20);
            background: rgba(255,255,255,0.035);
            border-radius: 6px;
            color: #d6e7ee;
            font-size: 0.74rem;
            padding: 5px 6px;
        }
        .foundation-flow-block.hot {
            border-color: rgba(255,176,0,0.36);
        }
        .foundation-flow-block.ai {
            border-color: rgba(183,148,244,0.38);
        }
        .foundation-quality-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 6px;
            margin-top: 10px;
        }
        .foundation-quality-card {
            background: linear-gradient(180deg, rgba(14, 29, 42, 0.94), rgba(9, 20, 30, 0.96));
            border: 1px solid rgba(57,197,187,0.25);
            border-top: 3px solid #39c5bb;
            border-radius: 8px;
            padding: 9px;
            min-height: 108px;
        }
        .foundation-quality-card b {
            color: #f8fdff;
            display: block;
            font-size: 0.88rem;
            margin-bottom: 6px;
        }
        .foundation-quality-card span {
            color: #c8dce6;
            display: block;
            font-size: 0.74rem;
            line-height: 1.28;
        }
        .foundation-quality-card em {
            color: #39c5bb;
            display: block;
            font-size: 0.7rem;
            font-style: normal;
            font-weight: 800;
            margin-bottom: 5px;
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
                padding: 22px 18px;
            }
            .presentation-slide h2 {
                font-size: 1.45rem;
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
            .architecture-node:not(:last-child)::after {
                content: "↓";
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
            .metric-value {
                white-space: normal;
            }
            .snapshot-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
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


def foundation_flow_map_html() -> str:
    columns = [
        (
            "Source data",
            "業務データ",
            ["財務・計画", "案件・調達・工程", "マスタ"],
            "",
        ),
        (
            "Quality gates",
            "品質ゲート",
            ["数字の照合", "版と粒度の固定", "案件への紐付け"],
            "hot",
        ),
        (
            "Trusted foundation",
            "FP&Aデータ基盤",
            ["共通KPI", "差異とリスク", "根拠リンク"],
            "",
        ),
        (
            "AI-driven FP&A",
            "経営説明",
            ["根拠付きコメント", "案件アクション", "意思決定"],
            "ai",
        ),
    ]
    blocks = []
    for eyebrow, title, items, variant in columns:
        item_html = "".join(f'<div class="foundation-flow-item">{escape(item)}</div>' for item in items)
        blocks.append(
            f"""
            <div class="foundation-flow-block {variant}">
                <span>{escape(eyebrow)}</span>
                <b>{escape(title)}</b>
                <div class="foundation-flow-items">{item_html}</div>
            </div>
            """
        )
    return f'<div class="foundation-flow-map">{"".join(blocks)}</div>'


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
    render_header("デモ閲覧前 / Client Preview", "Before viewing the AI-driven FP&A cockpit")

    slide = render_presentation_controls(
        "client_pre_demo",
        ["導入メッセージ", "現在のデモデータ", "押さえる3点", "デモの流れ", "前提"],
    )

    if slide == 0:
        render_presentation_slide(
            "Client Preview",
            "まず確認していただきたい状況",
            """
            <div class="presentation-lead">
            このデモは、重工業のFP&amp;Aで起こりやすい「売上は伸びているのに、利益率とキャッシュフローは悪化している」
            状況を扱います。見るべきポイントは、AIが文章を作ることそのものではありません。
            </div>
            <div class="presentation-note">
            経営KPI、差異要因、案件リスク、根拠データが同じ流れでつながり、会議で説明責任を果たせる状態になることを確認します。
            </div>
            """,
            "この後の画面は、経営説明の流れを疑似体験するための架空データデモです。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Current Snapshot",
            "最新デモデータの状態",
            presentation_snapshot_html(kpis, risk, metadata),
            "数値はすべて架空データです。実在企業の財務・案件情報は含みません。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Key Points",
            "最初に押さえていただきたい3点",
            presentation_cards_html(
                [
                    (
                        "1. 経営KPIから原因へたどれる",
                        "全社の売上、営業利益、利益率、キャッシュフローを起点に、セグメント別・要因別の差異へ掘り下げます。",
                    ),
                    (
                        "2. 案件リスクまでつながる",
                        "EAC悪化、設計変更、外注費、工程遅延など、利益悪化の原因を案件単位のアクションへつなげます。",
                    ),
                    (
                        "3. AIの前提は信頼できるデータ",
                        "AIコメントの品質は、ERP、EPM、案件EAC、調達、工程、マスタが整備されているかで決まります。",
                    ),
                ],
                columns=3,
            ),
            "AIの見せ場よりも、根拠データがつながっていることを重視して見てください。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Demo Flow",
            "デモは、経営説明の一本の流れとして見ます",
            """
            <div class="presentation-lead">
            画面を順番に紹介するのではなく、経営会議で説明するときの思考の流れに沿って見ます。
            まず全社KPIの違和感をつかみ、差異の主因を絞り、最後に案件アクションとコメントへ落とします。
            </div>
            """
            + presentation_cards_html(
                [
                    ("全社KPI", "売上は伸びているのに、利益率とキャッシュフローが悪化する矛盾を最初に捉えます。"),
                    ("差異と案件", "差異要因を追い、利益悪化につながる案件へ話を落とします。"),
                    ("説明と実装論点", "AIコメントを見たうえで、なぜデータ基盤が必要なのかへつなげます。"),
                ],
                columns=3,
            ),
            "画面を個別機能としてではなく、経営説明の一連の流れとして確認します。",
        )
    else:
        render_presentation_slide(
            "Assumptions",
            "このデモの前提",
            """
            <div class="presentation-lead">
            数値はすべて架空です。実在企業の財務・案件情報は使っていません。
            ここで見ていただきたいのは、完成済み製品ではなく、AI-driven FP&amp;Aがどのような経営説明体験になるかです。
            </div>
            <div class="presentation-note">
            画面の見た目だけでなく、どのデータがつながると説明が速く、深く、再現可能になるかを確認してください。
            </div>
            """,
            "この前提を置いたうえで、次のダッシュボード以降を確認します。",
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
            「予実差、見込差、案件リスク、根拠データがつながると、経営会議の説明品質とスピードが変わる」です。
            AIは最後の文章化を担い、価値の源泉は信頼できるFP&amp;Aデータ基盤にある、という順で説明します。
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
                    <td>AI-driven FP&amp;Aの完成イメージを先に合わせる</td>
                    <td>架空データであること、重工業の典型課題、見る順を短く説明する。</td>
                    <td>「すぐ本番で使えます」と言い切る。</td>
                </tr>
                <tr>
                    <td>Dashboard</td>
                    <td>売上が良くても利益とCFは悪化しうる</td>
                    <td>経営層が最初に見るべき矛盾を提示する。Revenue上振れとOP/CF悪化を対比する。</td>
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
                    <td>Marine &amp; OffshoreのCritical案件、Aerospace &amp; DefenseのHigh案件を例にする。</td>
                    <td>案件リスクを財務KPIと別物として扱う。</td>
                </tr>
                <tr>
                    <td>AI Commentary</td>
                    <td>経営会議向けコメント作成を速くする</td>
                    <td>金額、要因、案件、推奨アクションを含むコメントになる点を見せる。</td>
                    <td>AIが最終判断まで自動化すると受け取られる言い方。</td>
                </tr>
                <tr>
                    <td>Data Foundation</td>
                    <td>本番化の主戦場はデータ基盤</td>
                    <td>ERP、EPM、EAC、調達、工程、マスタ、品質ゲートを説明する。</td>
                    <td>LLM連携だけを導入テーマにする。</td>
                </tr>
                <tr>
                    <td>Client Follow-up</td>
                    <td>次は自社データで検証する</td>
                    <td>データ棚卸し、KPI定義、代表ユースケース、PoC範囲へ落とす。</td>
                    <td>デモ後の具体アクションを曖昧にする。</td>
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
                <span>「重工業のFP&amp;Aで、売上・利益・CF・案件リスクを一気通貫で説明するデモです。」</span>
            </div>
            <div class="briefing-step">
                <b>0:45 Dashboard</b>
                <span>売上は上振れ、利益とCFは悪化という矛盾を見せる。</span>
            </div>
            <div class="briefing-step">
                <b>1:45 Variance</b>
                <span>差異要因をウォーターフォールで分解し、主要ドライバーへ絞る。</span>
            </div>
            <div class="briefing-step">
                <b>3:00 Risk + AI</b>
                <span>案件リスクへ落とし、AIコメントで会議資料化する。</span>
            </div>
            <div class="briefing-step">
                <b>4:30 Close</b>
                <span>「実現の鍵はLLMではなく、根拠データがつながるFP&amp;A基盤です。」で締める。</span>
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
                    <td>AIは何を判断していますか</td>
                    <td>AIは承認済みのKPI、差異要因、案件リスクをもとに説明文を生成します。最終判断は人が行います。</td>
                </tr>
                <tr>
                    <td>最初に何から始めるべきですか</td>
                    <td>代表ユースケースを1つ選び、KPI定義、比較軸、必要データ、品質ゲートを棚卸しするのが現実的です。</td>
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
    render_header("デモ閲覧後 / Client Follow-up", "What to take away and how to move next")

    slide = render_presentation_controls(
        "client_post_demo",
        ["持ち帰り", "主な示唆", "確認観点", "次の進め方", "次回アジェンダ"],
    )

    if slide == 0:
        render_presentation_slide(
            "Client Follow-up",
            "デモ後に持ち帰っていただきたいこと",
            """
            <div class="presentation-lead">
            AI-driven FP&amp;Aの価値は、月次差異説明を速くするだけではありません。
            経営KPIから差異要因、案件リスク、推奨アクション、根拠データまでを同じ流れで確認できるようにし、
            経営会議での説明を再現可能にすることが本質です。
            </div>
            <div class="presentation-note">
            重要なのは、AIコメントを出すことではなく、コメントの根拠をKPI、差異要因、案件、元データまで戻れる状態にすることです。
            </div>
            """,
            "ここから自社適用の論点へ切り替えます。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Takeaways",
            "主な示唆",
            presentation_cards_html(
                [
                    (
                        "1. 差異説明は標準化できる",
                        "毎月の説明が担当者ごとのExcel作業に依存していても、KPIと差異要因を定義すれば再利用可能な説明プロセスにできます。",
                    ),
                    (
                        "2. 重工業では案件粒度が重要",
                        "売上・利益・CFの変化は、案件EAC、設計変更、調達、工程遅延とつながっています。全社KPIだけでは打ち手に届きません。",
                    ),
                    (
                        "3. AI導入はデータ基盤から始める",
                        "LLM連携より先に、データソース、KPI定義、シナリオ版管理、案件マスタ、品質ゲートを整理する必要があります。",
                    ),
                ],
                columns=3,
            ),
            "この3点がPoCの設計と本番化の優先順位を決めます。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Assessment",
            "自社適用に向けた確認観点",
            """
            <div class="presentation-lead">
            次に確認すべきことは多くありません。最初の会議テーマを選び、その説明に必要なデータとKPI定義を確認し、
            PoCで扱う範囲を小さく決めます。
            </div>
            """
            + presentation_cards_html(
                [
                    ("会議テーマ", "予実差、見込差、案件リスク、CF悪化など、最初に改善したい説明業務を一つに絞ります。"),
                    ("根拠データ", "ERP、EPM、案件EAC、調達、工程、マスタの所在と責任部門を確認します。"),
                    ("PoC範囲", "代表セグメント、代表案件、代表KPIに限定し、短期間で価値を検証します。"),
                ],
                columns=3,
            ),
            "詳細なチェックリストは、この3点が決まってから作る方が会話が進みます。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Next Steps",
            "次の進め方",
            """
            <div class="presentation-lead">
            進め方は、最初から大きな基盤を作るのではなく、説明責任が高い会議テーマから小さく始めます。
            そのテーマで必要なデータとKPIを固め、画面とAIコメントを検証し、月次運用に組み込みます。
            </div>
            """
            + presentation_flow_html(
                [
                    ("課題を絞る", "最初に変えたい会議とKPIを決めます。"),
                    ("根拠をそろえる", "必要データ、比較軸、案件粒度を固定します。"),
                    ("運用に入れる", "PoC後に月次サイクルと承認プロセスへ組み込みます。"),
                ],
                columns=3,
            ),
            "最初から全社展開を狙わず、説明責任が高い会議テーマに絞って検証します。",
        )
    else:
        render_presentation_slide(
            "Recommended Agenda",
            "推奨する次回アジェンダ",
            presentation_cards_html(
                [
                    ("ユースケース", "自社で最初に改善したい会議テーマを一つ選びます。"),
                    ("データとKPI", "必要データの所在、粒度、比較軸、責任部門を確認します。"),
                    ("進め方", "短期PoCで何を示し、どの範囲を本番設計に進めるかを決めます。"),
                ],
                columns=3,
            ),
            "AIコメント生成より前に、整えるべきデータ基盤の範囲を明確にします。",
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
        segment_base = (
            trend_df[trend_df["scenario"] == "Budget"]
            .groupby(["segment_ja", "segment_en"], observed=True)[["revenue_jpy_mn", "operating_profit_jpy_mn", "cash_flow_jpy_mn"]]
            .sum()
            .reset_index()
        )
        segment_actual = (
            trend_df[trend_df["scenario"] == "Actual"]
            .groupby(["segment_ja", "segment_en"], observed=True)[["revenue_jpy_mn", "operating_profit_jpy_mn", "cash_flow_jpy_mn"]]
            .sum()
            .reset_index()
        )
        segment_var = segment_actual.merge(segment_base, on=["segment_ja", "segment_en"], suffixes=("_actual", "_budget"))
        for col in ["revenue_jpy_mn", "operating_profit_jpy_mn", "cash_flow_jpy_mn"]:
            segment_var[f"{col}_variance"] = segment_var[f"{col}_actual"] - segment_var[f"{col}_budget"]
        melted = segment_var.melt(
            id_vars=["segment_ja"],
            value_vars=[
                "revenue_jpy_mn_variance",
                "operating_profit_jpy_mn_variance",
                "cash_flow_jpy_mn_variance",
            ],
            var_name="KPI",
            value_name="variance_jpy_mn",
        )
        melted["KPI"] = melted["KPI"].map(
            {
                "revenue_jpy_mn_variance": "売上",
                "operating_profit_jpy_mn_variance": "営業利益",
                "cash_flow_jpy_mn_variance": "キャッシュフロー",
            }
        )
        melted["variance_jpy_bn"] = melted["variance_jpy_mn"] / 1_000
        fig = px.bar(
            melted,
            x="segment_ja",
            y="variance_jpy_bn",
            color="KPI",
            barmode="group",
            color_discrete_map={"売上": "#39c5bb", "営業利益": "#ffb000", "キャッシュフロー": "#ff647c"},
            title="セグメント別差異 / Segment Variance vs Budget",
            labels={"segment_ja": "", "variance_jpy_bn": "Variance (JPY bn)"},
        )
        st.plotly_chart(style_fig(fig, 420), width="stretch")

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


def render_project_risk(risk: pd.DataFrame, show_guide: bool = True) -> None:
    render_header("Project Risk", "EAC deterioration / Loss risk / Recommended actions")
    if show_guide:
        render_page_guide(
            "この画面の見方 / 損益悪化を案件単位で先読みする",
            "重工業のFP&Aでは、全社KPIだけでなく、どの案件のEAC悪化や納期遅延が将来損益を押し下げるかを把握することが重要です。",
            [
                "CriticalとHighを優先して、赤字化リスク、EAC悪化額、見込利益率を確認します。",
                "リスクマップでは、EAC悪化額が大きく、利益率が低い案件ほど経営論点になりやすいです。",
                "推奨アクションでは、責任部署、確認観点、次の打ち手を案件単位で整理します。",
            ],
            "このデモでは、Marine & OffshoreにCritical案件、Aerospace & DefenseにHigh案件が出るように設計しています。",
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

    cards = st.columns(4)
    metrics = [
        ("Critical案件", int((scoped["risk_level"] == "Critical").sum()), "件"),
        ("High案件", int((scoped["risk_level"] == "High").sum()), "件"),
        ("赤字化リスク", int(scoped["loss_risk_flag"].sum()), "件"),
        ("EAC悪化額", scoped["eac_deterioration_jpy_mn"].sum(), "amount"),
    ]
    for col, (label, value, unit) in zip(cards, metrics):
        with col:
            display = format_amount(float(value)) if unit == "amount" else f"{int(value):,}{unit}"
            metric_card(label, display, "Project Risk Monitor", favorable=None)

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

    st.markdown('<div class="section-label">推奨アクション / Action List</div>', unsafe_allow_html=True)
    display_cols = [
        "risk_level",
        "risk_score",
        "segment_ja",
        "business_unit",
        "project_id",
        "project_name",
        "eac_deterioration_jpy_mn",
        "forecast_margin_pct",
        "schedule_delay_days",
        "loss_risk_flag",
        "primary_driver_ja",
        "owner_department",
        "recommended_action_ja",
    ]
    st.dataframe(
        scoped.sort_values(["risk_score", "eac_deterioration_jpy_mn"], ascending=[False, False])[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "eac_deterioration_jpy_mn": st.column_config.NumberColumn("EAC悪化 (JPY mn)", format="%.0f"),
            "forecast_margin_pct": st.column_config.NumberColumn("見込利益率", format="%.1f%%"),
            "loss_risk_flag": st.column_config.CheckboxColumn("赤字化リスク"),
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
    render_header("データ基盤 / Data Foundation", "AI投資の前に、経営データの分断を解く")

    slide = render_presentation_controls(
        "data_foundation",
        ["経営リスク", "3つの論点", "データの流れ", "つなぐデータ", "品質ゲート", "経営判断"],
    )

    if slide == 0:
        render_presentation_slide(
            "Executive Risk Message",
            "データが分断されたままAIを導入しても、経営会議で説明責任を果たせるAIにはならない。",
            """
            <div class="presentation-lead">
            LLMは文章を作ることはできます。しかし、売上・利益・キャッシュフロー・案件EAC・調達・工程・マスタが分断されたままでは、
            AIは「なぜ業績が悪化したのか」「どの案件に手を打つべきか」を根拠付きで説明できません。
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
            "経営層に伝える3つの論点",
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
            + foundation_flow_map_html(),
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
            + foundation_quality_gates_html(),
            "まずは代表KPIと重点案件に絞り、この3つの品質ゲートを月次運用に組み込みます。",
        )
    else:
        render_presentation_slide(
            "Executive Decisions",
            "経営層が決めること",
            """
            <div class="presentation-lead">
            経営層が決めるべきことは、AIツールの種類ではありません。
            どの会議で、どのKPIを、どの事業領域から説明可能にするかです。
            </div>
            """
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
            "This demo currently uses generated fictional Parquet files. In a real implementation, these files are replaced by a governed FP&A data mart.",
        )


def render_reference_architecture() -> None:
    st.markdown('<div id="demo-briefing-page"></div>', unsafe_allow_html=True)
    render_header(
        "リファレンス構成 / Reference Architecture",
        "AI-driven FP&Aの推奨構成を説明する",
    )

    slide = render_presentation_controls(
        "reference_architecture",
        ["全体像", "推奨構成", "構成要素"],
    )

    if slide == 0:
        render_presentation_slide(
            "Reference View",
            "AI-driven FP&Aは、画面、データ、AI、統制を一体で設計する",
            """
            <div class="presentation-lead">
            最初に決めるべきことは、LLMの製品名ではありません。
            どの会議で、どのKPIを、どの根拠データに戻れる状態で説明するかを定義し、その上にAIを置きます。
            </div>
            """
            + architecture_diagram_html(
                [
                    ("Business", "会議テーマ", "最初に変えたい経営説明を決める。"),
                    ("Data", "根拠データ", "財務、案件、マスタを同じ粒度でつなぐ。"),
                    ("AI & Control", "AIと統制", "コメント生成、権限、ログ、承認を同じ運用に入れる。"),
                ]
            ),
            "このデモは画面デモですが、本番化の焦点はFP&Aデータ基盤、AI実装、統制運用をつなぐことです。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Recommended Architecture",
            "推奨構成: 全社AI基盤とつなぐFP&A構成",
            architecture_diagram_html(
                [
                    ("Data", "全社データ基盤", "DWH、MDM、リネージを活用し、FP&Aデータマートへ渡す。"),
                    ("AI", "AI共通基盤", "モデルゲートウェイ、RAG、評価、監査ログを共通化する。"),
                    ("Workflow", "業務ポータル", "FP&A Cockpitを入口に、承認、通知、会議アクションへつなぐ。"),
                ]
            )
            + presentation_cards_html(
                [
                    ("設計の考え方", "FP&Aだけで閉じず、全社データ基盤とAI共通基盤に接続できる形で作る。"),
                    ("最初の範囲", "代表KPI、代表会議、重点案件に絞り、FP&Aデータマートの最小構成から始める。"),
                ],
                columns=2,
            ),
            "短期PoCでも、この将来構成に接続できる形でKPI定義とデータ粒度を決めます。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Building Blocks",
            "主要な構成要素",
            """
            <div class="presentation-lead">
            構成要素は多く見えますが、説明するときは三層で十分です。
            根拠データを整え、AIがその根拠を読めるようにし、会議とアクションに戻す。
            </div>
            """
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
            "このデモは、架空データを使ったFP&A差異分析コックピットです",
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
        render_presentation_slide(
            "Architecture / 04",
            "分析処理は、読み込み、KPI生成、フィルタ、可視化に分けています",
            """
            <div class="presentation-lead">
            実装上は多くの関数に分かれていますが、説明としては「読み込む」「意味づける」「見せる」の三段階です。
            本番化する場合は、読み込みと意味づけの部分を外部のデータ基盤やセマンティックレイヤーへ寄せます。
            </div>
            """
            + architecture_diagram_html(
                [
                    ("Load", "読み込む", "Parquet/JSONを読み、キャッシュで画面操作を軽くする。"),
                    ("Semantic", "意味づける", "売上、利益、CF、差異、案件リスクを同じ定義で計算する。"),
                    ("Experience", "見せる", "PlotlyとStreamlitで、会議中に論点を切り替えられるようにする。"),
                ]
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
        ("Client Pre-Demo", "デモ閲覧前", False),
        ("Data Foundation", "データ基盤", False),
        ("Reference Architecture", "リファレンス構成", False),
        ("Client Post-Demo", "デモ閲覧後", False),
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
    presenter_pages = presentation_pages + internal_pages
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
        st.session_state["active_page"] = "Dashboard" if client_only else "Client Pre-Demo"
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
            active_page = "Client Pre-Demo"
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
            st.session_state["active_page"] = "Client Pre-Demo"
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
            st.session_state["active_page"] = current_page if current_page in valid_pages else "Client Pre-Demo"
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

    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(COMPANY_NAME)
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
                "presenter": "プレゼンテーション資料",
                "internal": "当デモに関する情報",
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
            selected_page = st.radio(
                "ページ",
                page_options,
                index=page_options.index(active_page) if active_page in page_options else 0,
                format_func=lambda key: surface_page_labels[key],
                key="presenter_page_selector",
            )
            if selected_page != st.session_state.get("active_page"):
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
            "presenter": "プレゼンテーション資料",
            "operational": "本体デモ確認",
            "internal": "当デモに関する情報",
        }.get(current_surface, "プレゼンテーション資料")
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
        mode_label = "プレゼンテーション資料"
    elif st.session_state.get("active_surface") == "internal":
        internal_label_map = {
            "Internal Demo Guide": "説明者向けガイド",
            "Tech Architecture": "技術構成",
            "Data Explorer": "データ確認",
        }
        current_label = internal_label_map.get(page, page)
        mode_label = "当デモに関する情報"
    else:
        mode_label = "本体デモ確認"
    surface_note = "このページは、共有・説明の目的に合わせた補助ページです。"
    if page == "Internal Demo Guide":
        surface_note = "社内・登壇者向けの台本です。公開デモでは確認用ページとして表示しています。"
    elif page in {"Dashboard", "Variance Analysis", "Project Risk", "AI Commentary"}:
        surface_note = "分析画面は、サイドバーからデモ用ガイド付き表示にも切り替えられます。"
    if page == "Client Pre-Demo":
        render_client_pre_demo(kpis, data["project_risk"], metadata)
    elif page == "Internal Demo Guide":
        render_internal_demo_guide()
    elif page == "Client Post-Demo":
        render_client_post_demo()
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
        render_project_risk(data["project_risk"], show_guide=show_guide)
    elif page == "AI Commentary":
        render_ai_commentary(kpis, data["fact_variance_drivers"], data["project_risk"], show_guide=show_guide)
    elif page == "Data Foundation":
        render_data_foundation(data)
    elif page == "Reference Architecture":
        render_reference_architecture()
    elif page == "Data Explorer":
        render_data_explorer(data)
    else:
        render_tech_architecture(data)


if __name__ == "__main__":
    main()
