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
            font-size: clamp(1.7rem, 2.7vw, 2.55rem);
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
            font-size: clamp(1.25rem, 2.2vw, 1.85rem);
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
            font-size: clamp(1.8rem, 3.2vw, 2.85rem);
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
    index = int(st.session_state.get(index_key, 0))
    index = max(0, min(index, len(titles) - 1))

    controls = st.columns([1.15, *([0.48] * len(titles)), 1.15], gap="small")
    with controls[0]:
        if st.button("← 前へ", key=f"{deck_key}_prev", disabled=index == 0, width="stretch"):
            index = max(0, index - 1)
    for slide_index, _title in enumerate(titles):
        with controls[slide_index + 1]:
            if st.button(
                str(slide_index + 1),
                key=f"{deck_key}_jump_{slide_index}",
                type="primary" if slide_index == index else "secondary",
                width="stretch",
            ):
                index = slide_index
    with controls[-1]:
        if st.button("次へ →", key=f"{deck_key}_next", disabled=index == len(titles) - 1, width="stretch"):
            index = min(len(titles) - 1, index + 1)

    st.session_state[index_key] = index
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
    render_header("デモ閲覧前 / Client Preview", "Before viewing the AI FP&A cockpit")

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
            "デモで見る流れ",
            presentation_flow_html(
                [
                    ("1. Dashboard", "売上上振れと利益・CF悪化が同時に起きている全社状況を確認します。"),
                    ("2. Variance Analysis", "予算、見込、前年などの比較軸で、差異の主要因を分解します。"),
                    ("3. Project Risk", "利益悪化につながるCritical/High案件を特定します。"),
                    ("4. AI Commentary", "分析結果を経営会議で使える日本語コメントに変換します。"),
                    ("5. Data Foundation", "AI活用に必要なデータ、品質管理、導入ステップを確認します。"),
                    ("6. Reference Architecture", "複数の構築案と、本番化に向けた導入アプローチを確認します。"),
                ],
                columns=3,
            ),
            "画面を個別機能としてではなく、経営説明の一連の流れとして確認します。",
        )
    else:
        render_presentation_slide(
            "Assumptions",
            "このデモの前提",
            presentation_table_html(
                ["項目", "内容"],
                [
                    ["データ", "すべて架空データです。実在企業の財務・案件情報は使っていません。"],
                    ["目的", "完成済み製品の紹介ではなく、AIを使ったFP&A差異分析の完成イメージと実装論点を共有することです。"],
                    ["見てほしい観点", "画面の見た目だけでなく、どのデータがつながると経営説明が速く・深く・再現可能になるかを確認してください。"],
                ],
            ),
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
                    <td>AI FP&amp;Aの完成イメージを先に合わせる</td>
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
            AI FP&amp;Aの価値は、月次差異説明を速くするだけではありません。
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
            presentation_table_html(
                ["論点", "確認すること", "初期アウトプット"],
                [
                    ["ユースケース", "予実差、見込差、案件リスク、CF悪化など、最初に解きたい会議テーマを1つ選ぶ。", "PoC対象KPIと対象会議"],
                    ["データソース", "ERP、EPM、案件EAC、調達、工程、為替、マスタの所在と責任部門を確認する。", "データソース棚卸し"],
                    ["KPI定義", "売上、営業利益、利益率、CF、EACなどの計算式、粒度、比較軸を固定する。", "KPI・差異要因定義書"],
                    ["品質ゲート", "実績照合、シナリオ版管理、案件ID紐づけ、勘定科目マッピング、根拠追跡を確認する。", "データ品質チェックリスト"],
                    ["PoC範囲", "代表セグメント、代表案件、代表KPIに絞り、短期間で価値を検証する。", "PoCスコープと成功条件"],
                ],
            ),
            "この表を次回ディスカッションの確認リストとして使います。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Next Steps",
            "次の進め方",
            presentation_flow_html(
                [
                    ("1. 課題選定", "経営会議、予実会議、見込更新など、最初に改善したい説明業務を選びます。"),
                    ("2. データ棚卸し", "必要データの所在、粒度、更新頻度、品質課題を確認します。"),
                    ("3. KPI設計", "比較軸、KPI、差異要因、案件リスクの定義をそろえます。"),
                    ("4. PoC構築", "代表領域に絞り、ダッシュボード、差異分解、AIコメントを検証します。"),
                    ("5. 業務定着", "月次サイクル、承認プロセス、データ品質運用へ組み込みます。"),
                ]
            ),
            "最初から全社展開を狙わず、説明責任が高い会議テーマに絞って検証します。",
        )
    else:
        render_presentation_slide(
            "Recommended Agenda",
            "推奨する次回アジェンダ",
            presentation_cards_html(
                [
                    ("1. 代表ユースケース", "自社で最初に改善したい会議テーマを1つ選びます。"),
                    ("2. 必要データ", "ERP、EPM、案件EAC、調達、工程、マスタの所在と粒度を確認します。"),
                    ("3. KPI定義", "比較軸、計算式、更新頻度、責任部門を揃えます。"),
                    ("4. 品質課題", "欠損、版ずれ、案件ID不一致、勘定科目マッピングの課題を整理します。"),
                    ("5. アーキテクチャ案", "Quick MVP、Governed FP&A、Enterprise AI、Vendor-ledのどれを軸にするかを決めます。"),
                ],
                columns=5,
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


def render_foundation_sankey(height: int = 460) -> None:
    labels = [
        "ERP実績",
        "EPM予算・見込",
        "案件EAC",
        "調達・外注",
        "工程・納期",
        "マスタ",
        "手元Excel連結",
        "根拠を追えないAIコメント",
        "Ingestion & Quality Gates",
        "Trusted FP&A Data Foundation",
        "KPI / Variance / Project Risk Layer",
        "根拠付きAIコメント",
        "案件アクション",
        "説明責任の不足",
        "経営会議で意思決定",
    ]
    node_colors = [
        "#60a5fa",
        "#60a5fa",
        "#60a5fa",
        "#60a5fa",
        "#60a5fa",
        "#9fb2c3",
        "#ef4444",
        "#dc2626",
        "#f59e0b",
        "#39c5bb",
        "#76d275",
        "#b794f4",
        "#ffb000",
        "#991b1b",
        "#d9fffb",
    ]
    source = [
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        6, 7,
        8, 9, 10, 10, 11, 12,
    ]
    target = [
        6, 6, 6, 6, 6, 6,
        8, 8, 8, 8, 8, 8,
        7, 13,
        9, 10, 11, 12, 14, 14,
    ]
    value = [
        1, 1, 1, 1, 1, 1,
        4, 4, 3, 2, 2, 2,
        6, 6,
        17, 17, 11, 6, 11, 6,
    ]
    red = "rgba(220,38,38,0.30)"
    teal = "rgba(57,197,187,0.25)"
    link_colors = [
        red, red, red, red, red, red,
        teal, teal, teal, teal, teal, teal,
        "rgba(220,38,38,0.42)",
        "rgba(153,27,27,0.48)",
        "rgba(245,158,11,0.36)",
        "rgba(57,197,187,0.38)",
        "rgba(118,210,117,0.34)",
        "rgba(255,176,0,0.30)",
        "rgba(183,148,244,0.36)",
        "rgba(255,176,0,0.32)",
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "pad": 18,
                    "thickness": 16,
                    "line": {"color": "rgba(237,246,249,0.45)", "width": 0.6},
                    "label": labels,
                    "color": node_colors,
                },
                link={
                    "source": source,
                    "target": target,
                    "value": value,
                    "color": link_colors,
                },
            )
        ]
    )
    fig.update_layout(
        title="Disconnected facts produce unverifiable AI comments; governed data produces accountable management commentary",
        height=height,
        paper_bgcolor="#071016",
        plot_bgcolor="#071016",
        font={"color": "#edf6f9", "size": 12},
        margin={"l": 12, "r": 12, "t": 54, "b": 18},
    )
    st.plotly_chart(fig, width="stretch")


def render_foundation_quality_matrix() -> None:
    z = [
        [3, 1, 1, 3, 3, 3],
        [2, 3, 1, 1, 3, 3],
        [1, 2, 3, 1, 2, 3],
        [2, 1, 2, 3, 2, 3],
        [1, 1, 2, 1, 3, 3],
        [1, 2, 3, 3, 2, 3],
    ]
    text = [
        ["必須", "参照", "参照", "必須", "必須", "必須"],
        ["重要", "必須", "参照", "参照", "必須", "必須"],
        ["参照", "重要", "必須", "参照", "重要", "必須"],
        ["重要", "参照", "重要", "必須", "重要", "必須"],
        ["参照", "参照", "重要", "参照", "必須", "必須"],
        ["参照", "重要", "必須", "必須", "重要", "必須"],
    ]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=["照合", "版管理", "案件ID", "勘定分類", "時間粒度", "リネージ"],
                y=["ERP実績", "予算・見込", "案件EAC", "調達・外注", "工程・納期", "マスタ"],
                text=text,
                texttemplate="%{text}",
                textfont={"color": "#0f172a", "size": 12},
                colorscale=[
                    [0.0, "#cbd5e1"],
                    [0.34, "#7dd3fc"],
                    [0.67, "#5eead4"],
                    [1.0, "#f59e0b"],
                ],
                showscale=False,
                hovertemplate="Data: %{y}<br>Gate: %{x}<br>Priority: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Data domain x quality gate priority",
        template="plotly_white",
        height=380,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        font={"color": "#0f172a", "size": 12},
        title_font={"color": "#0f172a", "size": 15},
        margin={"l": 12, "r": 12, "t": 54, "b": 34},
    )
    fig.update_xaxes(
        tickfont={"color": "#334155", "size": 12},
        title_font={"color": "#334155"},
        linecolor="rgba(15, 23, 42, 0.22)",
        gridcolor="rgba(15, 23, 42, 0.08)",
    )
    fig.update_yaxes(
        tickfont={"color": "#334155", "size": 12},
        title_font={"color": "#334155"},
        linecolor="rgba(15, 23, 42, 0.22)",
        gridcolor="rgba(15, 23, 42, 0.08)",
    )
    st.plotly_chart(fig, width="stretch")


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
            """,
        )
        render_foundation_sankey(height=410)
    elif slide == 3:
        render_presentation_slide(
            "Data Domains",
            "つなぐべきデータ",
            presentation_table_html(
                ["Domain", "Source", "Why it matters"],
                [
                    ["Actuals / 実績", "ERP, GL, sub-ledger", "予算・見込との差異説明の起点"],
                    ["Budget & Forecast / 予算・見込", "EPM, planning system", "比較軸の正本"],
                    ["Project EAC / 案件見積総原価", "Project control, PMO", "赤字化リスクと設計変更影響の中核"],
                    ["Procurement / 調達・外注", "Procurement, supplier contracts", "材料費・外注費・長納期部品の影響"],
                    ["Schedule / 工程・納期", "Project schedule, milestones", "計上時期、検収遅れ、CF悪化"],
                    ["Master Data / マスタ", "MDM, ERP master", "セグメント、案件、勘定科目、顧客の定義"],
                ],
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
            """,
        )
        render_foundation_quality_matrix()
    else:
        render_presentation_slide(
            "Executive Decisions",
            "経営層が決めること",
            presentation_table_html(
                ["Theme", "Executive question", "Initial scope"],
                [
                    ["KPI", "どの経営指標を、AIで説明責任を果たせる状態にするか", "売上、営業利益、営業利益率、キャッシュフロー"],
                    ["Meeting", "どの会議の説明品質と意思決定スピードを変えるか", "月次経営会議、予実会議、見込更新会議"],
                    ["Business area", "どのセグメント・案件タイプから始めるか", "赤字化リスクが高い案件群、EAC悪化が大きい事業"],
                ],
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
        "リファレンスアーキテクチャ / Reference Architecture",
        "AI-enabled FP&Aの構築案と導入アプローチ",
    )

    slide = render_presentation_controls(
        "reference_architecture",
        [
            "全体像",
            "案A: Quick MVP",
            "案B: Governed FP&A",
            "案C: Enterprise AI",
            "案D: Vendor-led",
            "選び方",
            "導入アプローチ",
        ],
    )

    if slide == 0:
        render_presentation_slide(
            "Architecture Options",
            "AI入りFP&Aに唯一の正解はない。目的、データ成熟度、統制要件で構成を選ぶ。",
            """
            <div class="presentation-lead">
            クライアントがこのデモのようなAI入りFP&amp;Aを構築する場合、最初に決めるべきことは「どのLLMを使うか」ではありません。
            どの会議で、どのKPIを、どの根拠データに戻れる状態で説明するかを決め、その成熟度に合うアーキテクチャを選びます。
            </div>
            """
            + presentation_cards_html(
                [
                    ("案A: Quick MVP", "既存Excel、EPM出力、BIデータを使い、短期間で画面とAIコメントの業務価値を検証する。"),
                    ("案B: Governed FP&A", "FP&Aデータマートと品質ゲートを整備し、説明責任を持てるAIコメント基盤を作る。"),
                    ("案C: Enterprise AI", "全社データ基盤、セマンティック層、AIエージェントを組み合わせ、複数業務へ拡張する。"),
                    ("案D: Vendor-led", "ERP/EPM/BIベンダーのAI機能を起点にし、構築負荷を下げながら標準機能を活用する。"),
                ],
                columns=4,
            ),
            "多くの場合は、案Aで業務仮説を検証し、案Bを本命アーキテクチャとして設計するのが現実的です。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Option A",
            "案A: Quick MVP / 既存データを使った短期検証型",
            presentation_table_html(
                ["Layer", "Reference design", "Key point"],
                [
                    ["Data sources", "EPM出力、Excel、既存BIデータ、案件リスト", "まずは代表KPIと代表案件に絞る"],
                    ["Data processing", "Python / SQL / lightweight data mart", "再現可能な集計ロジックだけは固定する"],
                    ["AI commentary", "ルール + プロンプトテンプレート + 人のレビュー", "AIの自由回答より、根拠付き定型コメントを優先する"],
                    ["Experience", "Streamlit / Power BI prototype", "経営会議の説明順で画面を作る"],
                ],
            )
            + presentation_cards_html(
                [
                    ("向いているケース", "4〜8週間で価値仮説を見せたい。現行データ基盤の整備を待たず、業務側の反応を取りたい。"),
                    ("主なリスク", "Excel依存のままだと、データの版、根拠、更新責任が曖昧になり、本番運用へ移行しにくい。"),
                    ("成功条件", "PoCでもKPI定義、比較軸、データ更新日、AIコメントの根拠を明記する。"),
                ],
                columns=3,
            ),
            "この案は本番アーキテクチャではなく、何を作るべきかを決めるための検証アーキテクチャです。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Option B",
            "案B: Governed FP&A Data Mart + AI Commentary Service",
            presentation_table_html(
                ["Layer", "Reference design", "Key point"],
                [
                    ["Source systems", "ERP、EPM、Project EAC、調達、工程、マスタ", "会計、計画、案件を同じ粒度でつなぐ"],
                    ["Data foundation", "FP&Aデータマート、品質ゲート、シナリオ版管理", "実績照合、案件ID、勘定科目、比較軸を統制する"],
                    ["Semantic layer", "KPI定義、差異要因定義、権限付きメトリクス", "AIが読む前に、業務定義を固定する"],
                    ["AI layer", "コメント生成API、根拠引用、プロンプト管理、出力ログ", "説明文と根拠データをセットで管理する"],
                    ["Experience", "FP&A Cockpit、BI、Teams/Slack通知、会議資料出力", "会議前確認、会議中説明、会議後アクションに使う"],
                ],
            )
            + presentation_cards_html(
                [
                    ("標準推奨", "このデモの本番化に最も近い構成。FP&Aの説明責任とAI活用を両立しやすい。"),
                    ("主なリスク", "データ統制を軽く見ると、AIの品質ではなく基礎データの不整合が問題になる。"),
                    ("成功条件", "最初から全社ではなく、代表KPI、代表セグメント、代表会議で始める。"),
                ],
                columns=3,
            ),
            "クライアントに提案する本命案としては、この構成を基準に置くのが最も説明しやすいです。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Option C",
            "案C: Enterprise Lakehouse + Agentic FP&A Assistant",
            presentation_table_html(
                ["Layer", "Reference design", "Key point"],
                [
                    ["Enterprise data", "Lakehouse / warehouse、MDM、data catalog、lineage", "FP&A以外の業務データにも広げる前提"],
                    ["Analytics", "Feature store、semantic model、forecasting、scenario simulation", "予測、シナリオ、異常検知を組み込む"],
                    ["AI assistant", "RAG、tool calling、agent workflow、approval workflow", "AIが集計、要因探索、コメント草案、アクション提示を支援する"],
                    ["Governance", "RBAC、監査ログ、model gateway、prompt/output evaluation", "権限、監査、評価を全社標準で管理する"],
                    ["Experience", "FP&A portal、chat UI、BI embedded、workflow integration", "会話UIだけでなく業務ワークフローに埋め込む"],
                ],
            )
            + presentation_cards_html(
                [
                    ("向いているケース", "全社データ基盤があり、FP&A以外にもAI分析基盤を横展開したい。"),
                    ("主なリスク", "初期投資と設計範囲が大きい。業務ユースケースが曖昧だと大規模基盤だけが先行する。"),
                    ("成功条件", "案B相当のFP&Aデータ定義を先に固め、その上にエージェントを載せる。"),
                ],
                columns=3,
            ),
            "大企業向けの将来像として有効ですが、最初のPoCからこの構成を全部作る必要はありません。",
        )
    elif slide == 4:
        render_presentation_slide(
            "Option D",
            "案D: Vendor-led / ERP・EPM・BIベンダー機能を起点にする",
            presentation_table_html(
                ["Layer", "Reference design", "Key point"],
                [
                    ["Core platforms", "既存ERP、EPM、BI、planning toolのAI機能", "標準機能でできる範囲を先に使う"],
                    ["Data model", "ベンダー標準データモデル + 必要最小限の拡張", "独自実装を増やしすぎない"],
                    ["AI capability", "説明文生成、異常検知、ドリルダウン、レポート作成支援", "ベンダー機能の透明性と監査性を確認する"],
                    ["Integration", "既存認証、権限、ワークフロー、レポート配布", "運用負荷と変更管理を抑える"],
                ],
            )
            + presentation_cards_html(
                [
                    ("向いているケース", "既存EPM/BIへの投資が大きく、標準機能を優先して活用したい。"),
                    ("主なリスク", "ベンダー標準に寄るため、重工業固有の案件EAC、工程、調達の説明が弱くなる場合がある。"),
                    ("成功条件", "標準機能で足りる領域と、独自データ基盤が必要な領域を切り分ける。"),
                ],
                columns=3,
            ),
            "構築負荷を下げる選択肢ですが、説明責任に必要なデータ連携が標準機能だけで足りるかを確認します。",
        )
    elif slide == 5:
        render_presentation_slide(
            "Decision Criteria",
            "どの案を選ぶべきか",
            presentation_table_html(
                ["観点", "案A: Quick MVP", "案B: Governed FP&A", "案C: Enterprise AI", "案D: Vendor-led"],
                [
                    ["初期スピード", "最速。短期デモ向き", "中。基礎設計が必要", "遅い。全社設計が必要", "中。標準機能次第"],
                    ["説明責任", "弱い。人の補完が必要", "強い。根拠管理しやすい", "強いが設計が重い", "機能とログの透明性次第"],
                    ["拡張性", "限定的", "FP&A領域で高い", "全社横展開に強い", "ベンダー範囲に依存"],
                    ["技術負荷", "低〜中", "中", "高", "低〜中"],
                    ["向いている場面", "価値仮説検証", "本番化の標準案", "全社AI基盤化", "既存投資活用"],
                ],
            )
            + """
            <div class="presentation-note">
            推奨判断: 短期では案Aで経営・FP&amp;A部門の反応を確認し、本番化は案Bを軸にする。
            全社AI基盤が既にある企業では案Cへ拡張し、既存EPM投資が大きい企業では案Dとの併用を検討します。
            </div>
            """,
            "選択肢を1つに固定せず、PoC、本番化、全社展開で段階的に組み合わせます。",
        )
    else:
        render_presentation_slide(
            "Implementation Approach",
            "導入アプローチ: 会議テーマを起点に、データ基盤とAIを段階的に広げる",
            presentation_flow_html(
                [
                    ("0. 課題選定", "対象会議、KPI、説明したい差異、代表セグメントを決める。"),
                    ("1. データ棚卸し", "ERP、EPM、EAC、調達、工程、マスタの所在、粒度、責任者を確認する。"),
                    ("2. MVP構築", "代表KPIと代表案件に絞り、ダッシュボード、差異分解、AIコメントを作る。"),
                    ("3. 統制設計", "KPI定義、品質ゲート、権限、ログ、承認フロー、根拠追跡を設計する。"),
                    ("4. 本番化", "データ連携を自動化し、月次FP&Aプロセスと会議運営に組み込む。"),
                ]
            )
            + presentation_cards_html(
                [
                    ("成果物", "KPI定義書、データソース棚卸し、To-Beアーキテクチャ、PoC画面、AIコメント評価観点、運用設計。"),
                    ("ガバナンス", "権限、監査ログ、プロンプト管理、出力レビュー、根拠リンク、モデル変更管理を初期から設計する。"),
                    ("次の一手", "クライアントの代表ユースケースを1つ選び、案Aで検証しながら案Bの本番設計を並行して進める。"),
                ],
                columns=3,
            ),
            "AI FP&Aはツール導入ではなく、経営説明プロセスとデータ基盤を再設計する取り組みです。",
        )


def render_tech_architecture(data: dict[str, Any]) -> None:
    metadata = data.get("metadata", {})
    row_counts = metadata.get("row_counts", {})
    generated_at = format_generated_at(metadata)
    total_rows = row_count_text(metadata)

    render_header("技術構成 / Tech Architecture", "社内説明用: デモの構成、処理、公開運用、次の拡張を説明する")
    st.caption("このページは社内・説明者向けです。クライアント向けダッシュボードURLには表示されません。")

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
            presentation_metric_cards_html(
                [
                    ("App", "Streamlit", "Pythonだけで画面、集計、可視化を構成"),
                    ("Data", total_rows, f"生成日時: {generated_at}"),
                    ("AI", "No external API", "デモではAPIキーや外部AIサービスを使わない"),
                    ("Deploy", "Streamlit Cloud", "GitHub上のエントリーポイント別に公開"),
                ],
                columns=4,
            )
            + presentation_flow_html(
                [
                    ("1. データ生成", "scripts/generate_demo_data.py が完全な架空データを作成"),
                    ("2. 保存", "ParquetとJSONでリポジトリ内の data フォルダに保持"),
                    ("3. 読み込み", "Streamlit cache で高速にロード"),
                    ("4. 分析", "pandasでKPI、差異、案件リスクを集計"),
                    ("5. 体験", "PlotlyとStreamlitでダッシュボードと説明資料を表示"),
                ],
                columns=5,
            ),
            "本質は、AI単体ではなく、信頼できるFP&Aデータ基盤と説明プロセスを体験させることです。",
        )
    elif slide == 1:
        render_presentation_slide(
            "Architecture / 02",
            "URLを分けるために、入口ファイルを3つに分けています",
            presentation_table_html(
                ["入口ファイル", "公開対象", "主な利用者", "表示されるページ", "用途"],
                [
                    [
                        "client_app.py",
                        "https://heavy-industry-ai-demo-client.streamlit.app/",
                        "クライアント",
                        "ダッシュボード、差異分析、案件リスク、AIコメント",
                        "相手が触る本体デモ。説明者向け資料は出さない。",
                    ],
                    [
                        "presenter_app.py",
                        "社内・説明者向けURL",
                        "当社側の説明者",
                        "閲覧前/後の説明、データ基盤、技術構成、データ確認",
                        "商談や社内説明で投影する支援資料。",
                    ],
                    [
                        "app.py",
                        "フルアプリURL",
                        "開発・QA担当",
                        "全ページ",
                        "動作確認、保守、公開前チェック用。",
                    ],
                ],
            )
            + presentation_cards_html(
                [
                    ("なぜ分けるか", "クライアントが触るURLには、本体デモ以外の説明者向けページを出さないため。"),
                    ("アクセス制限との違い", "認証ではなく、Streamlit Cloudのデプロイ単位を分けて見せる範囲を制御している。"),
                    ("運用の利点", "同じコードベースを使いながら、URLごとに体験を変えられる。"),
                ],
                columns=3,
            ),
            "クライアント向けURLはダッシュボード専用、社内向けURLは説明資料専用、フルアプリは保守用です。",
        )
    elif slide == 2:
        render_presentation_slide(
            "Architecture / 03",
            "データは全て架空で、Parquetを中心に軽量に配布しています",
            presentation_table_html(
                ["ファイル", "役割", "粒度", "現在行数"],
                [
                    [
                        "data/dim_projects.parquet",
                        "案件、セグメント、顧客、地域、契約形態などのマスタ",
                        "1行 = 1案件",
                        str(row_counts.get("dim_projects", "N/A")),
                    ],
                    [
                        "data/fact_finance.parquet",
                        "Actual/Budget/Forecastの売上、利益、キャッシュフロー",
                        "1行 = 案件 x 月 x シナリオ",
                        str(row_counts.get("fact_finance", "N/A")),
                    ],
                    [
                        "data/fact_variance_drivers.parquet",
                        "差異要因を価格、数量、調達、工程、為替などに分解",
                        "1行 = 案件 x 月 x KPI x 要因",
                        str(row_counts.get("fact_variance_drivers", "N/A")),
                    ],
                    [
                        "data/project_risk.parquet",
                        "案件リスク、EAC乖離、遅延、調達圧力、品質問題",
                        "1行 = 1案件の最新リスク状態",
                        str(row_counts.get("project_risk", "N/A")),
                    ],
                    [
                        "data/demo_metadata.json",
                        "生成日時、総行数、生成条件、デモ前提",
                        "1ファイル = データセット説明",
                        "metadata",
                    ],
                ],
            )
            + presentation_cards_html(
                [
                    ("配布しやすい", "DB接続なしでStreamlit Cloudに置けるため、デモ公開が速い。"),
                    ("説明しやすい", "実データではなく架空データなので、守秘情報を含めずに商談で使える。"),
                    ("本番化しやすい", "本番ではParquet部分をDWHやデータマートに置き換える想定。"),
                ],
                columns=3,
            ),
            "このデモではファイル配布を優先し、本番ではERP/EPM/案件管理システムからの連携に置き換えます。",
        )
    elif slide == 3:
        render_presentation_slide(
            "Architecture / 04",
            "分析処理は、読み込み、KPI生成、フィルタ、可視化に分けています",
            presentation_table_html(
                ["層", "主な処理", "実装上の役割", "本番化時の置き換え先"],
                [
                    [
                        "Load",
                        "load_data()",
                        "Parquet/JSONを読み、欠損時はセットアップ案内を表示する。",
                        "DWH、Lakehouse、API、権限付きデータサービス",
                    ],
                    [
                        "Cache",
                        "@st.cache_data",
                        "Streamlit上で同じデータ読み込みを再利用し、画面操作を軽くする。",
                        "データ更新時刻、パーティション、クエリキャッシュ管理",
                    ],
                    [
                        "Semantic layer",
                        "build_project_month_kpis()",
                        "売上、営業利益、利益率、CF、差異を同じ定義で計算する。",
                        "KPI定義テーブル、dbt、BIセマンティックレイヤー",
                    ],
                    [
                        "Interaction",
                        "期間、セグメント、KPI、比較軸のフィルタ",
                        "説明者が会議中に論点を切り替えられるようにする。",
                        "ロール別ビュー、案件権限、保存済みシナリオ",
                    ],
                    [
                        "Visual",
                        "Plotly charts",
                        "カード、トレンド、ウォーターフォール、リスク散布図を表示する。",
                        "正式BI、埋め込みアプリ、経営会議ポータル",
                    ],
                ],
            ),
            "StreamlitはMVPの画面実装に適しており、本番ではデータ処理と権限制御を外部基盤に寄せます。",
        )
    elif slide == 4:
        render_presentation_slide(
            "Architecture / 05",
            "AIコメントは、現時点では外部APIを使わない決定的な生成です",
            presentation_flow_html(
                [
                    ("入力", "選択された期間、KPI、セグメント、差異要因、案件リスク"),
                    ("要約", "差異の方向、金額影響、利益率/CF影響をpandasで集計"),
                    ("論点抽出", "価格、調達、工程、為替、品質、案件遅延などの上位要因を抽出"),
                    ("文章化", "経営会議で使える日本語コメントの型に埋め込む"),
                    ("出力", "要旨、主因、確認事項、次アクションを提示"),
                ],
                columns=5,
            )
            + presentation_cards_html(
                [
                    ("デモでAPIを使わない理由", "APIキー管理、従量課金、通信失敗を避け、公開デモを安定させるため。"),
                    ("現在の限界", "LLMの柔軟な言い換えや質疑応答は未実装。根拠は集計済みデータに限定される。"),
                    ("本番での拡張", "OpenAI等を使う場合は、Secrets管理、プロンプト管理、根拠ログ、承認フローを追加する。"),
                ],
                columns=3,
            ),
            "クライアントには『AIの前に、説明可能なデータとロジックが必要』というメッセージを伝えます。",
        )
    elif slide == 5:
        render_presentation_slide(
            "Architecture / 06",
            "画面は、クライアント操作用と説明者支援用に分けています",
            presentation_table_html(
                ["区分", "ページ", "目的", "クライアント向けURLでの表示"],
                [
                    ["本体デモ", "Dashboard", "経営トップ向けに全社KPIと異常値をつかむ。", "表示する"],
                    ["本体デモ", "Variance Analysis", "差異要因をドリルダウンし、説明可能な粒度に分解する。", "表示する"],
                    ["本体デモ", "Project Risk", "利益悪化や納期遅延につながる案件を特定する。", "表示する"],
                    ["本体デモ", "AI Commentary", "会議資料に使う日本語コメントを生成する。", "表示する"],
                    ["説明者支援", "Client Pre/Post Demo", "デモ前後の説明、期待値調整、次アクション整理。", "表示しない"],
                    ["説明者支援", "Data Foundation", "AI活用に必要なデータ基盤の論点を説明する。", "表示しない"],
                    ["説明者支援", "Tech Architecture", "この技術構成資料。実装と運用を説明する。", "表示しない"],
                    ["内部確認", "Data Explorer", "架空データの列、件数、サンプルを検証する。", "表示しない"],
                ],
            ),
            "『相手に触ってもらう画面』と『こちらが説明する画面』を分けることで、デモ体験が混ざらないようにしています。",
        )
    elif slide == 6:
        render_presentation_slide(
            "Architecture / 07",
            "公開運用は、GitHubを正とし、Streamlit Cloudが読み込む形です",
            presentation_flow_html(
                [
                    ("1. ローカル編集", "cloneした作業フォルダで app.py、client_app.py、presenter_app.py を更新"),
                    ("2. 動作確認", "py_compile と Streamlit AppTest で主要導線を確認"),
                    ("3. commit", "GitHub Desktopまたはgitで変更理由を残す"),
                    ("4. push", "GitHubのmainへ反映"),
                    ("5. redeploy", "Streamlit Cloudが更新を検知し、各URLへ反映"),
                ],
                columns=5,
            )
            + presentation_table_html(
                ["運用項目", "現在の方針", "注意点"],
                [
                    ["Secrets", "デモでは外部APIを使わないため未設定", "将来APIを使う場合はStreamlit Secretsに置き、GitHubへ置かない。"],
                    ["不要ファイル", ".venv、__pycache__、ログ、QA画像はcommitしない", ".gitignoreで除外し、公開リポジトリを軽く保つ。"],
                    ["データ更新", "scripts/generate_demo_data.pyで再生成", "更新後はmetadataの日時と件数を確認する。"],
                    ["公開前QA", "クライアントURLに本体4ページだけ出ることを確認", "説明者向けページが混ざっていないかを見る。"],
                ],
            ),
            "長期運用では、GitHub上で履歴を残し、ローカルと公開環境の差分を小さく保つことが重要です。",
        )
    else:
        render_presentation_slide(
            "Architecture / 08",
            "本番化では、データ接続、権限、AI統制、監査ログを追加します",
            presentation_table_html(
                ["優先度", "テーマ", "やること", "成果物"],
                [
                    ["1", "実データ棚卸し", "ERP、EPM、案件管理、調達、工程、為替、マスタの所在と責任者を整理する。", "データソース一覧"],
                    ["2", "KPI定義", "売上、営業利益、利益率、CF、EAC、リスク指標の定義を合意する。", "KPI定義書"],
                    ["3", "データ基盤", "DWH/データマートにFP&A用の統合テーブルを設計する。", "To-Beアーキテクチャ"],
                    ["4", "AI実装", "LLM API、プロンプト、根拠引用、出力レビュー、禁止事項を設計する。", "AIコメント設計書"],
                    ["5", "統制", "権限、監査ログ、更新頻度、承認フロー、モデル変更管理を決める。", "運用設計"],
                    ["6", "段階展開", "代表セグメントまたは重点案件からPoCし、月次FP&Aプロセスへ組み込む。", "PoC計画"],
                ],
            )
            + presentation_cards_html(
                [
                    ("営業メッセージ", "このデモは完成品の販売ではなく、FP&A高度化の完成イメージを短時間で共有するためのもの。"),
                    ("次の会話", "相手企業のデータ成熟度、会議体、KPI運用、AI利用ルールを確認する。"),
                    ("提案の焦点", "画面開発だけでなく、データ基盤、業務設計、AI統制をセットで提案する。"),
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
        st.session_state["active_surface"] = "client"
        st.session_state["show_guide"] = False

    if st.session_state.get("active_surface") == "demo":
        st.session_state["active_surface"] = (
            "internal" if st.session_state.get("active_page") in internal_page_keys else "client"
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
    elif st.session_state.get("active_surface") not in {"client", "presenter", "operational", "internal"}:
        st.session_state["active_surface"] = "client"

    active_surface = st.session_state.get("active_surface")
    active_page = st.session_state.get("active_page")
    if active_surface == "client" and active_page not in client_page_keys:
        st.session_state["active_page"] = "Dashboard"
        st.session_state["show_guide"] = False
    elif active_surface == "presenter" and active_page not in presenter_page_keys:
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
            st.session_state["active_page"] = current_page if current_page in presenter_page_keys else "Client Pre-Demo"
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
            st.session_state["show_guide"] = presenter_page_guides.get(page_key, False)
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
            surface_options = ["client", "presenter", "operational", "internal"]
            surface_labels = {
                "client": "クライアント操作用",
                "presenter": "説明者用",
                "operational": "実務コックピット",
                "internal": "社内資料",
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
            page_options = [key for key, _, _ in presenter_pages]
            active_page = st.session_state.get("active_page")
            selected_page = st.radio(
                "ページ",
                page_options,
                index=page_options.index(active_page) if active_page in page_options else 0,
                format_func=lambda key: presenter_page_labels[key],
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
            "client": "クライアント操作用",
            "presenter": "説明者用",
            "operational": "実務コックピット",
            "internal": "社内資料",
        }.get(current_surface, "クライアント操作用")
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
        mode_label = "クライアント操作用"
    elif st.session_state.get("active_surface") == "presenter":
        current_label = presenter_page_labels.get(page, page)
        mode_label = "説明者用"
    elif st.session_state.get("active_surface") == "internal":
        internal_label_map = {
            "Internal Demo Guide": "説明者向けガイド",
            "Tech Architecture": "技術構成",
            "Data Explorer": "データ確認",
        }
        current_label = internal_label_map.get(page, page)
        mode_label = "社内使用モード"
    else:
        mode_label = "実務コックピット"
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
