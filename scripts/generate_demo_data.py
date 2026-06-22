from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COMPANY_NAME = "Nippon Advanced Heavy Industries"
FISCAL_YEAR = "FY2026"
RNG_SEED = 20260622

SEGMENTS = [
    {
        "segment_en": "Aerospace & Defense",
        "segment_ja": "航空・防衛",
        "segment_code": "AD",
        "annual_revenue_budget_jpy_mn": 1_850_000,
        "project_count": 70,
        "base_ratios": {
            "Material Cost": 0.340,
            "Outsourcing Cost": 0.170,
            "Labor Cost": 0.180,
            "Fixed Cost": 0.120,
            "Engineering Change / Rework": 0.025,
            "Other Operating Cost": 0.060,
        },
        "business_units": [
            "Aero Engines",
            "Defense Systems",
            "Space Components",
            "Avionics & Controls",
        ],
    },
    {
        "segment_en": "Energy Systems",
        "segment_ja": "エネルギーシステム",
        "segment_code": "ES",
        "annual_revenue_budget_jpy_mn": 1_500_000,
        "project_count": 65,
        "base_ratios": {
            "Material Cost": 0.370,
            "Outsourcing Cost": 0.120,
            "Labor Cost": 0.160,
            "Fixed Cost": 0.110,
            "Engineering Change / Rework": 0.015,
            "Other Operating Cost": 0.105,
        },
        "business_units": [
            "Gas Turbine Solutions",
            "Grid & Storage",
            "Hydrogen Systems",
            "Thermal Service",
        ],
    },
    {
        "segment_en": "Marine & Offshore",
        "segment_ja": "船舶・海洋",
        "segment_code": "MO",
        "annual_revenue_budget_jpy_mn": 850_000,
        "project_count": 55,
        "base_ratios": {
            "Material Cost": 0.410,
            "Outsourcing Cost": 0.140,
            "Labor Cost": 0.170,
            "Fixed Cost": 0.120,
            "Engineering Change / Rework": 0.030,
            "Other Operating Cost": 0.060,
        },
        "business_units": [
            "Offshore Platforms",
            "Naval Systems",
            "Commercial Vessels",
            "Marine Service",
        ],
    },
    {
        "segment_en": "Industrial Machinery & Robotics",
        "segment_ja": "産業機械・ロボット",
        "segment_code": "IR",
        "annual_revenue_budget_jpy_mn": 650_000,
        "project_count": 70,
        "base_ratios": {
            "Material Cost": 0.330,
            "Outsourcing Cost": 0.120,
            "Labor Cost": 0.200,
            "Fixed Cost": 0.160,
            "Engineering Change / Rework": 0.020,
            "Other Operating Cost": 0.080,
        },
        "business_units": [
            "Factory Automation",
            "Robotics Cells",
            "Compressors & Drives",
            "Aftermarket Service",
        ],
    },
]

SCENARIOS = {
    "Actual": "実績",
    "Budget": "予算",
    "Previous Forecast": "前回見込",
    "Latest Forecast": "最新見込",
    "Prior Year Actual": "前年実績",
    "Mid-term Plan": "中期計画",
}

ACCOUNT_LABELS = {
    "Revenue": "売上",
    "Material Cost": "材料費",
    "Outsourcing Cost": "外注費",
    "Labor Cost": "労務費",
    "Fixed Cost": "固定費",
    "Engineering Change / Rework": "設計変更・手戻り",
    "Other Operating Cost": "その他営業費用",
    "Cash Flow": "キャッシュフロー",
}

VARIANCE_DRIVERS = {
    "Material Cost Variance": "材料費差異",
    "Outsourcing Cost Variance": "外注費差異",
    "FX Variance": "為替差異",
    "Schedule / Revenue Timing Variance": "納期遅延・売上計上時期差異",
    "Project EAC Deterioration": "案件EAC悪化",
    "Other": "その他",
}

COMPARISONS = {
    "Budget vs Actual": ("Budget", "Actual"),
    "Previous Forecast vs Latest Forecast": ("Previous Forecast", "Latest Forecast"),
    "Prior Year Actual vs Actual": ("Prior Year Actual", "Actual"),
    "Mid-term Plan vs Latest Forecast": ("Mid-term Plan", "Latest Forecast"),
}

MONTHS = pd.period_range("2025-04", periods=12, freq="M")
MONTH_WEIGHTS = np.array([0.070, 0.073, 0.075, 0.079, 0.081, 0.083, 0.081, 0.083, 0.091, 0.086, 0.092, 0.106])
MONTH_WEIGHTS = MONTH_WEIGHTS / MONTH_WEIGHTS.sum()

REGIONS = ["Japan", "APAC", "North America", "Europe", "Middle East"]
CONTRACT_TYPES = ["Fixed price", "Cost plus", "Long-term service", "Framework agreement"]
PHASES = ["Design", "Procurement", "Manufacturing", "Commissioning", "Service"]


def scenario_policy(segment_en: str, scenario: str) -> dict[str, object]:
    base = {
        "revenue_factor": 1.000,
        "cost_deltas": {
            "Material Cost": 0.000,
            "Outsourcing Cost": 0.000,
            "Labor Cost": 0.000,
            "Fixed Cost": 0.000,
            "Engineering Change / Rework": 0.000,
            "Other Operating Cost": 0.000,
        },
        "cash_conversion": 0.92,
        "working_capital_drag": 0.010,
        "risk_sensitivity": 0.10,
    }

    actual = {
        "Aerospace & Defense": {
            "revenue_factor": 1.055,
            "cost_deltas": {
                "Material Cost": 0.012,
                "Outsourcing Cost": 0.035,
                "Labor Cost": 0.004,
                "Fixed Cost": 0.006,
                "Engineering Change / Rework": 0.006,
                "Other Operating Cost": 0.004,
            },
            "cash_conversion": 0.70,
            "working_capital_drag": 0.043,
            "risk_sensitivity": 0.45,
        },
        "Energy Systems": {
            "revenue_factor": 1.080,
            "cost_deltas": {
                "Material Cost": -0.010,
                "Outsourcing Cost": -0.004,
                "Labor Cost": -0.004,
                "Fixed Cost": -0.006,
                "Engineering Change / Rework": -0.003,
                "Other Operating Cost": -0.003,
            },
            "cash_conversion": 0.98,
            "working_capital_drag": 0.003,
            "risk_sensitivity": 0.05,
        },
        "Marine & Offshore": {
            "revenue_factor": 1.015,
            "cost_deltas": {
                "Material Cost": 0.030,
                "Outsourcing Cost": 0.025,
                "Labor Cost": 0.015,
                "Fixed Cost": 0.025,
                "Engineering Change / Rework": 0.065,
                "Other Operating Cost": 0.018,
            },
            "cash_conversion": 0.52,
            "working_capital_drag": 0.080,
            "risk_sensitivity": 0.75,
        },
        "Industrial Machinery & Robotics": {
            "revenue_factor": 0.925,
            "cost_deltas": {
                "Material Cost": 0.004,
                "Outsourcing Cost": 0.002,
                "Labor Cost": 0.006,
                "Fixed Cost": 0.020,
                "Engineering Change / Rework": 0.006,
                "Other Operating Cost": 0.010,
            },
            "cash_conversion": 0.68,
            "working_capital_drag": 0.028,
            "risk_sensitivity": 0.35,
        },
    }

    previous_forecast = {
        "Aerospace & Defense": (1.040, 0.35),
        "Energy Systems": (1.055, -0.25),
        "Marine & Offshore": (1.045, 0.20),
        "Industrial Machinery & Robotics": (0.965, 0.20),
    }
    latest_forecast = {
        "Aerospace & Defense": (1.060, 0.90),
        "Energy Systems": (1.090, -0.80),
        "Marine & Offshore": (1.005, 1.10),
        "Industrial Machinery & Robotics": (0.910, 0.85),
    }
    prior_year = {
        "Aerospace & Defense": (0.945, 0.05),
        "Energy Systems": (0.930, 0.10),
        "Marine & Offshore": (0.940, -0.05),
        "Industrial Machinery & Robotics": (1.015, -0.02),
    }
    mid_term = {
        "Aerospace & Defense": (1.125, -0.30),
        "Energy Systems": (1.155, -0.35),
        "Marine & Offshore": (1.115, -0.25),
        "Industrial Machinery & Robotics": (1.095, -0.25),
    }

    if scenario == "Budget":
        return base
    if scenario == "Actual":
        return actual[segment_en]

    if scenario == "Previous Forecast":
        revenue_factor, delta_scale = previous_forecast[segment_en]
        source = actual[segment_en]
    elif scenario == "Latest Forecast":
        revenue_factor, delta_scale = latest_forecast[segment_en]
        source = actual[segment_en]
    elif scenario == "Prior Year Actual":
        revenue_factor, delta_scale = prior_year[segment_en]
        source = actual[segment_en]
    elif scenario == "Mid-term Plan":
        revenue_factor, delta_scale = mid_term[segment_en]
        source = actual[segment_en]
    else:
        return base

    policy = {
        "revenue_factor": revenue_factor,
        "cost_deltas": {
            account: delta * delta_scale for account, delta in source["cost_deltas"].items()
        },
        "cash_conversion": max(0.50, min(1.05, source["cash_conversion"] + (1 - delta_scale) * 0.12)),
        "working_capital_drag": max(0.000, source["working_capital_drag"] * max(delta_scale, 0.0)),
        "risk_sensitivity": source["risk_sensitivity"] * max(delta_scale, 0.05),
    }
    return policy


def make_projects(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    name_tokens = {
        "AD": ["Aster", "Stratos", "Vector", "Helios", "Nimbus", "Falcon Ridge"],
        "ES": ["Aurora", "Kinetic", "Hydra", "Terra", "Pulse", "Celsius"],
        "MO": ["Harbor", "Pelagus", "Blue Arc", "Triton", "North Quay", "Oceanus"],
        "IR": ["Atlas", "Forge", "Servo", "Mira", "Axis", "Nova"],
    }
    suffixes = ["Modernization", "Module", "Package", "Program", "Retrofit", "Service Block"]

    for segment in SEGMENTS:
        weights = rng.lognormal(mean=0.0, sigma=0.75, size=segment["project_count"])
        weights = weights / weights.sum()
        annual_revenues = weights * segment["annual_revenue_budget_jpy_mn"]
        code = segment["segment_code"]

        for idx, annual_revenue in enumerate(annual_revenues, start=1):
            if code == "MO" and idx <= 7:
                risk_bias = rng.uniform(0.86, 0.98)
                embedded_tier = "Critical seed"
            elif code == "AD" and idx <= 10:
                risk_bias = rng.uniform(0.66, 0.84)
                embedded_tier = "High seed"
            elif code == "IR" and idx <= 8:
                risk_bias = rng.uniform(0.45, 0.68)
                embedded_tier = "Demand watch"
            else:
                risk_bias = float(rng.beta(2.0, 5.0))
                embedded_tier = "Normal"

            token = rng.choice(name_tokens[code])
            suffix = rng.choice(suffixes)
            business_unit = rng.choice(segment["business_units"])
            project_id = f"{code}-{idx:03d}"
            project_name = f"{token} {suffix} {idx:03d}"
            rows.append(
                {
                    "company": COMPANY_NAME,
                    "project_id": project_id,
                    "project_name": project_name,
                    "segment_en": segment["segment_en"],
                    "segment_ja": segment["segment_ja"],
                    "segment_code": code,
                    "business_unit": business_unit,
                    "customer_region": rng.choice(REGIONS, p=[0.42, 0.22, 0.14, 0.14, 0.08]),
                    "contract_type": rng.choice(CONTRACT_TYPES, p=[0.48, 0.18, 0.22, 0.12]),
                    "project_phase": rng.choice(PHASES, p=[0.15, 0.20, 0.30, 0.20, 0.15]),
                    "annual_revenue_budget_jpy_mn": float(annual_revenue),
                    "risk_bias": risk_bias,
                    "embedded_story_tier": embedded_tier,
                }
            )

    return pd.DataFrame(rows)


def make_finance_fact(projects: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_config = {s["segment_en"]: s for s in SEGMENTS}

    for project in projects.to_dict("records"):
        segment = segment_config[project["segment_en"]]
        annual_budget = float(project["annual_revenue_budget_jpy_mn"])
        risk_bias = float(project["risk_bias"])
        project_noise = rng.normal(1.0, 0.015)

        for month_idx, period in enumerate(MONTHS, start=1):
            period_start = period.to_timestamp()
            month_weight = MONTH_WEIGHTS[month_idx - 1]
            budget_revenue_month = annual_budget * month_weight * project_noise

            for scenario, scenario_ja in SCENARIOS.items():
                policy = scenario_policy(project["segment_en"], scenario)
                revenue_factor = float(policy["revenue_factor"])
                risk_sensitivity = float(policy["risk_sensitivity"])
                cost_deltas = policy["cost_deltas"]

                timing_noise = rng.normal(1.0, 0.012)
                if project["segment_code"] == "MO" and scenario in {"Actual", "Latest Forecast"} and month_idx in {8, 9, 10}:
                    timing_noise *= 0.965
                if project["segment_code"] == "ES" and scenario in {"Actual", "Latest Forecast"} and month_idx in {6, 7, 8}:
                    timing_noise *= 1.035

                revenue = budget_revenue_month * revenue_factor * timing_noise
                op_amount = revenue

                rows.append(
                    {
                        "company": COMPANY_NAME,
                        "fiscal_year": FISCAL_YEAR,
                        "period": str(period),
                        "period_start": period_start,
                        "fiscal_month": month_idx,
                        "scenario": scenario,
                        "scenario_ja": scenario_ja,
                        "segment_en": project["segment_en"],
                        "segment_ja": project["segment_ja"],
                        "segment_code": project["segment_code"],
                        "business_unit": project["business_unit"],
                        "project_id": project["project_id"],
                        "project_name": project["project_name"],
                        "account": "Revenue",
                        "account_ja": ACCOUNT_LABELS["Revenue"],
                        "account_type": "Revenue",
                        "amount_jpy_mn": round(revenue, 3),
                    }
                )

                for account, base_ratio in segment["base_ratios"].items():
                    delta = float(cost_deltas[account])
                    if delta > 0:
                        delta *= 1 + risk_bias * risk_sensitivity
                    elif delta < 0:
                        delta *= 1 - min(risk_bias * 0.20, 0.10)

                    ratio = max(0.0, base_ratio + delta)
                    line_noise = rng.normal(1.0, 0.010)
                    if account == "Fixed Cost":
                        amount = -budget_revenue_month * ratio * line_noise
                    else:
                        amount = -revenue * ratio * line_noise
                    op_amount += amount
                    rows.append(
                        {
                            "company": COMPANY_NAME,
                            "fiscal_year": FISCAL_YEAR,
                            "period": str(period),
                            "period_start": period_start,
                            "fiscal_month": month_idx,
                            "scenario": scenario,
                            "scenario_ja": scenario_ja,
                            "segment_en": project["segment_en"],
                            "segment_ja": project["segment_ja"],
                            "segment_code": project["segment_code"],
                            "business_unit": project["business_unit"],
                            "project_id": project["project_id"],
                            "project_name": project["project_name"],
                            "account": account,
                            "account_ja": ACCOUNT_LABELS[account],
                            "account_type": "Operating Cost",
                            "amount_jpy_mn": round(amount, 3),
                        }
                    )

                cash_conversion = float(policy["cash_conversion"])
                wc_drag = float(policy["working_capital_drag"])
                cash_flow = op_amount * cash_conversion - budget_revenue_month * wc_drag
                cash_flow *= rng.normal(1.0, 0.018)
                rows.append(
                    {
                        "company": COMPANY_NAME,
                        "fiscal_year": FISCAL_YEAR,
                        "period": str(period),
                        "period_start": period_start,
                        "fiscal_month": month_idx,
                        "scenario": scenario,
                        "scenario_ja": scenario_ja,
                        "segment_en": project["segment_en"],
                        "segment_ja": project["segment_ja"],
                        "segment_code": project["segment_code"],
                        "business_unit": project["business_unit"],
                        "project_id": project["project_id"],
                        "project_name": project["project_name"],
                        "account": "Cash Flow",
                        "account_ja": ACCOUNT_LABELS["Cash Flow"],
                        "account_type": "Cash Flow",
                        "amount_jpy_mn": round(cash_flow, 3),
                    }
                )

    return pd.DataFrame(rows)


def driver_coefficients(segment_en: str, comparison: str) -> dict[str, tuple[float, float, float]]:
    # Tuple is impact on revenue, operating profit, and cash flow as a share of budget revenue.
    if comparison == "Budget vs Actual":
        template = {
            "Aerospace & Defense": {
                "Material Cost Variance": (0.000, -0.007, -0.006),
                "Outsourcing Cost Variance": (0.000, -0.020, -0.019),
                "FX Variance": (0.000, -0.012, -0.014),
                "Schedule / Revenue Timing Variance": (0.044, 0.010, -0.013),
                "Project EAC Deterioration": (0.000, -0.004, -0.006),
                "Other": (0.004, -0.003, -0.004),
            },
            "Energy Systems": {
                "Material Cost Variance": (0.000, 0.004, 0.004),
                "Outsourcing Cost Variance": (0.000, 0.002, 0.002),
                "FX Variance": (0.000, -0.003, -0.003),
                "Schedule / Revenue Timing Variance": (0.071, 0.032, 0.028),
                "Project EAC Deterioration": (0.000, 0.001, 0.001),
                "Other": (0.006, 0.004, 0.003),
            },
            "Marine & Offshore": {
                "Material Cost Variance": (0.000, -0.012, -0.012),
                "Outsourcing Cost Variance": (0.000, -0.011, -0.011),
                "FX Variance": (0.000, -0.004, -0.005),
                "Schedule / Revenue Timing Variance": (-0.018, -0.012, -0.028),
                "Project EAC Deterioration": (0.000, -0.065, -0.072),
                "Other": (0.004, -0.008, -0.009),
            },
            "Industrial Machinery & Robotics": {
                "Material Cost Variance": (0.000, -0.002, -0.002),
                "Outsourcing Cost Variance": (0.000, -0.001, -0.001),
                "FX Variance": (0.000, -0.002, -0.002),
                "Schedule / Revenue Timing Variance": (-0.074, -0.040, -0.046),
                "Project EAC Deterioration": (0.000, -0.003, -0.004),
                "Other": (-0.002, -0.018, -0.019),
            },
        }
    elif comparison == "Previous Forecast vs Latest Forecast":
        template = {
            "Aerospace & Defense": {
                "Material Cost Variance": (0.000, -0.003, -0.003),
                "Outsourcing Cost Variance": (0.000, -0.010, -0.010),
                "FX Variance": (0.000, -0.009, -0.010),
                "Schedule / Revenue Timing Variance": (0.018, 0.004, -0.004),
                "Project EAC Deterioration": (0.000, -0.004, -0.005),
                "Other": (0.002, -0.002, -0.002),
            },
            "Energy Systems": {
                "Material Cost Variance": (0.000, 0.002, 0.002),
                "Outsourcing Cost Variance": (0.000, 0.001, 0.001),
                "FX Variance": (0.000, -0.001, -0.001),
                "Schedule / Revenue Timing Variance": (0.034, 0.018, 0.017),
                "Project EAC Deterioration": (0.000, 0.000, 0.000),
                "Other": (0.001, 0.002, 0.002),
            },
            "Marine & Offshore": {
                "Material Cost Variance": (0.000, -0.010, -0.010),
                "Outsourcing Cost Variance": (0.000, -0.008, -0.009),
                "FX Variance": (0.000, -0.002, -0.003),
                "Schedule / Revenue Timing Variance": (-0.040, -0.018, -0.030),
                "Project EAC Deterioration": (0.000, -0.055, -0.061),
                "Other": (-0.002, -0.006, -0.007),
            },
            "Industrial Machinery & Robotics": {
                "Material Cost Variance": (0.000, -0.001, -0.001),
                "Outsourcing Cost Variance": (0.000, -0.001, -0.001),
                "FX Variance": (0.000, -0.001, -0.001),
                "Schedule / Revenue Timing Variance": (-0.055, -0.030, -0.034),
                "Project EAC Deterioration": (0.000, -0.002, -0.003),
                "Other": (-0.002, -0.012, -0.013),
            },
        }
    elif comparison == "Prior Year Actual vs Actual":
        template = {
            "Aerospace & Defense": {
                "Material Cost Variance": (0.000, -0.006, -0.006),
                "Outsourcing Cost Variance": (0.000, -0.016, -0.016),
                "FX Variance": (0.000, -0.010, -0.012),
                "Schedule / Revenue Timing Variance": (0.100, 0.028, 0.010),
                "Project EAC Deterioration": (0.000, -0.004, -0.005),
                "Other": (0.006, -0.003, -0.003),
            },
            "Energy Systems": {
                "Material Cost Variance": (0.000, 0.004, 0.004),
                "Outsourcing Cost Variance": (0.000, 0.002, 0.002),
                "FX Variance": (0.000, -0.003, -0.003),
                "Schedule / Revenue Timing Variance": (0.145, 0.052, 0.044),
                "Project EAC Deterioration": (0.000, 0.001, 0.001),
                "Other": (0.006, 0.004, 0.003),
            },
            "Marine & Offshore": {
                "Material Cost Variance": (0.000, -0.010, -0.010),
                "Outsourcing Cost Variance": (0.000, -0.009, -0.009),
                "FX Variance": (0.000, -0.004, -0.004),
                "Schedule / Revenue Timing Variance": (0.075, 0.008, -0.010),
                "Project EAC Deterioration": (0.000, -0.060, -0.067),
                "Other": (0.004, -0.007, -0.008),
            },
            "Industrial Machinery & Robotics": {
                "Material Cost Variance": (0.000, -0.002, -0.002),
                "Outsourcing Cost Variance": (0.000, -0.001, -0.001),
                "FX Variance": (0.000, -0.002, -0.002),
                "Schedule / Revenue Timing Variance": (-0.090, -0.048, -0.054),
                "Project EAC Deterioration": (0.000, -0.003, -0.004),
                "Other": (-0.002, -0.018, -0.020),
            },
        }
    else:
        template = {
            segment["segment_en"]: {
                "Material Cost Variance": (0.000, -0.006, -0.006),
                "Outsourcing Cost Variance": (0.000, -0.006, -0.006),
                "FX Variance": (0.000, -0.003, -0.003),
                "Schedule / Revenue Timing Variance": (-0.050, -0.025, -0.030),
                "Project EAC Deterioration": (0.000, -0.018, -0.020),
                "Other": (-0.006, -0.010, -0.011),
            }
            for segment in SEGMENTS
        }
        template["Energy Systems"]["Schedule / Revenue Timing Variance"] = (0.018, 0.012, 0.010)
        template["Energy Systems"]["Other"] = (0.002, 0.004, 0.004)

    return template[segment_en]


def make_variance_drivers(projects: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    project_records = projects.to_dict("records")

    for project in project_records:
        annual_budget = float(project["annual_revenue_budget_jpy_mn"])
        risk_bias = float(project["risk_bias"])

        for month_idx, period in enumerate(MONTHS, start=1):
            period_start = period.to_timestamp()
            budget_revenue_month = annual_budget * MONTH_WEIGHTS[month_idx - 1]

            for comparison, (base_scenario, target_scenario) in COMPARISONS.items():
                coeffs = driver_coefficients(project["segment_en"], comparison)
                for driver, driver_ja in VARIANCE_DRIVERS.items():
                    revenue_coeff, op_coeff, cf_coeff = coeffs[driver]
                    noise = rng.normal(1.0, 0.055)
                    risk_scale = 1 + risk_bias * 0.55 if op_coeff < 0 else 1 - risk_bias * 0.08
                    if driver == "Project EAC Deterioration":
                        risk_scale += risk_bias * 0.85
                    if driver == "Schedule / Revenue Timing Variance" and project["segment_code"] in {"MO", "IR"}:
                        risk_scale += risk_bias * 0.35

                    impact_revenue = budget_revenue_month * revenue_coeff * noise
                    impact_op = budget_revenue_month * op_coeff * noise * risk_scale
                    impact_cf = budget_revenue_month * cf_coeff * noise * risk_scale

                    rows.append(
                        {
                            "company": COMPANY_NAME,
                            "fiscal_year": FISCAL_YEAR,
                            "period": str(period),
                            "period_start": period_start,
                            "fiscal_month": month_idx,
                            "comparison_type": comparison,
                            "base_scenario": base_scenario,
                            "target_scenario": target_scenario,
                            "segment_en": project["segment_en"],
                            "segment_ja": project["segment_ja"],
                            "segment_code": project["segment_code"],
                            "business_unit": project["business_unit"],
                            "project_id": project["project_id"],
                            "project_name": project["project_name"],
                            "variance_driver": driver,
                            "variance_driver_ja": driver_ja,
                            "impact_revenue_jpy_mn": round(impact_revenue, 3),
                            "impact_op_jpy_mn": round(impact_op, 3),
                            "impact_cash_flow_jpy_mn": round(impact_cf, 3),
                            "ai_confidence": round(float(rng.uniform(0.62, 0.93)), 3),
                        }
                    )

    return pd.DataFrame(rows)


def make_project_risk(projects: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for project in projects.to_dict("records"):
        annual_revenue = float(project["annual_revenue_budget_jpy_mn"])
        risk_bias = float(project["risk_bias"])
        segment = project["segment_en"]
        budget_cost = annual_revenue * 0.905

        if segment == "Marine & Offshore":
            eac_delta_ratio = 0.035 + risk_bias * 0.165 + rng.normal(0.0, 0.008)
            schedule_delay = int(np.clip(20 + risk_bias * 135 + rng.normal(0, 9), 8, 180))
            risk_score = int(np.clip(46 + risk_bias * 58 + rng.normal(0, 4), 35, 99))
            primary_driver = "案件EAC悪化 / 設計変更 / 納期遅延"
            owner = "Marine Program Control"
            action = "EAC再見積、変更契約の早期交渉、工程クリティカルパスの週次レビュー"
        elif segment == "Aerospace & Defense":
            eac_delta_ratio = 0.015 + risk_bias * 0.075 + rng.normal(0.0, 0.006)
            schedule_delay = int(np.clip(10 + risk_bias * 80 + rng.normal(0, 7), 0, 120))
            risk_score = int(np.clip(38 + risk_bias * 58 + rng.normal(0, 4), 25, 92))
            primary_driver = "外注費上振れ / 為替 / 長納期部品"
            owner = "A&D Supply Chain"
            action = "外注単価の再交渉、為替感応度のヘッジ確認、長納期部品の代替調達判断"
        elif segment == "Industrial Machinery & Robotics":
            eac_delta_ratio = 0.006 + risk_bias * 0.045 + rng.normal(0.0, 0.006)
            schedule_delay = int(np.clip(5 + risk_bias * 45 + rng.normal(0, 6), 0, 80))
            risk_score = int(np.clip(28 + risk_bias * 48 + rng.normal(0, 4), 18, 78))
            primary_driver = "需要鈍化 / 固定費負担"
            owner = "Industrial FP&A"
            action = "受注確度の再評価、固定費吸収計画の見直し、在庫水準の圧縮"
        else:
            eac_delta_ratio = -0.004 + risk_bias * 0.018 + rng.normal(0.0, 0.004)
            schedule_delay = int(np.clip(3 + risk_bias * 28 + rng.normal(0, 5), 0, 55))
            risk_score = int(np.clip(18 + risk_bias * 35 + rng.normal(0, 4), 8, 62))
            primary_driver = "高採算案件の前倒し / 調達条件改善"
            owner = "Energy Program Office"
            action = "前倒し計上の実現性確認、追加オーダー獲得、粗利率維持条件のレビュー"

        eac_delta = max(annual_revenue * eac_delta_ratio, -annual_revenue * 0.015)
        latest_eac = budget_cost + eac_delta
        latest_margin = (annual_revenue - latest_eac) / annual_revenue
        loss_risk = latest_margin < 0.015 or risk_score >= 88

        if risk_score >= 86 or (segment == "Marine & Offshore" and risk_bias > 0.82):
            risk_level = "Critical"
        elif risk_score >= 70 or (segment == "Aerospace & Defense" and risk_bias > 0.64):
            risk_level = "High"
        elif risk_score >= 48:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        rows.append(
            {
                "company": COMPANY_NAME,
                "project_id": project["project_id"],
                "project_name": project["project_name"],
                "segment_en": segment,
                "segment_ja": project["segment_ja"],
                "business_unit": project["business_unit"],
                "customer_region": project["customer_region"],
                "contract_type": project["contract_type"],
                "project_phase": project["project_phase"],
                "risk_level": risk_level,
                "risk_score": risk_score,
                "annual_revenue_budget_jpy_mn": round(annual_revenue, 3),
                "budget_eac_jpy_mn": round(budget_cost, 3),
                "latest_eac_jpy_mn": round(latest_eac, 3),
                "eac_deterioration_jpy_mn": round(eac_delta, 3),
                "forecast_margin_pct": round(latest_margin * 100, 2),
                "schedule_delay_days": schedule_delay,
                "loss_risk_flag": bool(loss_risk),
                "primary_driver_ja": primary_driver,
                "owner_department": owner,
                "recommended_action_ja": action,
            }
        )

    risk = pd.DataFrame(rows)

    # Make the demo narrative explicit even if random variation is changed later.
    marine_idx = risk.query("segment_en == 'Marine & Offshore'").nlargest(5, "annual_revenue_budget_jpy_mn").index
    risk.loc[marine_idx, "risk_level"] = "Critical"
    risk.loc[marine_idx, "risk_score"] = np.maximum(risk.loc[marine_idx, "risk_score"], 91)
    risk.loc[marine_idx, "loss_risk_flag"] = True
    risk.loc[marine_idx, "forecast_margin_pct"] = np.minimum(risk.loc[marine_idx, "forecast_margin_pct"], -2.5)
    risk.loc[marine_idx, "primary_driver_ja"] = "案件EAC悪化 / 設計変更 / 納期遅延"

    ad_idx = risk.query("segment_en == 'Aerospace & Defense'").nlargest(6, "annual_revenue_budget_jpy_mn").index
    risk.loc[ad_idx, "risk_level"] = "High"
    risk.loc[ad_idx, "risk_score"] = np.maximum(risk.loc[ad_idx, "risk_score"], 76)
    risk.loc[ad_idx, "primary_driver_ja"] = "外注費上振れ / 為替 / 長納期部品"

    ad_all = risk["segment_en"] == "Aerospace & Defense"
    risk.loc[ad_all & (risk["risk_level"] == "Critical"), "risk_level"] = "High"
    risk.loc[ad_all, "risk_score"] = np.minimum(risk.loc[ad_all, "risk_score"], 85)

    return risk.sort_values(["risk_score", "eac_deterioration_jpy_mn"], ascending=[False, False]).reset_index(drop=True)


def write_outputs(projects: pd.DataFrame, finance: pd.DataFrame, drivers: pd.DataFrame, risk: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    projects.to_parquet(DATA_DIR / "dim_projects.parquet", index=False)
    finance.to_parquet(DATA_DIR / "fact_finance.parquet", index=False)
    drivers.to_parquet(DATA_DIR / "fact_variance_drivers.parquet", index=False)
    risk.to_parquet(DATA_DIR / "project_risk.parquet", index=False)

    metadata = {
        "company": COMPANY_NAME,
        "fiscal_year": FISCAL_YEAR,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": "Demo data only. All figures are fictional.",
        "row_counts": {
            "dim_projects": int(len(projects)),
            "fact_finance": int(len(finance)),
            "fact_variance_drivers": int(len(drivers)),
            "project_risk": int(len(risk)),
            "total": int(len(projects) + len(finance) + len(drivers) + len(risk)),
        },
        "embedded_story": {
            "company": "Revenue is above plan, while operating profit, margin, and cash flow deteriorate.",
            "Aerospace & Defense": "Revenue is solid; outsourcing, FX, and long-lead parts pressure margin and cash flow.",
            "Energy Systems": "High-margin project acceleration improves revenue and operating profit.",
            "Marine & Offshore": "EAC deterioration, design changes, and delays create major profit pressure and loss-making risk.",
            "Industrial Machinery & Robotics": "Demand slowdown causes revenue shortfall and fixed-cost absorption pressure.",
        },
    }
    (DATA_DIR / "demo_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fictional FP&A demo data for the AI Variance Analysis Cockpit.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for reproducible demo data.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    projects = make_projects(rng)
    finance = make_finance_fact(projects, rng)
    drivers = make_variance_drivers(projects, rng)
    risk = make_project_risk(projects, rng)
    write_outputs(projects, finance, drivers, risk)

    total_rows = len(projects) + len(finance) + len(drivers) + len(risk)
    print(f"Generated demo data in {DATA_DIR}")
    print(f"- dim_projects: {len(projects):,} rows")
    print(f"- fact_finance: {len(finance):,} rows")
    print(f"- fact_variance_drivers: {len(drivers):,} rows")
    print(f"- project_risk: {len(risk):,} rows")
    print(f"- total: {total_rows:,} rows")


if __name__ == "__main__":
    main()
