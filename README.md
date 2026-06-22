# AI Variance Analysis Cockpit

Fictional FP&A variance analysis demo for **Nippon Advanced Heavy Industries**.

[Dashboard/client app](https://heavy-industry-ai-demo-client.streamlit.app/)

[Full app](https://heavy-industry-ai-demo-6apcskxwufgremlq9bivhk.streamlit.app/)

Deploy targets:

- Dashboard/client app: `client_app.py`
- Presenter/support app: `presenter_app.py`
- Internal/full app: `app.py`

- Demo data only
- All figures are fictional
- No real company financial data is used

## Current Status

- Demo data generated at **2026-06-22 16:56:49 JST**
- Current generated row count: **225,160 rows**
- FY2026 total story: revenue is **+JPY 192.9bn vs Budget**, while operating profit is **-JPY 298.8bn**, margin is **-6.3pt**, and cash flow is **-JPY 341.8bn**
- Current risk queue: **11 Critical**, **32 High**, and **37 loss-risk** projects

## Setup

```bash
pip install -r requirements.txt
python scripts/generate_demo_data.py
streamlit run app.py
```

Open the app at:

```text
http://localhost:8501
```

## Demo Flow

The app separates three usage modes:

- **Dashboard/client app**: only the cockpit screens a client can display and touch.
- **Presenter/support app**: briefing, data foundation, reference architecture, follow-up, and support pages shown by us.
- **Internal/full app**: all modes in one app for maintenance and QA.

Dashboard/client app:

1. **Dashboard**: Show the executive story: revenue is above plan, while operating profit, margin, and cash flow deteriorate.
2. **Variance Analysis**: Select period, segment, KPI, and comparison type; explain the waterfall and top drivers.
3. **Project Risk**: Highlight Critical Marine & Offshore projects and High Aerospace & Defense projects.
4. **AI Commentary**: Generate Japanese FP&A comments for management meeting materials.

Presenter/support app:

- **Client Preview**: Align expectations, fictional-data premise, and what to watch.
- **Data Foundation**: Explain why trusted FP&A data is the prerequisite for AI variance analysis.
- **Reference Architecture**: Explain implementation options and rollout approach for AI-enabled FP&A.
- **Client Follow-up**: Summarize takeaways and move into data assessment / PoC scoping.
- **デモ解説用 / Presenter Guide**: Internal talk track, Q&A, and wording guardrails.
- **技術構成プレゼン / Tech Architecture**: Slide-style internal page explaining how the demo is built, operated, and extended.
- **Data Explorer**: Inspect generated fictional data, row counts, columns, samples, and statistics.

## Recommended Next Steps

| Priority | Theme | Action | Output |
|---|---|---|---|
| 1 | Demo script | Fix the story sequence: Dashboard -> Variance Analysis -> Project Risk -> AI Commentary | 5-minute and 15-minute talk tracks |
| 2 | Story depth | Prepare representative Critical and High project examples | Focus project list and expected Q&A |
| 3 | Data credibility | Document fictional assumptions, data grain, KPI definitions, and variance logic | Data definition slide |
| 4 | AI positioning | Explain the rule-based Japanese commentary logic and its limits | AI value explanation |
| 5 | Deployment | Push to GitHub and deploy on Streamlit Community Cloud | Shareable demo URL |

## Data Foundation Message

The central point of this demo is not only that AI can write variance comments.
The more important message is that AI-driven FP&A requires a trusted data foundation.

ERP actuals, EPM budget and forecast data, project EAC, procurement costs, schedule milestones, FX data, and master data must be reconciled and governed before AI can produce reliable management commentary.

```mermaid
flowchart LR
    A[ERP / GL actuals] --> H[Ingestion and validation]
    B[EPM budget and forecast] --> H
    C[Project EAC system] --> H
    D[Procurement and supplier costs] --> H
    E[Schedule and milestone data] --> H
    F[FX and treasury data] --> H
    G[Master data] --> H
    H --> I[Trusted FP&A data foundation]
    I --> J[KPI and variance semantic layer]
    J --> K[AI Variance Analysis Cockpit]
    K --> L[Executive commentary]
    K --> M[Project risk actions]
```

Required quality gates:

- Reconcile FP&A totals with ERP and GL.
- Lock scenario versions such as Budget, Previous Forecast, and Latest Forecast.
- Map project IDs, departments, segments, accounts, and customers consistently.
- Align monthly, quarterly, and annual grains.
- Preserve data lineage so AI comments can be traced back to source facts.

## Technical Architecture

```mermaid
flowchart LR
    A[Fictional assumptions] --> B[scripts/generate_demo_data.py]
    B --> C[fact_finance.parquet]
    B --> D[fact_variance_drivers.parquet]
    B --> E[project_risk.parquet]
    C --> F[Streamlit @st.cache_data]
    D --> F
    E --> F
    F --> G[pandas KPI and variance layer]
    G --> H[Plotly visual layer]
    H --> I[Dashboard]
    H --> J[Variance Analysis]
    H --> K[Project Risk]
    G --> L[Rule-based commentary]
    L --> N[AI Commentary]
    C --> O[Data Explorer]
    D --> O
    E --> O
```

## Data

`python scripts/generate_demo_data.py` creates:

- `data/fact_finance.parquet`
- `data/fact_variance_drivers.parquet`
- `data/project_risk.parquet`
- `data/dim_projects.parquet`
- `data/demo_metadata.json`

Current generated row count is **225,160 rows**.
