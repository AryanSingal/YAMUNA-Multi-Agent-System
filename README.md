# Yamuna River Water-Quality Assessment Platform (MAS-IFS + MCP)

An interactive, uncertainty-aware water-quality assessment platform for the Delhi stretch of the Yamuna River, combining a modular **Multi-Agent System (MAS)**, **Atanassov Intuitionistic Fuzzy Sets (IFS)**, standard regulatory indices (**NSF-WQI** and **CCME-WQI**), a **Model Context Protocol (MCP)** server, and a **Gradio** web interface.

---

## Key Features

- **Modular Multi-Agent Pipeline**: Orchestrated through a central `CoordinatorAgent` managing specialized agents for data ingestion, validation, NSF-WQI, CCME-WQI, IFS modeling, spatiotemporal inference, distance ranking, alert generation, visualization, reporting, and conversational assistance.
- **Intuitionistic Fuzzy Set (IFS) Modeling**:
  - Continuous membership functions for **pH**, **BOD**, **DO**, and **Fecal Coliform (FC)**.
  - Direct spatiotemporal relations across seasons ($R$), monitoring sites ($X$), and joint season–site cells ($T^*$).
  - Three-term Szmidt–Kacprzyk normalized Hamming and Euclidean distances from the upstream reference station (**Palla**).
  - Pure Atanassov $(0, 0, 1)$ hesitation encoding for missing or non-detect laboratory readings so data gaps are tracked explicitly rather than silently imputed.
- **Regulatory WQI Baselines**: Native `NaN`-masked **NSF-WQI** and **CCME-WQI** evaluators with automatic classification bands and station-level summaries.
- **MCP Integration**: Exposes live tools, prompts, and resources (`run_water_quality_pipeline`, `evaluate_yamuna_data`, `memory://historical_context`, `logs://agent_activity`) for structured agentic interoperability.
- **Interactive Gradio Dashboard**: Role-aware interface (`public` and `official` views) supporting dataset upload, date/site filtering, interactive visual diagnostics, automated alerts, downloadable reports, and an optional LLM assistant.

---

## System Architecture

```text
DPCC Monitoring Dataset (.xlsx)
        │
        ▼
PreprocessingAgent ──► Validation & Seasonal Mapping (Winter, Summer, Monsoon, Post-Monsoon)
        │
        ├──► WQIAgent (NSF-WQI)
        ├──► CCMEAgent (CCME-WQI)
        └──► IFSAgent (Relations R & X, Hesitation π)
                │
                ▼
         InferenceAgent (Joint Spatiotemporal Relation T*)
                │
                ▼
         DistanceAgent (3-Term Hamming & Euclidean Ranking)
                │
                ├──► LLMDecisionAgent & AlertNotificationAgent
                ├──► VisualizationAgent & ReportAgent
                └──► MCP Server & Gradio Dashboard
```

---

## Dataset Overview

The included file `data/data_all_sites_cleaned.xlsx` compiles monthly grab-sample water-quality records published by the **Delhi Pollution Control Committee (DPCC)** from **January 2024 to September 2026** across **9 monitoring locations** along the Delhi stretch of the Yamuna River:

- **Upstream / Entry**: Palla, Wazirabad
- **Urban / Drain-Impacted Stretch**: ISBT Bridge, ITO Bridge, Nizamuddin Bridge, Okhla Barrage, Agra Canal (Okhla), Shahdara Drain Outfall, Asgarpur

Core physicochemical and microbiological indicators include **pH**, **Dissolved Oxygen (DO)**, **Biochemical Oxygen Demand (BOD)**, **Fecal Coliform (FC)**, and **Chemical Oxygen Demand (COD)**.

---

## Repository Structure

```text
yamuna-mas-wqi/
├── app.py                              # Complete MAS-IFS platform, MCP server, and Gradio UI
├── data/
│   └── data_all_sites_cleaned.xlsx     # Compiled DPCC Yamuna dataset (9 sites, 2024–2026)
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Standard Python and local log exclusions
└── README.md                           # Project documentation
```

---

## Installation & Quick Start

### 1. Clone or Extract the Repository
```bash
cd yamuna-mas-wqi
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3. Configure Optional LLM API Keys (Environment Variables)
All core analytical engines (IFS, NSF-WQI, CCME-WQI, alerts, visualizations, and reports) run completely offline without API keys. To enable the conversational `LLMAssistantAgent`, set either environment variable before launching:

```bash
export GROQ_API_KEY="your-groq-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

### 4. Launch the Application
```bash
python app.py
```

Once started, open the local Gradio URL shown in your terminal to explore site rankings, seasonal trends, IFS hesitation diagnostics, regulatory WQI comparisons, and automated alerts.

---

## MCP Endpoints

When `mcp` is installed, `app.py` registers an `MCPServer` instance (`Yamuna_Platform_Server`) exposing:

| Type | Identifier | Description |
| :--- | :--- | :--- |
| **Tool** | `run_water_quality_pipeline` | Runs the full MAS-IFS, NSF-WQI, and CCME-WQI pipeline on a specified dataset and date range |
| **Prompt** | `evaluate_yamuna_data` | Builds a grounded diagnostic prompt from the latest cached pipeline state and historical memory |
| **Resource** | `memory://historical_context` | Returns summarized historical pipeline runs stored in memory |
| **Resource** | `logs://agent_activity` | Returns recent structured inter-agent communication logs |

---

## Data Source

Original monthly water-quality analysis reports are publicly published by the **Delhi Pollution Control Committee (DPCC)** at:
https://dpcc.delhi.gov.in/dpcc/analysis-reports
