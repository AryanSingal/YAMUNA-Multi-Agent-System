# Yamuna River Water-Quality Assessment Platform (MAS-IFS + MCP)

An interactive, uncertainty-aware water-quality assessment platform for the Delhi stretch of the Yamuna River, combining a modular **Multi-Agent System (MAS)**, **Atanassov Intuitionistic Fuzzy Sets (IFS)**, regulatory water-quality indices (**NSF-WQI** and **CCME-WQI**), a **Model Context Protocol (MCP)** server, automated **SMTP critical-alert emails**, and a **Gradio** web interface.

---

## Key Features

- **Modular Multi-Agent Pipeline**: Coordinated by `CoordinatorAgent` across specialized agents for preprocessing, NSF-WQI, CCME-WQI, IFS modeling, joint spatiotemporal inference, distance ranking, AI-assisted decisions, critical alerts, visualizations, PDF/CSV reporting, and conversational Q&A.
- **Intuitionistic Fuzzy Set (IFS) Engine**:
  - Continuous membership functions for **pH**, **BOD**, **DO**, and **Fecal Coliform (FC)**.
  - Direct seasonal ($R$), spatial ($X$), and joint season–site ($T^*$) relations.
  - Three-term Szmidt–Kacprzyk normalized Hamming and Euclidean distances from the upstream reference station (**Palla**).
  - Atanassov $(0, 0, 1)$ hesitation encoding for missing/non-detect laboratory readings so data gaps are preserved explicitly.
- **Regulatory WQI Baselines**: Native `NaN`-masked **NSF-WQI** and **CCME-WQI** scoring with station-level summaries and classification bands.
- **Role-Based Gradio Dashboard**: Separate **Official Administrator** (full controls, API/SMTP settings, PDF/CSV exports, critical alerts) and **Read-Only Viewer** roles.
- **Live MCP Server**: Exposes pipeline tools, diagnostic prompts, memory context, and agent activity logs over SSE.

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
                ├──► LLMDecisionAgent & AlertNotificationAgent (SMTP Alerts)
                ├──► VisualizationAgent & ReportAgent (PDF / CSV)
                └──► MCP Server (:8070) & Gradio Dashboard (:7870)
```

---

## Repository Structure

```text
yamuna-mas-wqi/
├── app.py                              # Complete MAS-IFS platform, MCP server, and Gradio UI
├── data/
│   └── data_all_sites_cleaned.xlsx     # Compiled DPCC Yamuna dataset (9 sites, 2024–2026)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Standard Python and local log exclusions
└── README.md                           # Setup and usage guide
```

---

## Quick Start (First-Time Setup)

### Prerequisites
- **Python 3.10 or higher**

### 1. Install Dependencies
```bash
cd yamuna-mas-wqi
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Launch the Platform
```bash
python app.py
```

By default, this starts:
- **Gradio Web Dashboard**: `http://127.0.0.1:7870`
- **MCP SSE Server**: `http://127.0.0.1:8070`

*(To generate a temporary public Gradio link, run with `GRADIO_SHARE=true python app.py`.)*

### 3. Sign In to the Dashboard
Open `http://127.0.0.1:7870` in your browser and sign in with one of the built-in demo accounts:

| Role | Default Username | Default Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Official Administrator** | `official` | `official123` | Full pipeline, API & SMTP settings, critical alerts, PDF/CSV export |
| **Read-Only Viewer** | `viewer` | `viewer123` | View dashboards, rankings, charts, and non-sensitive alerts |

You can override default credentials and recipient emails before launching using environment variables:
```bash
export YAMUNA_OFFICIAL_USER="official"
export YAMUNA_OFFICIAL_PASS="your-strong-password"
export YAMUNA_OFFICIAL_EMAIL="recipient@example.com"
export YAMUNA_VIEWER_USER="viewer"
export YAMUNA_VIEWER_PASS="viewer-password"
```

---

## Connecting Groq & Gemini API Keys (Optional)

All deterministic analytics (IFS, NSF-WQI, CCME-WQI, charts, rule-based alerts, and PDF/CSV reports) run completely offline without any API keys.

To enable AI-generated decision explanations and the conversational **LLM Assistant**:
1. Get a free API key from either (or both) providers:
   - **Groq (Primary)**: Create a key at [Groq Console](https://console.groq.com/keys) (default model: `openai/gpt-oss-20b`).
   - **Google Gemini (Fallback)**: Create a key at [Google AI Studio](https://aistudio.google.com/apikey) (default model: `gemini-2.5-flash`).
2. Sign in to the Gradio dashboard as **`official`**.
3. Open the **Settings / Configuration** section in the UI, paste your Groq and/or Gemini API key, and click **Save / Connect**.

*(Optional: override the default model names via `GROQ_MODEL` or `GEMINI_MODEL` environment variables before running `python app.py`.)*

---

## Setting Up Automated Email Alerts (SMTP)

When critical water-quality violations are detected, `AlertNotificationAgent` can automatically send email alerts via SSL SMTP (`smtplib.SMTP_SSL`) to official recipients.

### Gmail Setup Steps
1. Enable **2-Step Verification** on your Google Account.
2. Generate a 16-character **App Password** at [Google App Passwords](https://myaccount.google.com/apppasswords) (do **not** use your regular Gmail login password).
3. Set the recipient email address (optional, defaults to `admin@yamuna.gov.in`):
   ```bash
   export YAMUNA_OFFICIAL_EMAIL="your-alert-recipient@gmail.com"
   ```
4. Sign in to the Gradio dashboard as **`official`**, scroll to **Email Alert Settings (SMTP)**, and enter:
   - **SMTP Host**: `smtp.gmail.com`
   - **Port**: `465`
   - **Gmail Address**: your sender Gmail address
   - **App Password**: the 16-character Google App Password generated in Step 2
5. Save the SMTP settings and run the pipeline — any critical alerts will automatically trigger an email notification.

---

## Dataset & Custom File Format

The included dataset `data/data_all_sites_cleaned.xlsx` contains monthly grab-sample records from the **Delhi Pollution Control Committee (DPCC)** covering **January 2024 – September 2026** across 9 Delhi monitoring stations (*Palla, Wazirabad, ISBT Bridge, ITO Bridge, Nizamuddin Bridge, Okhla Barrage, Agra Canal, Shahdara Drain, Asgarpur*).

To analyze your own river dataset via the UI upload control (or by setting `YAMUNA_DATA_PATH=/path/to/file.xlsx`), format your `.xlsx` file with these columns:

| Column | Required | Description / Units |
| :--- | :--- | :--- |
| `Date` | Yes | Sample date (e.g., `2025-06-15`) |
| `Location` | Yes | Monitoring station name |
| `pH` | Yes | Standard pH units (0–14) |
| `DO` | Yes | Dissolved Oxygen (`mg/L`; blank or `NIL` handled via IFS hesitation) |
| `BOD` | Yes | Biochemical Oxygen Demand (`mg/L`) |
| `FC` | Yes | Fecal Coliform (`MPN/100 mL`) |
| `COD` | Optional | Chemical Oxygen Demand (`mg/L`) |

---

## MCP Endpoints

When `mcp` is installed, `app.py` starts `Yamuna_Platform_Server` on `http://127.0.0.1:8070` (configurable via `MCP_HOST` and `MCP_PORT`) with:

| Type | Identifier | Description |
| :--- | :--- | :--- |
| **Tool** | `run_water_quality_pipeline` | Runs the full MAS-IFS, NSF-WQI, and CCME-WQI pipeline on a dataset and date range |
| **Prompt** | `evaluate_yamuna_data` | Builds a grounded diagnostic prompt from the latest cached pipeline state and historical memory |
| **Resource** | `memory://historical_context` | Returns summarized historical pipeline runs stored in memory |
| **Resource** | `logs://agent_activity` | Returns recent structured inter-agent communication logs |

---

## Data Source

Original monthly water-quality analysis reports are publicly published by the **Delhi Pollution Control Committee (DPCC)** at:
https://dpcc.delhi.gov.in/dpcc/analysis-reports
