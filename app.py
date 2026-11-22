"""
Yamuna River Water-Quality Multi-Agent System (MAS) & Intuitionistic Fuzzy Set (IFS) Platform.

Implements the complete multi-agent architecture, Atanassov IFS compositional modeling
(Relations X, R, and T), 3-term normalized Hamming and Euclidean distances, native
NaN-masked NSF-WQI and CCME-WQI baselines, FastMCP tool/resource interface, and Gradio UI
described in:
    "LLM-Driven Multi-Agent Framework for Intuitionistic Fuzzy Water-Quality Assessment
     of the Yamuna River (Delhi Stretch)" (Sadhana, SADH-D-26-03827).

Usage:
    python app.py
"""
import os
os.environ.setdefault("YAMUNA_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "data_all_sites_cleaned.xlsx"))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  YAMUNA RIVER MAS-IFS PLATFORM — Agents, Engines, MCP & Gradio UI     ║
# ║  Author: Aryan | Teacher: Prof. Vibha   Gaur        ║
# ║                                                                          ║
# ║  Login:  official / official123  (full access — upload, run, alerts)    ║
# ║          viewer   / viewer123    (read-only — charts, IFS, CCME, chat)  ║
# ║  Logs:   printed to console + saved to /tmp/yamuna_agents.log          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import os, io, sys, glob, json, math, time, logging, smtplib, warnings, threading, concurrent.futures
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, Image as RLImage, PageBreak)
    RL_OK = True
except ImportError:
    RL_OK = False

try:
    import bcrypt
    BCRYPT_OK = True
except ImportError:
    BCRYPT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT LOGGER  (prints to console + writes to log file)
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = "/tmp/yamuna_agents.log"

class AgentLogger:
    """Centralised logger: console + file + in-memory ring buffer."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._buffer: deque = deque(maxlen=300)
        self._lock = threading.Lock()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(LOG_FILE, mode="a"),
            ],
        )
        self._log = logging.getLogger("Yamuna")
        self._log.setLevel(logging.INFO)

    def _icon(self, level: str) -> str:
        return {"INFO": "ℹ️ ", "MSG": "📨", "RESULT": "✅", "WARN": "⚠️ ",
                "ERROR": "❌", "LLM": "🤖", "MEM": "🧠", "ALERT": "🚨"}.get(level, "·")

    def log(self, sender: str, receiver: str, msg_type: str, detail: str = ""):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        icon = self._icon(msg_type)
        line = f"{icon} [{ts}] {sender:22s} → {receiver:22s}  |  {msg_type:8s}  |  {detail}"
        with self._lock:
            self._log.info(line)
            self._buffer.append({
                "ts": ts, "sender": sender, "receiver": receiver,
                "type": msg_type, "detail": detail, "line": line
            })

    def info(self, agent: str, msg: str):
        self.log(agent, "SYSTEM", "INFO", msg)

    def get_recent(self, n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._buffer)[-n:]

    def get_log_text(self, n: int = 80) -> str:
        return "\n".join(e["line"] for e in self.get_recent(n))


LOGGER = AgentLogger()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MESSAGE BUS  (pub/sub inter-agent events)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    sender:    str
    receiver:  str
    msg_type:  str      # REQUEST / RESULT / EVENT / RETRY_REQUEST / SKIP
    payload:   Any
    ts:        str = field(default_factory=lambda: datetime.now().isoformat())

class MessageBus:
    """Thread-safe in-memory Pub/Sub message bus. All inter-agent messages are logged."""
    def __init__(self):
        self._history: List[AgentMessage] = []
        self._subscribers: Dict[str, List] = {}
        self._lock = threading.Lock()

    def publish(self, sender: str, receiver: str, msg_type: str, payload: Any = None):
        msg = AgentMessage(sender, receiver, msg_type, payload)
        with self._lock:
            self._history.append(msg)
            subs = list(self._subscribers.get(receiver, []))
        LOGGER.log(sender, receiver, msg_type, str(payload)[:120] if payload else "")
        for cb in subs:
            try:
                cb(msg)
            except Exception:
                pass
        return msg

    def subscribe(self, agent_name: str, callback):
        with self._lock:
            self._subscribers.setdefault(agent_name, []).append(callback)

    def get_history(self) -> List[AgentMessage]:
        with self._lock:
            return self._history.copy()


BUS = MessageBus()


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENT MEMORY  (historical states + trend detection + LLM context)
# ─────────────────────────────────────────────────────────────────────────────

MEMORY_FILE = "/tmp/yamuna_memory.json"

class AgentMemory:
    """Stores up to 10 past pipeline runs. Detects trends. Feeds LLM context."""

    def __init__(self, capacity: int = 10):
        self._capacity = capacity
        self._records: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE) as f:
                    data = json.load(f)
                for r in data.get("records", []):
                    self._records.append(r)
                LOGGER.info("AgentMemory", f"Loaded {len(self._records)} historical records")
            except Exception:
                pass

    def _save(self):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump({"records": list(self._records)}, f, default=str)
        except Exception:
            pass

    def store(self, state: "PipelineState"):
        record = {
            "run_id":    state.run_id,
            "ts":        state.run_ts,
            "config":    {
                "start": str(state.config.start_date) if state.config else "",
                "end":   str(state.config.end_date)   if state.config else "",
            },
            "wqi_by_location": {},
            "ifs_mu_by_location": {},
            "dist_from_palla":  {},
            "n_critical": 0,
        }
        if state.wqi_summary is not None:
            for loc, grp in state.wqi_summary.groupby("Location"):
                record["wqi_by_location"][loc] = round(grp["WQI_Mean"].mean(), 3)
        if state.ifs_t is not None:
            for loc, grp in state.ifs_t.groupby("Location"):
                record["ifs_mu_by_location"][loc] = round(grp["Mu_T"].mean(), 3)
        if state.dist_df is not None:
            for _, row in state.dist_df.reset_index().iterrows():
                record["dist_from_palla"][row["Location"]] = round(row["Hamming"], 4)
        if state.decisions:
            record["n_critical"] = sum(1 for d in state.decisions if d.get("level") == "CRITICAL")

        with self._lock:
            self._records.append(record)
            self._save()
        LOGGER.log("AgentMemory", "Coordinator", "MEM",
                   f"Stored run {state.run_id}. Total records: {len(self._records)}")

    def detect_trend(self, location: str) -> str:
        """Returns IMPROVING / DEGRADING / STABLE / UNKNOWN."""
        series = [r["wqi_by_location"].get(location) for r in list(self._records)[-5:]
                  if location in r.get("wqi_by_location", {})]
        if len(series) < 2:
            return "UNKNOWN"
        delta = series[-1] - series[0]
        if delta > 8:   return "IMPROVING"
        if delta < -8:  return "DEGRADING"
        return "STABLE"

    def get_llm_context(self) -> str:
        if not self._records:
            return "No historical data available (first run)."
        lines = ["Historical analysis summary (up to last 5 runs):"]
        for r in list(self._records)[-5:]:
            wqi_str  = ", ".join(f"{k}: {v}" for k, v in r.get("wqi_by_location", {}).items())
            dist_str = ", ".join(f"{k}: {v}" for k, v in r.get("dist_from_palla", {}).items())
            lines.append(
                f"  [{r['ts'][:10]}] Run {r['run_id']} — "
                f"WQI = {{{wqi_str}}} | Hamming = {{{dist_str}}} | Critical alerts = {r.get('n_critical',0)}"
            )
        trends = []
        for loc in LOCATION_ORDER:
            t = self.detect_trend(loc)
            if t != "UNKNOWN":
                trends.append(f"{loc}: {t}")
        if trends:
            lines.append("Detected quality trends: " + ", ".join(trends))
        return "\n".join(lines)

    def count(self) -> int:
        return len(self._records)


MEMORY = AgentMemory()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LOCATION_ORDER = [
    "Palla", "Wazirabad", "ISBT Bridge", "ITO Bridge",
    "Nizamuddin Bridge", "Okhla Barrage", "Agra Canal at Okhla Barrage",
    "Hindon Cut", "Asgarpur"
]
SEASON_ORDER   = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
PARAMS         = ["pH", "DO", "BOD", "FC"]

LOCATION_COORDS = {
    "Palla":                       {"lat": 28.8138, "lon": 77.2100, "desc": "Upstream entry into Delhi — reference site"},
    "Wazirabad":                   {"lat": 28.7131, "lon": 77.2319, "desc": "Delhi water intake — upstream of Najafgarh drain"},
    "ISBT Bridge":                 {"lat": 28.6714, "lon": 77.2338, "desc": "Downstream of Najafgarh drain confluence"},
    "ITO Bridge":                  {"lat": 28.6284, "lon": 77.2535, "desc": "Central urban stretch — severe sewage load"},
    "Nizamuddin Bridge":           {"lat": 28.5916, "lon": 77.2715, "desc": "Downstream of Barapullah/Sen Nursing Home drains"},
    "Okhla Barrage":               {"lat": 28.5444, "lon": 77.3111, "desc": "Okhla Barrage impoundment"},
    "Agra Canal at Okhla Barrage": {"lat": 28.5454, "lon": 77.3111, "desc": "Agra Canal offtake at Okhla Barrage"},
    "Hindon Cut":                  {"lat": 28.5310, "lon": 77.3260, "desc": "Hindon Cut channel confluence"},
    "Asgarpur":                    {"lat": 28.4657, "lon": 77.3391, "desc": "Downstream exit after Shahdara & Tuglakabad drains"},
}
WEIGHTS        = {"DO": 0.309, "FC": 0.291, "pH": 0.200, "BOD": 0.200}

NSF_CLASSES = [
    (91, 100, "Excellent",  "#1DB954"),
    (71,  90, "Good",       "#57C785"),
    (51,  70, "Medium",     "#F0C040"),
    (26,  50, "Bad",        "#F07020"),
    ( 0,  25, "Very Bad",   "#E03030"),
]

CCME_CLASSES = [
    (95, 100, "Excellent",  "#16A34A"),
    (80,  94, "Good",       "#4ADE80"),
    (65,  79, "Fair",       "#FACC15"),
    (45,  64, "Marginal",   "#FB923C"),
    ( 0,  44, "Poor",       "#EF4444"),
]

# CCME objectives
CCME_OBJECTIVES = {
    "pH":  {"type": "range",   "min": 6.5,  "max": 8.5},
    "DO":  {"type": "minimum", "obj": 6.0},
    "BOD": {"type": "maximum", "obj": 5.0},
    "FC":  {"type": "maximum", "obj": 200.0},
}

ALERT_COLORS = {"CRITICAL": "#DC2626", "WARNING": "#D97706", "INFO": "#2563EB", "OK": "#16A34A"}

LOC_COLORS = {
    "Palla":                       "#16A34A",
    "Wazirabad":                   "#2563EB",
    "ISBT Bridge":                 "#7C3AED",
    "ITO Bridge":                  "#D97706",
    "Nizamuddin Bridge":           "#EA580C",
    "Okhla Barrage":               "#DC2626",
    "Agra Canal at Okhla Barrage": "#BE123C",
    "Hindon Cut":                  "#9333EA",
    "Asgarpur":                    "#7F1D1D",
}

FALLBACK_PALETTE = [
    "#0284C7", "#0D9488", "#4F46E5", "#9333EA", "#DB2777",
    "#EA580C", "#CA8A04", "#16A34A", "#DC2626", "#475569",
]

# Central configurable runtime parameters (override via env vars or AnalysisConfig)
POLICY_CONFIG = {
    "groq_model":       os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
    "gemini_model":     os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "pi_nil":           float(os.environ.get("IFS_PI_NIL", "0.0")),
    "pi_missing":       float(os.environ.get("IFS_PI_MISSING", "1.0")),
    "remediation_days": int(os.environ.get("REMEDIATION_DAYS", "30")),
    "urgency_map":      {"CRITICAL": 9, "WARNING": 5, "INFO": 2},
    "max_workers":      int(os.environ.get("MAS_MAX_WORKERS", "4")),
}

def resolve_locations(df: pd.DataFrame, preferred_order: Optional[List[str]] = None) -> List[str]:
    """Dynamically detects all monitoring locations in any river dataset without hardcoding."""
    present = list(dict.fromkeys(df["Location"].astype(str).str.strip().dropna().tolist()))
    order_ref = preferred_order if preferred_order is not None else LOCATION_ORDER
    ordered = [l for l in order_ref if l in present]
    extras  = sorted([l for l in present if l not in ordered])
    return ordered + extras

def get_location_color(loc: str, idx: int = 0) -> str:
    if loc in LOC_COLORS:
        return LOC_COLORS[loc]
    return FALLBACK_PALETTE[abs(hash(loc) + idx) % len(FALLBACK_PALETTE)]

def get_season(month: int) -> str:
    if month in (12, 1, 2):    return "Winter"
    if month in (3, 4, 5):     return "Summer"
    if month in (6, 7, 8, 9):  return "Monsoon"
    return "Post-Monsoon"

def _clean_zero(v: float, ndigits: int = 4) -> float:
    r = round(float(v), ndigits)
    return 0.0 if abs(r) < 1e-9 else r

def fmt_ifs(mu, nu, pi) -> str:
    m = _clean_zero(mu, 3)
    n = _clean_zero(nu, 3)
    p = _clean_zero(max(0.0, pi), 3)
    return f"({m:.3f}, {n:.3f}, {p:.3f})"

def nsf_class(score: float) -> Tuple[str, str]:
    for lo, hi, lbl, col in NSF_CLASSES:
        if score >= lo: return lbl, col
    return "Very Bad", "#E03030"

def nsf_class_vec(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized NSF-WQI class & color lookup (NaN-safe, releases GIL)."""
    s = np.asarray(scores, dtype=float)
    conds = [s >= 91, s >= 71, s >= 51, s >= 26]
    lbls = np.select(conds, ["Excellent", "Good", "Medium", "Bad"], default="Very Bad")
    cols = np.select(conds, ["#1DB954", "#57C785", "#F0C040", "#F07020"], default="#E03030")
    nan = np.isnan(s)
    return np.where(nan, "No data", lbls), np.where(nan, "#9CA3AF", cols)

def get_season_vec(months: np.ndarray) -> np.ndarray:
    """Vectorized IMD season mapping (June = Monsoon, releases GIL)."""
    m = np.asarray(months, dtype=int)
    return np.select(
        [(m == 12) | (m <= 2), (m >= 3) & (m <= 5), (m >= 6) & (m <= 9)],
        ["Winter", "Summer", "Monsoon"],
        default="Post-Monsoon",
    )

def ccme_class(score: float) -> Tuple[str, str]:
    for lo, hi, lbl, col in CCME_CLASSES:
        if score >= lo: return lbl, col
    return "Poor", "#EF4444"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserSession:
    user_id:  str
    username: str
    role:     str      # "official" | "viewer"
    name:     str
    email:    str = ""

@dataclass
class IFSTriplet:
    mu: float; nu: float; pi: float = field(init=False)
    def __post_init__(self):
        self.mu = _clean_zero(max(0.0, min(1.0, self.mu)), 4)
        self.nu = _clean_zero(max(0.0, min(1.0 - self.mu, self.nu)), 4)
        self.pi = _clean_zero(max(0.0, 1.0 - self.mu - self.nu), 4)
    def __repr__(self): return fmt_ifs(self.mu, self.nu, self.pi)

@dataclass
class AnalysisConfig:
    start_date:         Optional[str]       = None   # None -> auto-detect from dataset min date
    end_date:           Optional[str]       = None   # None -> auto-detect from dataset max date
    locations:          Optional[List[str]] = None   # None -> auto-detect ALL locations in dataset
    params:             List[str]           = field(default_factory=lambda: list(PARAMS))
    weights:            Dict[str, float]    = field(default_factory=lambda: dict(WEIGHTS))
    reference_location: Optional[str]       = None   # None -> "Palla" if present, else first station
    pi_nil:             float               = field(default_factory=lambda: POLICY_CONFIG["pi_nil"])
    pi_missing:         float               = field(default_factory=lambda: POLICY_CONFIG["pi_missing"])
    max_workers:        int                 = field(default_factory=lambda: POLICY_CONFIG["max_workers"])
    parallel_threshold: int                 = 75000  # Adaptive DAG threshold (sequential <75k, DAG-parallel >=75k on 2 CPUs)

@dataclass
class AgentResult:
    """Returned by each agent to Coordinator. Allows dynamic re-routing."""
    status:       str       # "OK" | "WARN" | "SKIP" | "RETRY" | "ERROR"
    data:         Any       = None
    message:      str       = ""
    retry_reason: str       = ""

@dataclass
class PipelineState:
    raw_df:         Optional[pd.DataFrame]   = None
    config:         Optional[AnalysisConfig] = None
    cleaned_df:     Optional[pd.DataFrame]   = None
    imputed_mask:   Optional[pd.DataFrame]   = None
    validation_log: Dict                     = field(default_factory=dict)
    wqi_df:         Optional[pd.DataFrame]   = None
    wqi_summary:    Optional[pd.DataFrame]   = None
    ccme_df:        Optional[pd.DataFrame]   = None
    ifs_df:         Optional[pd.DataFrame]   = None
    ifs_r:          Optional[pd.DataFrame]   = None
    ifs_x:          Optional[pd.DataFrame]   = None
    ifs_t:          Optional[pd.DataFrame]   = None
    dist_df:        Optional[pd.DataFrame]   = None
    decisions:      List[Dict]               = field(default_factory=list)
    errors:         List[str]                = field(default_factory=list)
    run_id:         str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_ts:         str = field(default_factory=lambda: datetime.now().isoformat())
    agent_log_snapshot: List[Dict] = field(default_factory=list)
    def ok(self): return len(self.errors) == 0



# ─────────────────────────────────────────────────────────────────────────────
# 6. IFS MATHEMATICAL ENGINE (Step 01 — Exact Missing / Zero-NIL Hesitation)
#    Pipeline per observation:
#      1) Compute membership μ(x) from regulatory piecewise-linear functions.
#      2) Compute hesitation π ONLY for missing / censored zero (NIL) data:
#           - Normal measured value (non-missing, non-NIL): π = 0.0
#           - Censored NIL / zero reading (DO == 0.0 reported as NIL in DPCC):
#             π = PI_NIL (0.15) — detection-limit / anoxic censoring uncertainty
#           - Missing / dry cell (NaN imputed by station-season median):
#             π = min(1 - μ, PI_MISSING) where PI_MISSING = 0.25
#           - Unimputed NaN: total ignorance (μ = 0.0, ν = 0.0, π = 1.0)
#      3) Compute non-membership ν = max(0.0, 1.0 - μ - π).
#      4) Effective Membership E = μ - (ν × π).
# ─────────────────────────────────────────────────────────────────────────────
class IFSEngine:
    PI_NIL     = 0.0    # Unused when NIL readings are stored as NaN in data_all_sites_cleaned.xlsx
    PI_MISSING = 1.0    # Pure Atanassov total ignorance (0, 0, 1) for every missing / NaN cell

    @staticmethod
    def _mu_vector(param: str, vals: np.ndarray) -> np.ndarray:
        """Vectorized piecewise-linear regulatory membership μ(x) in [0, 1]."""
        v = np.asarray(vals, dtype=float)
        if param == "pH":
            conds = [
                (v >= 7.0) & (v <= 8.0),
                (v >= 6.5) & (v < 7.0),
                (v > 8.0) & (v <= 8.5),
                (v >= 6.0) & (v < 6.5),
                (v > 8.5) & (v <= 9.0),
            ]
            choices = [
                1.0,
                0.3 + ((v - 6.5) / 0.5) * 0.7,
                0.3 + ((8.5 - v) / 0.5) * 0.7,
                ((v - 6.0) / 0.5) * 0.3,
                ((9.0 - v) / 0.5) * 0.3,
            ]
            return np.clip(np.select(conds, choices, default=0.0), 0.0, 1.0)

        if param == "BOD":
            conds = [
                v <= 3.0,
                (v > 3.0) & (v <= 5.0),
                (v > 5.0) & (v <= 15.0),
                (v > 15.0) & (v <= 30.0),
            ]
            choices = [
                1.0,
                1.0 - ((v - 3.0) / 2.0) * 0.3,
                0.7 - ((v - 5.0) / 10.0) * 0.5,
                0.2 - ((v - 15.0) / 15.0) * 0.2,
            ]
            return np.clip(np.select(conds, choices, default=0.0), 0.0, 1.0)

        if param == "DO":
            conds = [
                v >= 6.0,
                (v >= 4.0) & (v < 6.0),
                (v >= 2.0) & (v < 4.0),
                (v > 0.0) & (v < 2.0),
            ]
            choices = [
                1.0,
                0.7 + ((v - 4.0) / 2.0) * 0.3,
                0.2 + ((v - 2.0) / 2.0) * 0.5,
                (v / 2.0) * 0.2,
            ]
            return np.clip(np.select(conds, choices, default=0.0), 0.0, 1.0)

        if param == "FC":
            conds = [
                v <= 0.0,
                (v > 0.0) & (v <= 100.0),
                (v > 100.0) & (v <= 1000.0),
                (v > 1000.0) & (v <= 10000.0),
            ]
            choices = [
                1.0,
                1.0 - (v / 100.0) * 0.3,
                0.7 - ((v - 100.0) / 900.0) * 0.5,
                0.2 - ((v - 1000.0) / 9000.0) * 0.2,
            ]
            return np.clip(np.select(conds, choices, default=0.0), 0.0, 1.0)

        raise ValueError(f"Unsupported water-quality parameter: {param}")

    @classmethod
    def membership(cls, param: str, val: float, is_missing: bool = False) -> IFSTriplet:
        """Scalar IFS fuzzification (exact parity with fuzzify_dataframe): missing/NaN -> (0, 0, 1); measured -> (μ, 1-μ, 0)."""
        if is_missing or val is None or (isinstance(val, float) and np.isnan(val)):
            return IFSTriplet(0.0, 0.0)  # Pure Atanassov total ignorance: (μ=0, ν=0, π=1)
        v = float(val)
        mu = float(cls._mu_vector(param, np.array([v]))[0])
        nu = max(0.0, 1.0 - mu)
        return IFSTriplet(round(mu, 4), round(nu, 4))

    @classmethod
    def fuzzify_dataframe(cls, df: pd.DataFrame, params: List[str],
                          imputed_mask: Optional[pd.DataFrame] = None,
                          pi_nil: Optional[float] = None,
                          pi_missing: Optional[float] = None) -> pd.DataFrame:
        """Fully vectorized IFS fuzzification (no iterrows). π > 0 ONLY for missing or NIL (0) data."""
        out = pd.DataFrame({
            "Date": df["Date"].values if "Date" in df.columns else pd.NaT,
            "Season": df["Season"].values,
            "Location": df["Location"].values,
        })

        for p in params:
            vals = pd.to_numeric(df[p], errors="coerce").to_numpy(dtype=float)
            nan_mask = np.isnan(vals)
            safe_vals = np.where(nan_mask, 0.0, vals)

            mu = cls._mu_vector(p, safe_vals)
            if imputed_mask is not None and p in imputed_mask.columns:
                m_mask = imputed_mask[p].to_numpy(dtype=bool)
            else:
                m_mask = np.zeros(len(vals), dtype=bool)

            # Pure Atanassov IFS:
            #   - Any missing / NaN cell (or flagged in imputed_mask) -> total ignorance (0, 0, 1) when pi_missing == 1.0
            #   - All measured values -> crisp-hesitation fuzzy triplet (mu, 1 - mu, 0.0)
            missing_any = nan_mask | m_mask
            _pi_miss = cls.PI_MISSING if pi_missing is None else float(pi_missing)

            if _pi_miss >= 0.9999:
                mu = np.where(missing_any, 0.0, mu)
                nu = np.where(missing_any, 0.0, np.maximum(0.0, 1.0 - mu))
                pi = np.where(missing_any, 1.0, 0.0)
            else:
                # For sensitivity testing pi_missing in [0, 1): missing cells split (1 - pi_miss) equally or keep 0 membership with nu = 1 - pi_miss
                mu = np.where(missing_any, 0.0, mu)
                pi = np.where(missing_any, _pi_miss, 0.0)
                nu = np.where(missing_any, np.maximum(0.0, 1.0 - _pi_miss), np.maximum(0.0, 1.0 - mu))

            out[f"mu_{p}"] = np.round(mu, 4)
            out[f"nu_{p}"] = np.round(nu, 4)
            out[f"pi_{p}"] = np.round(pi, 4)

        return out

    @staticmethod
    def effective_membership(mu: float, nu: float, pi: float) -> float:
        return round(float(mu) - (float(nu) * float(pi)), 4)

    @staticmethod
    def hamming_3term(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / 2.0

    @staticmethod
    def euclidean_3term(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. NSF WQI ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class NSFWQIEngine:
    @staticmethod
    def q_DO(v):
        if v>=8: return 100.0
        if v<=0: return 0.0
        if v>=6: return 80+(v-6)/2*20
        if v>=4: return 40+(v-4)/2*40
        return max(0,(v/4)*40)

    @staticmethod
    def q_BOD(v):
        if v<=1: return 100.0
        if v>=30: return 2.0
        if v<=5:  return 100-(v-1)/4*30
        if v<=15: return 70-(v-5)/10*40
        return 30-(v-15)/15*28

    @staticmethod
    def q_pH(v):
        if 7.0<=v<=8.0: return 100.0
        if 6.5<=v<7.0:  return 70+(v-6.5)/0.5*30
        if 8.0<v<=8.5:  return 70+(8.5-v)/0.5*30
        if 6.0<=v<6.5:  return 30+(v-6.0)/0.5*40
        if 8.5<v<=9.0:  return 30+(9.0-v)/0.5*40
        if v<6.0:       return max(0,(v-4)/2*30)
        return max(0,(11-v)/2*30)

    @staticmethod
    def q_FC(v):
        if v<=0:       return 100.0
        if v>=100000:  return 2.0
        if v<=100:     return 100-v/100*20
        if v<=1000:    return 80-(v-100)/900*30
        if v<=10000:   return 50-(v-1000)/9000*30
        return 20-(v-10000)/90000*18

    @classmethod
    def compute_row(cls, row, weights) -> float:
        q = {"DO": cls.q_DO(row["DO"]), "BOD": cls.q_BOD(row["BOD"]),
             "pH": cls.q_pH(row["pH"]), "FC":  cls.q_FC(row["FC"])}
        w_sum = sum(weights[p] for p in PARAMS)
        return round(sum((weights[p] / w_sum) * q[p] for p in PARAMS), 3)

    @classmethod
    def compute_dataframe(cls, df: pd.DataFrame, weights: Dict[str, float],
                          missing: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Fully vectorized NSF-WQI with native NaN/missing-mask exclusion and weight renormalization."""
        raw_do  = pd.to_numeric(df["DO"],  errors="coerce").to_numpy(dtype=float)
        raw_bod = pd.to_numeric(df["BOD"], errors="coerce").to_numpy(dtype=float)
        raw_ph  = pd.to_numeric(df["pH"],  errors="coerce").to_numpy(dtype=float)
        raw_fc  = pd.to_numeric(df["FC"],  errors="coerce").to_numpy(dtype=float)

        do  = np.nan_to_num(raw_do,  nan=0.0)
        bod = np.nan_to_num(raw_bod, nan=0.0)
        ph  = np.nan_to_num(raw_ph,  nan=7.0)
        fc  = np.nan_to_num(raw_fc,  nan=0.0)

        q_do = np.select(
            [do >= 8, do <= 0, do >= 6, do >= 4],
            [100.0, 0.0, 80.0 + (do - 6.0) / 2.0 * 20.0, 40.0 + (do - 4.0) / 2.0 * 40.0],
            default=np.maximum(0.0, (do / 4.0) * 40.0),
        )
        q_bod = np.select(
            [bod <= 1, bod >= 30, bod <= 5, bod <= 15],
            [100.0, 2.0, 100.0 - (bod - 1.0) / 4.0 * 30.0, 70.0 - (bod - 5.0) / 10.0 * 40.0],
            default=30.0 - (bod - 15.0) / 15.0 * 28.0,
        )
        q_ph = np.select(
            [(ph >= 7.0) & (ph <= 8.0), (ph >= 6.5) & (ph < 7.0), (ph > 8.0) & (ph <= 8.5),
             (ph >= 6.0) & (ph < 6.5), (ph > 8.5) & (ph <= 9.0), ph < 6.0],
            [100.0, 70.0 + (ph - 6.5) / 0.5 * 30.0, 70.0 + (8.5 - ph) / 0.5 * 30.0,
             30.0 + (ph - 6.0) / 0.5 * 40.0, 30.0 + (9.0 - ph) / 0.5 * 40.0,
             np.maximum(0.0, (ph - 4.0) / 2.0 * 30.0)],
            default=np.maximum(0.0, (11.0 - ph) / 2.0 * 30.0),
        )
        q_fc = np.select(
            [fc <= 0, fc >= 100000, fc <= 100, fc <= 1000, fc <= 10000],
            [100.0, 2.0, 100.0 - fc / 100.0 * 20.0,
             80.0 - (fc - 100.0) / 900.0 * 30.0, 50.0 - (fc - 1000.0) / 9000.0 * 30.0],
            default=20.0 - (fc - 10000.0) / 90000.0 * 18.0,
        )
        raw_map = {"DO": raw_do, "BOD": raw_bod, "pH": raw_ph, "FC": raw_fc}
        q_map   = {"DO": q_do,   "BOD": q_bod,   "pH": q_ph,   "FC": q_fc}
        num = np.zeros(len(df), dtype=float)
        den = np.zeros(len(df), dtype=float)
        for p in PARAMS:
            is_miss = np.isnan(raw_map[p])
            if missing is not None and p in missing.columns:
                is_miss = is_miss | missing[p].to_numpy(dtype=bool)
            w_eff = np.where(is_miss, 0.0, float(weights[p]))
            num += w_eff * q_map[p]
            den += w_eff
        wqi = np.where(den > 0.0, num / np.maximum(den, 1e-12), np.nan)
        return np.round(wqi, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CCME WQI ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class CCMEWQIEngine:
    OBJECTIVES = CCME_OBJECTIVES

    @classmethod
    def compute(cls, df: pd.DataFrame, missing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if "Season" not in df.columns:
            df = df.copy()
            dt = pd.to_datetime(df["Date"], errors="coerce")
            df["Season"] = get_season_vec(dt.dt.month.fillna(1).to_numpy())
        rows = []
        for (loc, season), grp in df.groupby(["Location", "Season"], sort=False):
            m_grp = missing.loc[grp.index] if (missing is not None) else None
            result = cls._compute_group(grp, loc, season, missing_grp=m_grp)
            rows.append(result)
        return pd.DataFrame(rows)

    @classmethod
    def _compute_group(cls, grp: pd.DataFrame, location: str, season: str,
                       missing_grp: Optional[pd.DataFrame] = None) -> Dict:
        """Vectorized NumPy CCME-WQI excluding missing/NaN cells from total_tests and excursions."""
        total_tests      = 0
        tested_variables = 0
        failed_variables = set()
        failed_tests     = 0
        exc_sum          = 0.0
        exc_count        = 0

        for p in PARAMS:
            obj = cls.OBJECTIVES[p]
            raw = pd.to_numeric(grp[p], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(raw)
            if missing_grp is not None and p in missing_grp.columns:
                valid = valid & (~missing_grp[p].to_numpy(dtype=bool))
            arr = raw[valid]
            if arr.size == 0:
                continue
            tested_variables += 1
            total_tests += int(arr.size)

            if obj["type"] == "maximum":
                thr = float(obj["obj"])
                fail_mask = arr > thr
                if np.any(fail_mask):
                    failed_variables.add(p)
                    exc = (arr[fail_mask] / thr) - 1.0
                    failed_tests += int(exc.size)
                    exc_count += int(exc.size)
                    exc_sum += float(np.sum(exc))

            elif obj["type"] == "minimum":
                thr = float(obj["obj"])
                fail_mask = arr < thr
                if np.any(fail_mask):
                    failed_variables.add(p)
                    exc = (thr / np.maximum(arr[fail_mask], 0.001)) - 1.0
                    failed_tests += int(exc.size)
                    exc_count += int(exc.size)
                    exc_sum += float(np.sum(exc))

            elif obj["type"] == "range":
                lo, hi = float(obj["min"]), float(obj["max"])
                lo_mask = arr < lo
                hi_mask = arr > hi
                if np.any(lo_mask) or np.any(hi_mask):
                    failed_variables.add(p)
                    if np.any(lo_mask):
                        exc_lo = (lo / np.maximum(arr[lo_mask], 0.001)) - 1.0
                        failed_tests += int(exc_lo.size)
                        exc_count += int(exc_lo.size)
                        exc_sum += float(np.sum(exc_lo))
                    if np.any(hi_mask):
                        exc_hi = (arr[hi_mask] / hi) - 1.0
                        failed_tests += int(exc_hi.size)
                        exc_count += int(exc_hi.size)
                        exc_sum += float(np.sum(exc_hi))

        n_vars = tested_variables
        F1 = (len(failed_variables) / n_vars) * 100.0 if n_vars > 0 else 0.0
        F2 = (failed_tests / total_tests) * 100.0     if total_tests > 0 else 0.0

        nse = exc_sum / total_tests if total_tests > 0 else 0.0
        F3  = nse / (0.01 * nse + 0.01) if nse > 0 else 0.0


        ccme_wqi = 100 - (math.sqrt(F1**2 + F2**2 + F3**2) / 1.732)
        ccme_wqi = round(max(0.0, min(100.0, ccme_wqi)), 2)
        label, color = ccme_class(ccme_wqi)

        return {
            "Location":           location,
            "Season":             season,
            "CCME_WQI":           ccme_wqi,
            "CCME_Class":         label,
            "CCME_Color":         color,
            "F1_Scope":           round(F1, 2),
            "F2_Frequency":       round(F2, 2),
            "F3_Amplitude":       round(F3, 2),
            "Failed_Variables":   ", ".join(sorted(failed_variables)) or "None",
            "Failed_Tests":       failed_tests,
            "Total_Tests":        total_tests,
            "Excursion_Count":    exc_count,
            "Avg_Excursion":      round(exc_sum / exc_count, 3) if exc_count > 0 else 0.0,
        }

    @staticmethod
    def _test(param: str, val: float, obj: Dict) -> Tuple[bool, float]:
        if obj["type"] == "maximum":
            threshold = obj["obj"]
            if val > threshold:
                return True, (val / threshold) - 1.0
            return False, 0.0
        elif obj["type"] == "minimum":
            threshold = obj["obj"]
            if val < threshold:
                return True, (threshold / max(val, 0.001)) - 1.0
            return False, 0.0
        elif obj["type"] == "range":
            lo, hi = obj["min"], obj["max"]
            if val < lo:
                return True, (lo / max(val, 0.001)) - 1.0
            if val > hi:
                return True, (val / hi) - 1.0
            return False, 0.0
        return False, 0.0

    @staticmethod
    def interpretation(row: pd.Series) -> str:
        wqi   = row["CCME_WQI"]
        cls_  = row["CCME_Class"]
        f1,f2 = row["F1_Scope"], row["F2_Frequency"]
        fv    = row["Failed_Variables"]
        templates = {
            "Excellent": f"Water quality at {row['Location']} ({row['Season']}) is EXCELLENT (CCME = {wqi}). "
                         f"All monitored parameters meet Canadian guidelines with high confidence.",
            "Good":      f"Water quality is GOOD (CCME = {wqi}). Minor exceedances detected in: {fv}. "
                         f"{f2:.1f}% of tests failed — generally protected.",
            "Fair":      f"Water quality is FAIR (CCME = {wqi}). Parameters failing: {fv}. "
                         f"Scope F1={f1:.1f}%, Frequency F2={f2:.1f}%. Occasional threats to aquatic life.",
            "Marginal":  f"Water quality is MARGINAL (CCME = {wqi}). Significant failures in: {fv}. "
                         f"F1={f1:.1f}%, F2={f2:.1f}%. Frequent threats — intervention recommended.",
            "Poor":      f"Water quality is POOR (CCME = {wqi}). Critical failures in: {fv}. "
                         f"F1={f1:.1f}%, F2={f2:.1f}%. Water quality almost always threatened or impaired.",
        }
        return templates.get(cls_, f"CCME WQI = {wqi} ({cls_})")


# ─────────────────────────────────────────────────────────────────────────────
# 9. AUTH AGENT  (with email storage + multi-user + all-official email list)
# ─────────────────────────────────────────────────────────────────────────────

# Credentials read from environment variables (fallback only for local demo mode)
DEFAULT_USERS = [
    {"user_id": "off_001", "username": os.environ.get("YAMUNA_OFFICIAL_USER", "official"),
     "password": os.environ.get("YAMUNA_OFFICIAL_PASS", "official123"),
     "role": "official", "name": "Official Administrator",
     "email": os.environ.get("YAMUNA_OFFICIAL_EMAIL", "admin@yamuna.gov.in")},
    {"user_id": "view_001", "username": os.environ.get("YAMUNA_VIEWER_USER", "viewer"),
     "password": os.environ.get("YAMUNA_VIEWER_PASS", "viewer123"),
     "role": "viewer", "name": "Read-Only Viewer",
     "email": os.environ.get("YAMUNA_VIEWER_EMAIL", "viewer@yamuna.gov.in")},
]

class AuthAgent:
    NAME = "AuthAgent"

    def __init__(self, users_file: Optional[str] = None):
        self._users: Dict[str, Dict] = {}
        if users_file and os.path.exists(users_file):
            self._load_file(users_file)
        else:
            self._load_defaults()

    def _load_defaults(self):
        for u in DEFAULT_USERS:
            entry = dict(u)
            try:
                if BCRYPT_OK:
                    entry["password_hash"] = bcrypt.hashpw(u["password"].encode(), bcrypt.gensalt()).decode()
            except Exception:
                pass
            self._users[u["username"]] = entry

    def _load_file(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for u in data.get("users", []):
            self._users[u["username"]] = u

    def login(self, username: str, password: str) -> Tuple[Optional[UserSession], str]:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", f"login:{username}")
        u = self._users.get(username.strip())
        if not u:
            BUS.publish(self.NAME, "Coordinator", "RESULT", "login_failed:user_not_found")
            return None, "User not found."
        stored_hash = u.get("password_hash", "")
        try:
            ok = (BCRYPT_OK and stored_hash and
                  bcrypt.checkpw(password.encode(), stored_hash.encode()))
        except Exception:
            ok = False
        if not ok:
            ok = (password == u.get("password", ""))
        if not ok:
            BUS.publish(self.NAME, "Coordinator", "RESULT", "login_failed:wrong_password")
            return None, "Incorrect password."
        session = UserSession(u["user_id"], username, u["role"], u["name"], u.get("email",""))
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"login_ok:{username}:{u['role']}")
        return session, f"Welcome, {u['name']}  [{u['role'].upper()}]"

    def all_official_emails(self) -> List[str]:
        return [u["email"] for u in self._users.values()
                if u.get("role") == "official" and u.get("email")]

    def add_user(self, username: str, password: str, role: str, name: str, email: str = ""):
        entry = {"user_id": f"{role[:3]}_{len(self._users):03d}", "username": username,
                 "role": role, "name": name, "email": email, "password": password}
        try:
            if BCRYPT_OK:
                entry["password_hash"] = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        except Exception:
            pass
        self._users[username] = entry
        BUS.publish(self.NAME, "SYSTEM", "INFO", f"New user registered: {username} ({role})")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PREPROCESSING AGENT (Real-Data Safe + Station-Season Imputation & Mask)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "date": "Date", "sampling date": "Date", "month": "Date",
    "location": "Location", "station": "Location", "site": "Location", "monitoring location": "Location",
    "ph": "pH",
    "do": "DO", "do (mg/l)": "DO", "dissolved oxygen": "DO", "dissolved oxygen (mg/l)": "DO",
    "bod": "BOD", "bod (mg/l)": "BOD", "biochemical oxygen demand": "BOD",
    "fc": "FC", "fecal coliform": "FC", "faecal coliform": "FC",
    "fc (mpn/100ml)": "FC", "fc (cfu/100ml)": "FC", "fecal coliform (mpn/100ml)": "FC",
}

def normalize_dpcc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes common DPCC Excel/CSV column headers to Date, Location, pH, DO, BOD, FC."""
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)

def load_real_dpcc_data(explicit_path: Optional[str] = None, allow_colab_upload: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Automatically locates and loads the real DPCC Yamuna dataset (never synthetic).
    Search order: explicit_path -> common /content & local paths -> Colab upload dialog.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    search_dirs = [
        "/content/drive/MyDrive/River_Data",
        "/content",
        ".",
        "/content/drive/MyDrive",
    ]
    priority_names = [
        "data_all_sites_cleaned.xlsx", "data_all_sites.xlsx", "data.xlsx", "data.csv",
        "yamuna_data.xlsx", "yamuna_data.csv", "dpcc_data.xlsx",
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for name in priority_names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    candidates.append(p)
            # Also scan for any non-sample xlsx/csv in /content
            if d == "/content":
                for p in sorted(glob.glob(os.path.join(d, "*.xlsx")) + glob.glob(os.path.join(d, "*.csv"))):
                    if "sample" not in os.path.basename(p).lower() and p not in candidates:
                        candidates.append(p)

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
            df = normalize_dpcc_columns(df)
            req = {"Date", "Location", "pH", "DO", "BOD", "FC"}
            if req.issubset(set(df.columns)):
                LOGGER.info("DataLoader", f"Auto-loaded real DPCC dataset: {path} ({len(df)} rows)")
                return df, path

    if allow_colab_upload:
        try:
#             from google.colab import files  # type: ignore
            print("📂 Real DPCC dataset not found in /content/. Please upload your DPCC Excel/CSV file:")
            uploaded = files.upload()
            if uploaded:
                fname = next(iter(uploaded.keys()))
                path = os.path.abspath(fname)
                df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
                df = normalize_dpcc_columns(df)
                LOGGER.info("DataLoader", f"Loaded uploaded DPCC dataset: {path} ({len(df)} rows)")
                return df, path
        except ImportError:
            pass

    raise FileNotFoundError(
        "Real DPCC dataset not found. Upload your DPCC Excel/CSV file to /content/data.xlsx "
        "(required columns: Date, Location, pH, DO, BOD, FC). Synthetic fallback is disabled."
    )


class PreprocessingAgent:
    NAME = "PreprocessingAgent"

    # Physical sanity bounds (extreme high FC is flagged, NOT silently dropped)
    PHYSICAL_BOUNDS = {
        "pH":  (0.0, 14.0),
        "DO":  (0.0, 20.0),
        "BOD": (0.0, 500.0),
        "FC":  (0.0, None),   # No upper cap so valid extreme ITO/Okhla FC readings are preserved
    }

    def run(self, df: pd.DataFrame, config: AnalysisConfig) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", f"preprocess {len(df)} rows")
        log = {"original_rows": len(df), "issues": [], "warnings": [], "imputed_cells": 0}
        df = normalize_dpcc_columns(df)

        req = {"Date", "Location"} | set(config.params)
        miss = req - set(df.columns)
        if miss:
            msg = f"Missing required columns: {miss}"
            BUS.publish(self.NAME, "Coordinator", "ERROR", msg)
            return AgentResult("ERROR", message=msg)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad_dates = int(df["Date"].isna().sum())
        if bad_dates:
            log["issues"].append(f"{bad_dates} unparseable date rows dropped")
            df = df.dropna(subset=["Date"])

        if config.start_date:
            df = df[df["Date"] >= pd.to_datetime(config.start_date)]
        if config.end_date:
            df = df[df["Date"] <= pd.to_datetime(config.end_date)]

        df["Location"] = df["Location"].astype(str).str.strip()
        if config.locations:
            df = df[df["Location"].isin(config.locations)]

        if len(df) == 0:
            msg = "No data remaining after date/location filtering. Check start_date, end_date, and locations."
            BUS.publish(self.NAME, "Coordinator", "ERROR", msg)
            return AgentResult("ERROR", message=msg)

        # Single authoritative IMD season mapping (June = Monsoon)
        df["Season"] = df["Date"].dt.month.apply(get_season)
        df = df.reset_index(drop=True)

        # Coerce numerics, check physical bounds, and record missingness mask
        imputed_mask = pd.DataFrame(False, index=df.index, columns=config.params)

        for p in config.params:
            df[p] = pd.to_numeric(df[p], errors="coerce")
            lo, hi = self.PHYSICAL_BOUNDS.get(p, (None, None))
            if lo is not None:
                invalid_lo = df[p] < lo
                if invalid_lo.any():
                    log["issues"].append(f"{int(invalid_lo.sum())} impossible {p} < {lo} values marked missing")
                    df.loc[invalid_lo, p] = np.nan
            if hi is not None:
                invalid_hi = df[p] > hi
                if invalid_hi.any():
                    log["issues"].append(f"{int(invalid_hi.sum())} impossible {p} > {hi} values marked missing")
                    df.loc[invalid_hi, p] = np.nan
            if p == "FC":
                extreme_fc = df[p] > 2_000_000
                if extreme_fc.any():
                    log["warnings"].append(
                        f"{int(extreme_fc.sum())} extreme FC readings (>2,000,000 CFU/100 mL) preserved without clipping"
                    )

            miss_col = df[p].isna()
            if miss_col.any():
                imputed_mask[p] = miss_col
                n_miss = int(miss_col.sum())
                log["imputed_cells"] += n_miss
                # Preserve NaN natively: IFS maps NaN -> (0, 0, 1); NSF-WQI renormalizes weights; CCME-WQI excludes NaN from total_tests
                log["issues"].append(
                    f"{n_miss} missing/NaN {p} cells preserved as NaN (IFS -> (0, 0, 1); excluded from NSF/CCME denominators)"
                )

        log.update({
            "final_rows": len(df),
            "date_min": str(df["Date"].min().date()),
            "date_max": str(df["Date"].max().date()),
            "season_dist": df["Season"].value_counts().to_dict(),
            "location_dist": df["Location"].value_counts().to_dict(),
        })

        BUS.publish(self.NAME, "Coordinator", "RESULT",
                    f"cleaned {len(df)} rows | imputed={log['imputed_cells']} | seasons={log['season_dist']}")
        return AgentResult("OK", data=(df, imputed_mask, log))


# ─────────────────────────────────────────────────────────────────────────────
# 11. WQI AGENT (NSF)
# ─────────────────────────────────────────────────────────────────────────────

class WQIAgent:
    NAME = "WQIAgent"

    def run(self, df: pd.DataFrame, config: AnalysisConfig,
            imputed_mask: Optional[pd.DataFrame] = None) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "compute NSF WQI")
        wqi_vals = NSFWQIEngine.compute_dataframe(df, config.weights, missing=imputed_mask)
        wqi_lbls, wqi_cols = nsf_class_vec(wqi_vals)
        out_df = pd.DataFrame({
            "Date": df["Date"].values if "Date" in df.columns else pd.NaT,
            "Location": df["Location"].values,
            "Season": df["Season"].values,
            "WQI": wqi_vals,
            "WQI_Class": wqi_lbls,
            "WQI_Color": wqi_cols,
        })
        summary = (out_df.groupby(["Location","Season"], sort=False)["WQI"]
                   .agg(["mean","min","max","std"]).reset_index()
                   .rename(columns={"mean":"WQI_Mean","min":"WQI_Min","max":"WQI_Max","std":"WQI_Std"}))
        summary["WQI_Mean"] = summary["WQI_Mean"].round(3)
        s_lbls, s_cols = nsf_class_vec(summary["WQI_Mean"].to_numpy())
        summary["WQI_Class"], summary["WQI_Color"] = s_lbls, s_cols
        avg = float(summary["WQI_Mean"].mean())
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"avg WQI={avg:.2f} | {nsf_class(avg)[0]}")
        return AgentResult("OK", data=(out_df, summary))


# ─────────────────────────────────────────────────────────────────────────────
# 12. CCME WQI AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CCMEAgent:
    NAME = "CCMEAgent"

    def run(self, df: pd.DataFrame,
            imputed_mask: Optional[pd.DataFrame] = None) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "compute CCME WQI")
        result = CCMEWQIEngine.compute(df, missing=imputed_mask)
        avg = result["CCME_WQI"].mean()
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"avg CCME={avg:.2f} | {ccme_class(avg)[0]}")
        return AgentResult("OK", data=result)


# ─────────────────────────────────────────────────────────────────────────────
# 13. IFS AGENT (Vectorized, Unified IMD Season Mapping, Non-Zero Hesitation)
# ─────────────────────────────────────────────────────────────────────────────
class IFSAgent:
    NAME = "IFSAgent"

    def run(self, df: pd.DataFrame, config: AnalysisConfig,
            imputed_mask: Optional[pd.DataFrame] = None) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "vectorized IFS fuzzification")

        if "Season" in df.columns and not df["Season"].isna().any():
            work_df = df
        else:
            work_df = df.copy()
            dt = pd.to_datetime(work_df["Date"], errors="coerce")
            # Single authoritative IMD season mapping (June = Monsoon, vectorized)
            work_df["Season"] = get_season_vec(dt.dt.month.fillna(1).to_numpy())

        ifs_df = IFSEngine.fuzzify_dataframe(work_df, config.params, imputed_mask=imputed_mask, pi_nil=config.pi_nil, pi_missing=config.pi_missing)
        trip_cols = [c for c in ifs_df.columns if c.startswith(("mu_", "nu_", "pi_"))]

        rel_r = ifs_df.groupby("Season", as_index=False, sort=False)[trip_cols].mean()
        rel_x = ifs_df.groupby("Location", as_index=False, sort=False)[trip_cols].mean()

        # Order seasons & locations consistently
        rel_r["Season"] = pd.Categorical(rel_r["Season"], categories=SEASON_ORDER, ordered=True)
        rel_r = rel_r.sort_values("Season").reset_index(drop=True)
        rel_r["Season"] = rel_r["Season"].astype(str)

        loc_order = resolve_locations(work_df)
        rel_x["Location"] = pd.Categorical(rel_x["Location"], categories=loc_order, ordered=True)
        rel_x = rel_x.sort_values("Location").reset_index(drop=True)
        rel_x["Location"] = rel_x["Location"].astype(str)

        for p in config.params:
            rel_r[f"E_{p}"] = (rel_r[f"mu_{p}"] - rel_r[f"nu_{p}"] * rel_r[f"pi_{p}"]).round(4)
            rel_x[f"E_{p}"] = (rel_x[f"mu_{p}"] - rel_x[f"nu_{p}"] * rel_x[f"pi_{p}"]).round(4)
            rel_r[p] = rel_r.apply(lambda r: fmt_ifs(r[f"mu_{p}"], r[f"nu_{p}"], r[f"pi_{p}"]), axis=1)
            rel_x[p] = rel_x.apply(lambda r: fmt_ifs(r[f"mu_{p}"], r[f"nu_{p}"], r[f"pi_{p}"]), axis=1)

        mean_pi = float(ifs_df[[f"pi_{p}" for p in config.params]].mean().mean())
        BUS.publish(self.NAME, "Coordinator", "RESULT",
                    f"Vectorized Relation R/X computed | mean π={mean_pi:.4f}")
        return AgentResult("OK", data=(rel_r, rel_x, ifs_df))


# ─────────────────────────────────────────────────────────────────────────────
# 14. INFERENCE AGENT (Direct Joint Spatiotemporal T*(s, l) Aggregation)
# ─────────────────────────────────────────────────────────────────────────────
class InferenceAgent:
    NAME = "InferenceAgent"

    def run(self, ifs_df: pd.DataFrame, config: AnalysisConfig) -> AgentResult:
        """
        Computes joint spatiotemporal Relation T*(s, l) directly from record-level
        (Season, Location) IFS triplets using normalized parameter weights.
        Preserves true season × station interactions (prevents marginal R∘X clipping).
        """
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "joint spatiotemporal inference T*(s,l)")

        # Normalize weights from config so weight sensitivity tests work seamlessly
        raw_w = {p: float(config.weights.get(p, WEIGHTS[p])) for p in config.params}
        w_sum = sum(raw_w.values())
        w = {p: raw_w[p] / w_sum for p in config.params}

        trip_cols = [c for c in ifs_df.columns if c.startswith(("mu_", "nu_", "pi_"))]
        joint = ifs_df.groupby(["Season", "Location"], as_index=False)[trip_cols].mean()

        t_rows = []
        seasons = [s for s in SEASON_ORDER if s in joint["Season"].values]
        locations = resolve_locations(ifs_df)

        for s in seasons:
            for l in locations:
                sub = joint[(joint["Season"] == s) & (joint["Location"] == l)]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                mu_t = sum(w[p] * float(row[f"mu_{p}"]) for p in config.params)
                nu_t = sum(w[p] * float(row[f"nu_{p}"]) for p in config.params)
                pi_t = max(0.0, 1.0 - mu_t - nu_t)
                eff_e = mu_t - (nu_t * pi_t)

                t_rows.append({
                    "Season":   s,
                    "Location": l,
                    "Mu_T":     round(mu_t, 4),
                    "Nu_T":     round(nu_t, 4),
                    "Pi_T":     _clean_zero(max(0.0, pi_t), 4),
                    "E_T":      round(eff_e, 4),
                    "IFS":      fmt_ifs(mu_t, nu_t, pi_t),
                })

        result = pd.DataFrame(t_rows)
        best = result.groupby("Location")["E_T"].mean().idxmax() if not result.empty else "N/A"
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"Joint T*(s,l) computed | cleanest={best}")
        return AgentResult("OK", data=result)


# ─────────────────────────────────────────────────────────────────────────────
# 15. DISTANCE AGENT (3-Term Szmidt-Kacprzyk Normalized Hamming & Euclidean)
# ─────────────────────────────────────────────────────────────────────────────
class DistanceAgent:
    NAME = "DistanceAgent"

    def run(self, rel_t: pd.DataFrame, config: AnalysisConfig) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST",
                    f"compute 3-term Szmidt-Kacprzyk distances from {config.reference_location or 'auto-ref'}")

        dist_rows = []
        seasons = [s for s in SEASON_ORDER if s in rel_t["Season"].values]
        locations = resolve_locations(rel_t)
        ref_loc = config.reference_location or ("Palla" if "Palla" in locations else rel_t.groupby("Location")["E_T"].mean().idxmax())
        ref_data = rel_t[rel_t["Location"] == ref_loc].set_index("Season")

        if not ref_data.empty:
            for l in locations:
                hamming_sum = 0.0
                euclidean_sum = 0.0
                n_matched = 0

                for s in seasons:
                    if s not in ref_data.index:
                        continue
                    curr_sub = rel_t[(rel_t["Location"] == l) & (rel_t["Season"] == s)]
                    if curr_sub.empty:
                        continue
                    p_vals = ref_data.loc[s]
                    curr_vals = curr_sub.iloc[0]

                    d_mu = abs(float(p_vals["Mu_T"]) - float(curr_vals["Mu_T"]))
                    d_nu = abs(float(p_vals["Nu_T"]) - float(curr_vals["Nu_T"]))
                    d_pi = abs(float(p_vals["Pi_T"]) - float(curr_vals["Pi_T"]))

                    hamming_sum += (d_mu + d_nu + d_pi)
                    euclidean_sum += (d_mu**2 + d_nu**2 + d_pi**2)
                    n_matched += 1

                h_dist = hamming_sum / (2.0 * n_matched) if n_matched > 0 else 0.0
                e_dist = math.sqrt(euclidean_sum / (2.0 * n_matched)) if n_matched > 0 else 0.0
                mean_e = float(rel_t[rel_t["Location"] == l]["E_T"].mean())

                dist_rows.append({
                    "Location":        l,
                    "Matched_Seasons": n_matched,
                    "Mean_E_T":        _clean_zero(mean_e, 4),
                    "Hamming":         _clean_zero(h_dist, 4),
                    "Euclidean":       _clean_zero(e_dist, 4),
                })

        df = pd.DataFrame(dist_rows).sort_values("Hamming").reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"

        worst = df.sort_values("Hamming", ascending=False).iloc[0]["Location"] if not df.empty else "N/A"
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"most polluted: {worst}")
        return AgentResult("OK", data=df)

    @staticmethod
    def verification_table(rel_t: pd.DataFrame, reference_location: str = "Palla") -> pd.DataFrame:
        """Returns season-by-season |Δμ|, |Δν|, |Δπ| terms so Table 6 -> distance math is 100% reproducible."""
        ref = rel_t[rel_t["Location"] == reference_location].set_index("Season")
        rows = []
        for _, r in rel_t.iterrows():
            s = r["Season"]
            if s not in ref.index:
                continue
            p = ref.loc[s]
            d_mu = abs(float(p["Mu_T"]) - float(r["Mu_T"]))
            d_nu = abs(float(p["Nu_T"]) - float(r["Nu_T"]))
            d_pi = abs(float(p["Pi_T"]) - float(r["Pi_T"]))
            rows.append({
                "Location": r["Location"], "Season": s,
                "Ref_IFS": p["IFS"], "Loc_IFS": r["IFS"],
                "|Δμ|": round(d_mu, 4), "|Δν|": round(d_nu, 4), "|Δπ|": round(d_pi, 4),
                "Season_dH": round((d_mu + d_nu + d_pi) / 2.0, 4),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 16. LLM DECISION AGENT  (AI-powered + rule-based fallback + memory context)
# ─────────────────────────────────────────────────────────────────────────────

RULE_DECISIONS = [
    {"id":"WQI_CRIT","level":"CRITICAL","check":lambda r:r.get("wqi",100)<26,
     "param":"WQI","msg":"WQI critically low — water unsafe for any use",
     "action":"Emergency treatment and restrict public access.","show_viewer":True},
    {"id":"FC_CRIT","level":"CRITICAL","check":lambda r:r.get("fc_mu",1)<0.05,
     "param":"FC","msg":"Fecal Coliform near-zero — severe sewage contamination",
     "action":"Block upstream sewage discharge points urgently.","show_viewer":True},
    {"id":"DO_CRIT","level":"CRITICAL","check":lambda r:r.get("do_mu",1)<0.10,
     "param":"DO","msg":"Dissolved Oxygen critically depleted — aquatic life at risk",
     "action":"Emergency aeration. Alert fisheries department.","show_viewer":True},
    {"id":"CCME_POOR","level":"CRITICAL","check":lambda r:r.get("ccme",100)<45,
     "param":"CCME","msg":"CCME WQI Poor — Canadian guideline threshold breached",
     "action":"Comprehensive remediation plan required within 30 days.","show_viewer":True},
    {"id":"WQI_WARN","level":"WARNING","check":lambda r:26<=r.get("wqi",100)<51,
     "param":"WQI","msg":"Significant pollution detected","action":"Increase monitoring frequency.","show_viewer":True},
    {"id":"HAMMING_HIGH","level":"WARNING","check":lambda r:r.get("hamming",0)>0.20,
     "param":"Distance","msg":"High deviation from clean baseline (Palla)",
     "action":"Prioritise location for remediation.","show_viewer":True},
    {"id":"BOD_WARN","level":"WARNING","check":lambda r:r.get("bod_mu",1)<0.25,
     "param":"BOD","msg":"High organic load (BOD elevated)",
     "action":"Check industrial effluent discharge.","show_viewer":False},
    {"id":"PI_HIGH","level":"INFO","check":lambda r:r.get("pi",0)>0.15,
     "param":"Uncertainty","msg":"High IFS hesitation — data uncertain",
     "action":"Increase sampling frequency.","show_viewer":False},
]

LLM_DECISION_PROMPT = """
You are an expert water quality scientist analysing river data from the Yamuna River, Delhi.

CURRENT ANALYSIS DATA:
{data_summary}

HISTORICAL CONTEXT:
{history_context}

Based on this data, generate a JSON array of decision alerts. For each alert include:
{{
  "level": "CRITICAL" | "WARNING" | "INFO",
  "location": "<location name>",
  "parameter": "<parameter or WQI or CCME>",
  "ai_explanation": "<2-3 sentence scientific explanation of why this is concerning>",
  "likely_cause": "<probable environmental or anthropogenic cause>",
  "recommended_action": "<specific, actionable recommendation>",
  "urgency_score": <1-10>,
  "confidence": <0.0-1.0>
}}

Generate ONLY for locations/parameters with genuine concern. Return ONLY valid JSON array, no other text.
"""

class LLMDecisionAgent:
    NAME = "LLMDecisionAgent"

    def __init__(self):
        self._groq   = None
        self._gemini = None

    def init_llm(self, groq_key: str = "", gemini_key: str = ""):
        if GROQ_OK and groq_key:
            self._groq = Groq(api_key=groq_key)
            LOGGER.info(self.NAME, "Groq LLM initialised for decision generation")
        if GEMINI_OK and gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini = genai.GenerativeModel(POLICY_CONFIG["gemini_model"])
            LOGGER.info(self.NAME, "Gemini LLM initialised as fallback")

    def run(self, state: PipelineState, memory: AgentMemory) -> AgentResult:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "generate AI-enhanced decisions")

        # Always run rule-based first (guaranteed results)
        rule_decisions = self._rule_based(state)
        BUS.publish(self.NAME, "Coordinator", "RESULT",
                    f"Rule-based: {len(rule_decisions)} alerts")

        # Try LLM augmentation
        if self._groq or self._gemini:
            BUS.publish(self.NAME, "LLMAssistantAgent", "MSG",
                        "requesting AI decision augmentation")
            try:
                ai_decisions = self._llm_decisions(state, memory)
                # Merge: AI decisions add ai_explanation/cause fields to matching rule decisions
                merged = self._merge(rule_decisions, ai_decisions)
                BUS.publish(self.NAME, "Coordinator", "RESULT",
                            f"AI-augmented: {len(merged)} decisions | {len(ai_decisions)} AI insights")
                return AgentResult("OK", data=merged)
            except Exception as e:
                LOGGER.log(self.NAME, "Coordinator", "WARN", f"LLM decision failed: {e}")

        return AgentResult("OK", data=rule_decisions)

    def _rule_based(self, state: PipelineState) -> List[Dict]:
        decisions = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        if state.wqi_summary is None or state.ifs_t is None or state.dist_df is None:
            return decisions
        # Group by Location (deduplicated per station) and read TRUE per-parameter μ from ifs_x
        wqi_loc = state.wqi_summary.groupby("Location", as_index=False)["WQI_Mean"].mean()
        for _, row in wqi_loc.iterrows():
            loc = row["Location"]
            t_q = state.ifs_t[state.ifs_t["Location"] == loc]
            x_q = state.ifs_x[state.ifs_x["Location"] == loc] if state.ifs_x is not None else pd.DataFrame()
            dr  = state.dist_df.reset_index()
            dr  = dr[dr["Location"] == loc]
            ccme_q = state.ccme_df[state.ccme_df["Location"] == loc] if state.ccme_df is not None else pd.DataFrame()
            ctx = {
                "wqi":     float(row.get("WQI_Mean", 100)),
                "do_mu":   float(x_q["mu_DO"].iloc[0])  if (not x_q.empty and "mu_DO"  in x_q.columns) else 1.0,
                "fc_mu":   float(x_q["mu_FC"].iloc[0])  if (not x_q.empty and "mu_FC"  in x_q.columns) else 1.0,
                "bod_mu":  float(x_q["mu_BOD"].iloc[0]) if (not x_q.empty and "mu_BOD" in x_q.columns) else 1.0,
                "pi":      float(t_q["Pi_T"].mean())    if not t_q.empty else 0.0,
                "hamming": float(dr["Hamming"].iloc[0]) if not dr.empty else 0.0,
                "ccme":    float(ccme_q["CCME_WQI"].mean()) if not ccme_q.empty else 100.0,
            }
            for rule in RULE_DECISIONS:
                try:
                    if rule["check"](ctx):
                        decisions.append({
                            "rule_id":      rule["id"],
                            "level":        rule["level"],
                            "param":        rule["param"],
                            "location":     loc,
                            "wqi":          round(ctx["wqi"],2),
                            "ccme_wqi":     round(ctx["ccme"],2),
                            "hamming":      round(ctx["hamming"],4),
                            "message":      rule["msg"],
                            "action":       rule["action"],
                            "show_viewer":  rule["show_viewer"],
                            "ai_explanation": "",
                            "likely_cause":   "",
                            "urgency_score":  POLICY_CONFIG["urgency_map"].get(rule["level"],3),
                            "confidence":     0.9,
                            "timestamp":      ts,
                            "source":         "rule-based",
                        })
                except Exception:
                    pass
        return decisions

    def _llm_decisions(self, state: PipelineState, memory: AgentMemory) -> List[Dict]:
        data_summary = self._build_data_summary(state)
        history_ctx  = memory.get_llm_context()
        prompt = LLM_DECISION_PROMPT.format(
            data_summary=data_summary,
            history_context=history_ctx,
        )
        resp = ""
        if self._groq:
            r = self._groq.chat.completions.create(
                model=POLICY_CONFIG["groq_model"],
                messages=[{"role":"user","content":prompt}],
                max_tokens=1500, temperature=0.2,
            )
            resp = r.choices[0].message.content
        elif self._gemini:
            resp = self._gemini.generate_content(prompt).text

        # Strip markdown fences
        resp = resp.strip()
        if resp.startswith("```"):
            resp = resp.split("```")[1]
            if resp.startswith("json"): resp = resp[4:]
        resp = resp.strip()

        return json.loads(resp)

    @staticmethod
    def _build_data_summary(state: PipelineState) -> str:
        parts = []
        if state.wqi_summary is not None:
            parts.append("NSF WQI SUMMARY:\n" +
                         state.wqi_summary[["Location","Season","WQI_Mean","WQI_Class"]].to_string(index=False))
        if state.ccme_df is not None:
            parts.append("CCME WQI SUMMARY:\n" +
                         state.ccme_df[["Location","Season","CCME_WQI","CCME_Class","F1_Scope","F2_Frequency"]].to_string(index=False))
        if state.ifs_t is not None:
            parts.append("IFS COMPOSITION (Relation T):\n" +
                         state.ifs_t[["Season","Location","Mu_T","Nu_T","Pi_T"]].to_string(index=False))
        if state.dist_df is not None:
            parts.append("DISTANCE FROM PALLA:\n" +
                         state.dist_df.reset_index().to_string(index=False))
        return "\n\n".join(parts)

    @staticmethod
    def _merge(rule_decisions: List[Dict], ai_decisions: List[Dict]) -> List[Dict]:
        """Attach AI explanations to matching rule decisions; add purely AI ones."""
        merged = list(rule_decisions)
        for ai in ai_decisions:
            loc   = ai.get("location","")
            param = ai.get("parameter","")
            matched = False
            for d in merged:
                if d["location"] == loc and d["param"] == param:
                    d["ai_explanation"] = ai.get("ai_explanation","")
                    d["likely_cause"]   = ai.get("likely_cause","")
                    d["urgency_score"]  = ai.get("urgency_score", d.get("urgency_score",5))
                    d["confidence"]     = ai.get("confidence", d.get("confidence",0.9))
                    d["source"]         = "ai-enhanced"
                    matched = True
                    break
            if not matched:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                merged.append({
                    "rule_id": f"AI_{loc[:3].upper()}_{param[:3].upper()}",
                    "level":   ai.get("level","INFO"),
                    "param":   param, "location": loc,
                    "wqi": 0, "ccme_wqi": 0, "hamming": 0,
                    "message":      ai.get("ai_explanation",""),
                    "action":       ai.get("recommended_action",""),
                    "show_viewer":  ai.get("level","INFO") != "INFO",
                    "ai_explanation": ai.get("ai_explanation",""),
                    "likely_cause":   ai.get("likely_cause",""),
                    "urgency_score":  ai.get("urgency_score",5),
                    "confidence":     ai.get("confidence",0.7),
                    "timestamp":      ts,
                    "source":         "ai-only",
                })
        merged.sort(key=lambda d: {"CRITICAL":0,"WARNING":1,"INFO":2}.get(d["level"],9))
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# 17. ALERT NOTIFICATION AGENT (email ALL registered officials on CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

class AlertNotificationAgent:
    NAME = "AlertNotificationAgent"

    def __init__(self, smtp_config: Optional[Dict] = None):
        self.smtp = smtp_config

    def filter_by_role(self, decisions: List[Dict], role: str) -> List[Dict]:
        if role == "official": return decisions
        return [d for d in decisions if d.get("show_viewer", False)]

    def to_dataframe(self, decisions: List[Dict], role: str) -> pd.DataFrame:
        visible = self.filter_by_role(decisions, role)
        if not visible:
            return pd.DataFrame(columns=["Level","Location","Parameter","WQI","CCME_WQI",
                                         "Message","Action","AI Explanation","Urgency","Time"])
        rows = []
        for d in visible:
            rows.append({
                "Level":          d["level"],
                "Location":       d["location"],
                "Parameter":      d["param"],
                "WQI":            d.get("wqi","-"),
                "CCME_WQI":       d.get("ccme_wqi","-"),
                "Message":        d["message"],
                "Action":         d["action"],
                "AI Explanation": d.get("ai_explanation","") or "",
                "Likely Cause":   d.get("likely_cause","") or "",
                "Urgency":        d.get("urgency_score","-"),
                "Source":         "" if "rule" in d.get("source", "rule-based") else d.get("source", "ai-enhanced"),
                "Time":           d["timestamp"],
            })
        df = pd.DataFrame(rows)
        df["_sort"] = df["Level"].map({"CRITICAL":0,"WARNING":1,"INFO":2})
        return df.sort_values("_sort").drop("_sort",axis=1).reset_index(drop=True)

    def alert_html(self, decisions: List[Dict], role: str) -> str:
        visible = self.filter_by_role(decisions, role)
        if not visible:
            return '<div style="color:#64748B;padding:12px;font-size:13px">No alerts for current analysis.</div>'
        rows = []
        for d in sorted(visible, key=lambda x:{"CRITICAL":0,"WARNING":1,"INFO":2}.get(x["level"],9)):
            col  = ALERT_COLORS.get(d["level"],"#64748B")
            source_val = d.get("source", "rule-based")
            src_badge = "" if "rule" in source_val else f'<span style="font-size:10px;background:#E0F2FE;color:#0369A1;padding:2px 6px;border-radius:3px;margin-left:6px">{source_val}</span>'
            ai_block = ""
            if d.get("ai_explanation"):
                ai_block = f'<div style="font-size:11px;color:#6B21A8;margin-top:3px;font-style:italic">🤖 {d["ai_explanation"]}</div>'
            cause_block = ""
            if d.get("likely_cause"):
                cause_block = f'<div style="font-size:11px;color:#92400E;margin-top:2px">🔍 Likely cause: {d["likely_cause"]}</div>'
            rows.append(f"""
<div style="display:flex;gap:10px;padding:12px;margin:6px 0;background:#BAE6FD;
            border-left:4px solid {col};border-radius:0 8px 8px 0;
            box-shadow:0 1px 3px rgba(0,0,0,0.06)">
  <div style="min-width:80px">
    <span style="background:{col}20;color:{col};font-size:11px;font-weight:700;
                 padding:3px 8px;border-radius:4px;display:block;text-align:center">{d['level']}</span>
    <div style="font-size:10px;color:#94A3B8;margin-top:4px;text-align:center">
      Urgency: {d.get('urgency_score','?')}/10</div>
  </div>
  <div style="flex:1">
    <div style="font-size:13px;font-weight:600;color:white">{d['location']}
      <span style="color:#94A3B8;font-weight:400"> — {d['param']}</span>{src_badge}</div>
    <div style="font-size:12px;color:#475569;margin-top:2px">{d['message']}</div>
    <div style="font-size:11px;color:#0369A1;margin-top:3px;font-weight:500">→ {d['action']}</div>
    {ai_block}{cause_block}
    <div style="font-size:10px;color:#94A3B8;margin-top:3px">
      NSF WQI: {d.get('wqi','-')} | CCME WQI: {d.get('ccme_wqi','-')} | {d['timestamp']}</div>
  </div>
</div>""")
        return "\n".join(rows)

    def send_to_all_officials(self, decisions: List[Dict],
                              official_emails: List[str]) -> Dict[str, bool]:
        """Send critical alerts to ALL registered officials."""
        critical = [d for d in decisions if d.get("level") == "CRITICAL"]
        if not critical or not self.smtp or not official_emails:
            BUS.publish(self.NAME, "Coordinator", "INFO",
                        f"Email skipped: critical={len(critical)}, smtp={bool(self.smtp)}, "
                        f"recipients={len(official_emails)}")
            return {}
        results = {}
        for email in official_emails:
            ok, msg = self._send_one(critical, email)
            results[email] = ok
            BUS.publish(self.NAME, "AlertSystem", "ALERT",
                        f"Email to {email}: {'sent' if ok else 'failed'} — {msg}")
        return results

    def _send_one(self, critical: List[Dict], recipient: str) -> Tuple[bool, str]:
        body_lines = [
            f"YAMUNA WATER QUALITY — CRITICAL ALERT\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*50}\n"
        ]
        for d in critical:
            body_lines.append(
                f"[{d['level']}] {d['location']} — {d['param']}\n"
                f"  Message: {d['message']}\n"
                f"  Action Required: {d['action']}\n"
                + (f"  AI Analysis: {d['ai_explanation']}\n" if d.get("ai_explanation") else "")
                + (f"  Likely Cause: {d['likely_cause']}\n"  if d.get("likely_cause") else "")
                + f"  NSF WQI: {d.get('wqi','-')} | CCME: {d.get('ccme_wqi','-')}\n"
            )
        body = "\n".join(body_lines)
        msg = MIMEMultipart()
        msg["Subject"] = f"[CRITICAL] Yamuna Water Quality Alert — {len(critical)} critical findings"
        msg["From"]    = self.smtp["user"]
        msg["To"]      = recipient
        msg.attach(MIMEText(body, "plain"))
        try:
            with smtplib.SMTP_SSL(self.smtp["host"], int(self.smtp.get("port",465))) as s:
                s.login(self.smtp["user"], self.smtp["password"])
                s.send_message(msg)
            return True, "sent"
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 18. LLM ASSISTANT AGENT  (context-aware Q&A)
# ─────────────────────────────────────────────────────────────────────────────

SYS_TEMPLATE = """
You are an expert water quality scientist and data analyst specialising in Intuitionistic Fuzzy Sets
(IFS) and river water quality assessment for the Yamuna River, Delhi (Palla → Okhla Barrage).

FRAMEWORK:
- IFS: μ = membership (good quality), ν = non-membership (pollution), π = hesitation (uncertainty)
- E_T = Effective Membership = μ_T - (ν_T × π_T) (NEVER call E_T "Entropy")
- NSF WQI: weighted sum of sub-indices (DO=0.309, FC=0.291, pH=0.200, BOD=0.200)
- CCME WQI: Canadian guidelines — Excellent(95-100), Good(80-94), Fair(65-79), Marginal(45-64), Poor(<45)
- Reference location (cleanest): Palla

CURRENT ANALYSIS RESULTS:
{context}

INSTRUCTIONS: Answer concisely (≤250 words unless detail requested). Use specific numbers from results.
For theory: explain IFS/WQI/CCME clearly. For recommendations: give location-specific actions.
"""

class LLMAssistantAgent:
    NAME = "LLMAssistantAgent"

    def __init__(self):
        self._groq   = None
        self._gemini = None
        self._system = None
        self.groq_key   = ""
        self.gemini_key = ""
        self._disabled_models = set()

    def init(self, groq_key: str = "", gemini_key: str = ""):
        self.groq_key   = groq_key
        self.gemini_key = gemini_key
        if GROQ_OK and groq_key:
            self._groq = Groq(api_key=groq_key)
        if GEMINI_OK and gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini = genai.GenerativeModel(POLICY_CONFIG["gemini_model"])

    def set_context(self, state: PipelineState, memory: AgentMemory):
        BUS.publish(self.NAME, "LLMDecisionAgent", "MSG", "updating analysis context")
        parts = [f"Run: {state.run_ts}"]
        if state.dist_df    is not None:
            parts.append("DISTANCE RANKINGS:\n" + state.dist_df.reset_index().to_string(index=False))
        if state.ifs_t      is not None:
            parts.append("RELATION T* (JOINT SEASON×LOCATION):\n" +
                         state.ifs_t[["Season","Location","Mu_T","Nu_T","Pi_T","E_T"]].to_string(index=False))
        if state.ifs_r is not None:
            parts.append("RELATION R (SEASONS × PARAMETERS):\n" +
                         state.ifs_r[["Season"] + [f"mu_{p}" for p in PARAMS]].round(4).to_string(index=False))
        if state.ifs_x is not None:
            parts.append("RELATION X (LOCATIONS × PARAMETERS):\n" +
                         state.ifs_x[["Location"] + [f"mu_{p}" for p in PARAMS] + ["pi_DO"]].round(4).to_string(index=False))
            p_avgs = {p: round(float(state.ifs_x[f"mu_{p}"].mean()), 4) for p in PARAMS}
            parts.append("PARAMETER MEAN MEMBERSHIP ACROSS ALL SITES (Relation X): " + ", ".join(f"{k}={v}" for k, v in p_avgs.items()))
        if state.wqi_df is not None:
            rec_wqi = state.wqi_df.groupby("Location", as_index=False)["WQI"].mean().rename(columns={"WQI": "NSF_WQI_RecordMean"})
            parts.append("NSF WQI (RECORD-LEVEL STATION MEANS):\n" + rec_wqi.round(2).to_string(index=False))
        if state.wqi_summary is not None:
            parts.append("NSF WQI (SEASONAL):\n" +
                         state.wqi_summary[["Location","Season","WQI_Mean","WQI_Class"]].to_string(index=False))
        if state.ccme_df is not None:
            ccme_loc = state.ccme_df.groupby("Location", as_index=False)["CCME_WQI"].mean().rename(columns={"CCME_WQI": "CCME_WQI_StationMean"})
            parts.append("CCME WQI (STATION MEANS):\n" + ccme_loc.round(2).to_string(index=False))
            parts.append("CCME WQI (SEASONAL):\n" +
                         state.ccme_df[["Location","Season","CCME_WQI","CCME_Class"]].to_string(index=False))
        if state.decisions:
            parts.append("ACTIVE ALERTS:\n" +
                         "\n".join(f"[{d['level']}] {d['location']}: {d['message']}" for d in state.decisions[:12]))
        parts.append(memory.get_llm_context())
        self._system = SYS_TEMPLATE.format(context="\n\n".join(parts))

    def chat(self, message: str, history: list) -> Tuple[list, str]:
        if not self._system:
            reply = "No analysis has been run yet. Run the pipeline first, then ask me anything."
        elif self._groq:
            reply = self._groq_chat(message, history)
        elif self._gemini:
            reply = self._gemini_chat(message)
        else:
            reply = ("LLM not configured. Add your Groq or Gemini API key in Settings "
                     "to enable AI-powered answers.")
        BUS.publish(self.NAME, "Coordinator", "LLM", f"Q: {message[:60]}... A: {reply[:60]}...")
        new_history = list(history) + [
            {"role":"user","content":message},
            {"role":"assistant","content":reply},
        ]
        return new_history, ""

    def _groq_chat(self, message: str, history: list) -> str:
        msgs = [{"role":"system","content":self._system}]
        for h in history:
            if isinstance(h, dict):
                msgs.append(h)
            elif isinstance(h, (list, tuple)) and len(h)==2:
                msgs.append({"role":"user","content":str(h[0])})
                msgs.append({"role":"assistant","content":str(h[1])})
        msgs.append({"role":"user","content":message})
        # Auto-discover live Groq chat models on your account (separate rate-limit buckets, zero hardcoding)
        if not hasattr(self, "_cached_groq_models") or not self._cached_groq_models:
            live_g = []
            try:
                for m in self._groq.models.list().data:
                    mid = getattr(m, "id", "")
                    low = mid.lower()
                    if any(skip in low for skip in ("whisper", "guard", "tts", "playai", "embed", "vision", "audio", "safeguard")):
                        continue
                    live_g.append(mid)
            except Exception:
                pass
            preferred = [POLICY_CONFIG["groq_model"], "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
            self._cached_groq_models = list(dict.fromkeys([m for m in preferred if not live_g or m in live_g] + live_g))
            LOGGER.info(self.NAME, f"Auto-discovered live Groq models: {self._cached_groq_models[:6]}")
        candidate_models = self._cached_groq_models
        last_err = None
        for m_name in candidate_models:
            if m_name in self._disabled_models:
                continue
            for attempt in range(2):
                try:
                    r = self._groq.chat.completions.create(
                        model=m_name, messages=msgs,
                        max_tokens=1024, temperature=0.0, timeout=30)
                    txt = (r.choices[0].message.content or "").strip()
                    if not txt:
                        msgs_retry = msgs + [{"role": "user", "content": "Give the direct final answer in 1 concise sentence."}]
                        r2 = self._groq.chat.completions.create(
                            model=m_name, messages=msgs_retry,
                            max_tokens=1024, temperature=0.0, timeout=30)
                        txt = (r2.choices[0].message.content or "").strip()
                    if txt:
                        return txt
                except Exception as e:
                    last_err = e
                    err_s = str(e).lower()
                    if "429" in err_s or "rate limit" in err_s:
                        if "day" in err_s or "tpd" in err_s or attempt == 1:
                            self._disabled_models.add(m_name)
                            break
                        time.sleep(8)
                        continue
                    self._disabled_models.add(m_name)
                    break
        if self._gemini or self.gemini_key:
            return self._gemini_chat(message)
        return f"Groq error: {last_err}"

    def _discover_gemini_models(self) -> List[str]:
        if hasattr(self, "_cached_gemini_models") and self._cached_gemini_models:
            return self._cached_gemini_models
        found = []
        try:
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                name = getattr(m, "name", "")
                low = name.lower()
                if "generateContent" in methods and "gemini" in low:
                    if any(skip in low for skip in ("tts", "image", "audio", "vision", "embedding", "aqa", "robotics", "computer-use")):
                        continue
                    found.append(name.replace("models/", ""))
        except Exception:
            pass
        # Sort: prefer flash models (newest first), then pro models
        found.sort(key=lambda x: (0 if "flash" in x else 1, 1 if "lite" in x else 0, x), reverse=False)
        if not found:
            found = [POLICY_CONFIG["gemini_model"], "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self._cached_gemini_models = list(dict.fromkeys(found))
        LOGGER.info(self.NAME, f"Auto-discovered live Gemini models: {self._cached_gemini_models[:6]}")
        return self._cached_gemini_models

    def _gemini_chat(self, message: str) -> str:
        if not GEMINI_OK or not self.gemini_key:
            return "Gemini error: not configured"
        gemini_candidates = self._discover_gemini_models()
        last_err = "All discovered Gemini models unavailable or rate-limited"
        for gm in gemini_candidates:
            if gm in self._disabled_models:
                continue
            for attempt in range(2):
                try:
                    model = genai.GenerativeModel(gm)
                    resp = model.generate_content(
                        f"{self._system}\n\nUser: {message}\nAnswer directly in 1-2 concise sentences.",
                        generation_config={"temperature": 0.0},
                        request_options={"timeout": 30},
                    )
                    txt = (getattr(resp, "text", "") or "").strip()
                    if txt:
                        return txt
                except Exception as e:
                    last_err = e
                    err_s = str(e).lower()
                    if "404" in err_s or "not found" in err_s or "deprecated" in err_s or "not supported" in err_s:
                        self._disabled_models.add(gm)
                        break
                    if "429" in err_s or "quota" in err_s or "resource" in err_s:
                        if attempt == 0:
                            time.sleep(10)
                            continue
                        self._disabled_models.add(gm)
                        break
                    break
        return f"Gemini error: {last_err}"


# ─────────────────────────────────────────────────────────────────────────────
# 19. VISUALIZATION AGENT  (Plotly charts)
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationAgent:
    NAME = "VisualizationAgent"

    LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif",size=12,color="white"),
        margin=dict(l=20,r=20,t=48,b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)",borderwidth=0),
    )

    def _lay(self, fig, title=""):
        fig.update_layout(**self.LAYOUT,
            title=dict(text=title,font=dict(size=13,color="white")))
        fig.update_xaxes(showgrid=True,gridcolor="#E2E8F0",gridwidth=0.5,zeroline=False)
        fig.update_yaxes(showgrid=True,gridcolor="#E2E8F0",gridwidth=0.5,zeroline=False)
        return fig

    def _hmap(self, z, x, y, title):
        fig = go.Figure(go.Heatmap(
            z=z, x=x, y=y,
            colorscale=[[0,"#DC2626"],[0.35,"#FBBF24"],[0.65,"#86EFAC"],[1,"#16A34A"]],
            zmin=0, zmax=1, text=[[f"{v:.3f}" for v in r] for r in z],
            texttemplate="%{text}",
            hovertemplate="%{y} — %{x}: %{z:.3f}<extra></extra>"))
        return self._lay(fig, title)

    def heatmap_R(self, rel_r):
        d = rel_r.set_index("Season")[[f"mu_{p}" for p in PARAMS]]
        d.columns = PARAMS
        rows = [s for s in SEASON_ORDER if s in d.index]
        d = d.reindex(rows)
        return self._hmap(d.values, PARAMS, rows, "Relation R — Season × Parameter (μ)")

    def heatmap_X(self, rel_x):
        d = rel_x.set_index("Location")[[f"mu_{p}" for p in PARAMS]]
        d.columns = PARAMS
        locs = [l for l in LOCATION_ORDER if l in d.index] + [l for l in d.index if l not in LOCATION_ORDER]
        d = d.reindex(locs)
        return self._hmap(d.values, PARAMS, locs, "Relation X — Location × Parameter (μ)")

    def heatmap_T(self, ifs_t):
        piv = ifs_t.pivot(index="Season",columns="Location",values="Mu_T")
        rows = [s for s in SEASON_ORDER   if s in piv.index]
        cols = [l for l in LOCATION_ORDER if l in piv.columns] + [l for l in piv.columns if l not in LOCATION_ORDER]
        piv  = piv.reindex(index=rows, columns=cols)
        return self._hmap(piv.values.tolist(), cols, rows, "Relation T — Season × Location (μ_T)")

    def lollipop(self, dist_df):
        df = dist_df.reset_index(drop=True)
        fig = go.Figure()
        for i, row in df.iterrows():
            c = LOC_COLORS.get(row["Location"],"#64748B")
            for val, off, sym, dsh in [(row["Hamming"],0,"circle","solid"),
                                        (row["Euclidean"],0.28,"diamond","dot")]:
                fig.add_trace(go.Scatter(x=[0,val],y=[i+off,i+off],mode="lines",
                    line=dict(color=c,width=3,dash=dsh),showlegend=False,hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[val],y=[i+off],mode="markers",
                    marker=dict(size=12,color=c,symbol=sym),
                    name=f"{row['Location']} ({'H' if off==0 else 'E'})",
                    hovertemplate=f"{row['Location']}<br>{'Hamming' if off==0 else 'Euclidean'}: {val:.4f}<extra></extra>"))
        fig.update_layout(**self.LAYOUT,
            title=dict(text="Distance from Palla — Hamming (●) & Euclidean (◆)",font=dict(size=13)),
            xaxis_title="Distance",
            yaxis=dict(tickmode="array",tickvals=[i+0.14 for i in range(len(df))],
                       ticktext=df["Location"].tolist(),showgrid=False))
        fig.update_xaxes(showgrid=True,gridcolor="#E2E8F0")
        return fig

    def wqi_trend(self, wqi_df):
        df = wqi_df.copy(); df["Date"] = pd.to_datetime(df["Date"])
        m  = df.groupby(["Location",pd.Grouper(key="Date",freq="ME")])["WQI"].mean().reset_index()
        fig = px.line(m,x="Date",y="WQI",color="Location",
                      color_discrete_map=LOC_COLORS,markers=True)
        for lo,hi,_,col in NSF_CLASSES[:3]:
            fig.add_hrect(y0=lo,y1=hi,fillcolor=col,opacity=0.04,line_width=0)
        return self._lay(fig,"NSF WQI Trend — Monthly Average")

    def ccme_bar(self, ccme_df):
        d = ccme_df.copy()
        d["Season"] = pd.Categorical(d["Season"],categories=SEASON_ORDER,ordered=True)
        d = d.sort_values("Season")
        fig = px.bar(d,x="Season",y="CCME_WQI",color="Location",barmode="group",
                     color_discrete_map=LOC_COLORS)
        for lo,hi,lbl,col in CCME_CLASSES[:3]:
            fig.add_hrect(y0=lo,y1=hi,fillcolor=col,opacity=0.04,line_width=0,
                          annotation_text=lbl,annotation_position="right")
        return self._lay(fig,"CCME WQI by Season and Location")

    def ccme_heatmap(self, ccme_df):
        piv = ccme_df.pivot_table(index="Season",columns="Location",values="CCME_WQI",aggfunc="mean")
        rows = [s for s in SEASON_ORDER   if s in piv.index]
        cols = [l for l in LOCATION_ORDER if l in piv.columns] + [l for l in piv.columns if l not in LOCATION_ORDER]
        piv  = piv.reindex(index=rows,columns=cols)
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=cols, y=rows,
            colorscale=[[0,"#EF4444"],[0.45,"#FACC15"],[0.8,"#4ADE80"],[1,"#16A34A"]],
            zmin=0,zmax=100,text=[[f"{v:.1f}" for v in r] for r in piv.values],
            texttemplate="%{text}",
            hovertemplate="%{y} — %{x}: CCME = %{z:.1f}<extra></extra>"))
        return self._lay(fig,"CCME WQI Heatmap — Season × Location")

    def radar(self, rel_x):
        fig = go.Figure()
        for _, row in rel_x.iterrows():
            vals = [row[f"mu_{p}"] for p in PARAMS]+[row[f"mu_{PARAMS[0]}"]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=PARAMS+[PARAMS[0]],fill="toself",
                name=row["Location"],line_color=LOC_COLORS.get(row["Location"],"#64748B"),opacity=0.75))
        fig.update_layout(**self.LAYOUT,
            polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor="#E2E8F0"),
                       bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Parameter Membership Profiles — Radar (μ)",font=dict(size=13)))
        return fig

    def ifs_scatter(self, ifs_t):
        fig = px.scatter(ifs_t,x="Mu_T",y="Nu_T",color="Location",size="Pi_T",size_max=22,
                         symbol="Season",color_discrete_map=LOC_COLORS,
                         hover_data=["Season","Pi_T","IFS"])
        fig.add_trace(go.Scatter(x=[0,1],y=[1,0],mode="lines",
            line=dict(dash="dot",color="#94A3B8",width=1),name="IFS boundary μ+ν=1"))
        fig.update_xaxes(range=[0,1],title="Membership μ (quality)")
        fig.update_yaxes(range=[0,1],title="Non-membership ν (pollution)")
        return self._lay(fig,"IFS Space — μ vs ν  (bubble size = π hesitation)")

    def f_components_bar(self, ccme_df):
        """Stacked bar of F1, F2, F3 per location."""
        d = ccme_df.groupby("Location")[["F1_Scope","F2_Frequency","F3_Amplitude"]].mean().reset_index()
        locs = [l for l in LOCATION_ORDER if l in d["Location"].values]
        d = d.set_index("Location").reindex(locs).reset_index()
        fig = go.Figure()
        for col, clr, lbl in [("F1_Scope","#3B82F6","F1 Scope"),
                                ("F2_Frequency","#F59E0B","F2 Frequency"),
                                ("F3_Amplitude","#EF4444","F3 Amplitude")]:
            fig.add_trace(go.Bar(name=lbl,x=d["Location"],y=d[col],marker_color=clr))
        fig.update_layout(**self.LAYOUT,barmode="group",
            title=dict(text="CCME WQI Components — F1, F2, F3 per Location",font=dict(size=13)))
        return fig

    def wqi_compare(self, wqi_summary, ccme_df):
        """Side-by-side NSF vs CCME average WQI by location."""
        nsf  = wqi_summary.groupby("Location")["WQI_Mean"].mean().reset_index().rename(columns={"WQI_Mean":"NSF WQI"})
        cm   = ccme_df.groupby("Location")["CCME_WQI"].mean().reset_index().rename(columns={"CCME_WQI":"CCME WQI"})
        merged = nsf.merge(cm, on="Location")
        locs = [l for l in LOCATION_ORDER if l in merged["Location"].values]
        merged = merged.set_index("Location").reindex(locs).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="NSF WQI",x=merged["Location"],y=merged["NSF WQI"],
                             marker_color="#0369A1"))
        fig.add_trace(go.Bar(name="CCME WQI",x=merged["Location"],y=merged["CCME WQI"],
                             marker_color="#0284C7"))
        fig.update_layout(**self.LAYOUT,barmode="group",
            title=dict(text="NSF vs CCME WQI Comparison by Location",font=dict(size=13)),
            yaxis=dict(range=[0,105]))
        return fig

    def location_map(self, wqi_summary=None, ccme_df=None):
        """Interactive map of the 4 monitoring stations along the Yamuna Delhi stretch."""
        lats  = [LOCATION_COORDS[l]["lat"]  for l in LOCATION_ORDER if l in LOCATION_COORDS]
        lons  = [LOCATION_COORDS[l]["lon"]  for l in LOCATION_ORDER if l in LOCATION_COORDS]
        names = [l for l in LOCATION_ORDER if l in LOCATION_COORDS]
        descs = [LOCATION_COORDS[l]["desc"] for l in LOCATION_ORDER if l in LOCATION_COORDS]
        colors = [LOC_COLORS.get(l, "#64748B") for l in names]

        # Build hover text with WQI data if available
        hover_texts = []
        marker_sizes = []
        for loc in names:
            lines = [f"<b>{loc}</b>", LOCATION_COORDS[loc]["desc"]]
            if wqi_summary is not None:
                wq = wqi_summary[wqi_summary["Location"] == loc]["WQI_Mean"]
                if not wq.empty:
                    avg = wq.mean()
                    lbl, _ = nsf_class(avg)
                    lines.append(f"NSF WQI: {avg:.1f} ({lbl})")
            if ccme_df is not None:
                cq = ccme_df[ccme_df["Location"] == loc]["CCME_WQI"]
                if not cq.empty:
                    avg_c = cq.mean()
                    clbl, _ = ccme_class(avg_c)
                    lines.append(f"CCME WQI: {avg_c:.1f} ({clbl})")
            hover_texts.append("<br>".join(lines))
            marker_sizes.append(20 if wqi_summary is not None else 16)

        fig = go.Figure()
        # Draw the river path
        fig.add_trace(go.Scattermap(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=3, color="#0284C7"),
            name="Yamuna River (Delhi stretch)",
            hoverinfo="skip",
        ))
        # Draw station markers
        fig.add_trace(go.Scattermap(
            lat=lats, lon=lons, mode="markers+text",
            marker=dict(size=marker_sizes, color=colors, opacity=0.92),
            text=names, textposition="top right",
            textfont=dict(size=11, color="white"),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Monitoring Stations",
        ))
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=28.68, lon=77.26),
                zoom=9.5,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1, x=0.01, y=0.99),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 20. REPORT AGENT  (PDF with embedded matplotlib charts)
# ─────────────────────────────────────────────────────────────────────────────

def mpl_fig_to_bytes(fig, dpi=150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf

class ReportAgent:
    NAME = "ReportAgent"

    def export_csv(self, state: PipelineState, out_dir="/tmp") -> str:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "export CSV")
        path  = os.path.join(out_dir, f"yamuna_results_{state.run_id}.csv")
        parts = []
        for df, tag in [(state.wqi_summary,"WQI"),(state.ccme_df,"CCME"),
                        (state.ifs_t,"IFS_T"),(state.ifs_r,"IFS_R"),(state.ifs_x,"IFS_X")]:
            if df is not None:
                d = df.copy(); d["Table"] = tag; parts.append(d)
        if state.dist_df is not None:
            d = state.dist_df.reset_index().copy(); d["Table"]="Distance"; parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(path, index=False)
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"CSV: {path}")
        return path

    def _make_mpl_charts(self, state: PipelineState) -> List[Tuple[str, io.BytesIO]]:
        """Return list of (title, png_bytes) tuples for embedding in PDF."""
        charts = []

        # 1. NSF WQI season bar
        if state.wqi_summary is not None:
            fig, ax = plt.subplots(figsize=(10,4))
            df = state.wqi_summary.pivot_table(index="Season",columns="Location",
                                                values="WQI_Mean",aggfunc="mean")
            df = df.reindex([s for s in SEASON_ORDER if s in df.index])
            df.plot(kind="bar", ax=ax, color=[LOC_COLORS.get(c,"#888") for c in df.columns],
                    edgecolor="none", width=0.7)
            ax.axhline(51, color="#F59E0B", linestyle="--", alpha=0.6, label="Medium threshold")
            ax.axhline(26, color="#EF4444", linestyle="--", alpha=0.6, label="Bad threshold")
            ax.set_title("NSF WQI by Season and Location", fontsize=12, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("WQI Score")
            ax.legend(loc="upper right", fontsize=8)
            plt.xticks(rotation=0); plt.tight_layout()
            charts.append(("NSF WQI by Season", mpl_fig_to_bytes(fig)))

        # 2. CCME WQI comparison
        if state.ccme_df is not None and state.wqi_summary is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            nsf  = state.wqi_summary.groupby("Location")["WQI_Mean"].mean()
            ccme = state.ccme_df.groupby("Location")["CCME_WQI"].mean()
            locs = [l for l in LOCATION_ORDER if l in nsf.index]
            colors = [LOC_COLORS.get(l,"#888") for l in locs]
            axes[0].bar(locs, [nsf.get(l,0) for l in locs], color=colors, edgecolor="none")
            axes[0].set_title("NSF WQI (avg)", fontweight="bold"); axes[0].set_ylim(0,105)
            axes[0].axhline(51, color="#F59E0B", linestyle="--", alpha=0.6)
            axes[1].bar(locs, [ccme.get(l,0) for l in locs], color=colors, edgecolor="none")
            axes[1].set_title("CCME WQI (avg)", fontweight="bold"); axes[1].set_ylim(0,105)
            axes[1].axhline(65, color="#F59E0B", linestyle="--", alpha=0.6)
            for ax in axes: ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            charts.append(("NSF vs CCME WQI Comparison", mpl_fig_to_bytes(fig)))

        # 3. IFS T heatmap
        if state.ifs_t is not None:
            piv = state.ifs_t.pivot_table(index="Season",columns="Location",
                                          values="Mu_T",aggfunc="mean")
            piv = piv.reindex(index=[s for s in SEASON_ORDER if s in piv.index],
                              columns=[l for l in LOCATION_ORDER if l in piv.columns])
            fig, ax = plt.subplots(figsize=(8,4))
            im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=20)
            ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    ax.text(j, i, f"{piv.values[i,j]:.3f}", ha="center", va="center",
                            fontsize=9, color="black")
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.set_title("IFS Relation T — μ_T (Season × Location)", fontweight="bold")
            plt.tight_layout()
            charts.append(("IFS Composition Heatmap", mpl_fig_to_bytes(fig)))

        # 4. Lollipop distance
        if state.dist_df is not None:
            df = state.dist_df.reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            for i, row in df.iterrows():
                c = LOC_COLORS.get(row["Location"],"#64748B")
                ax.hlines(i,      0, row["Hamming"],   color=c, linewidth=3)
                ax.plot(row["Hamming"], i, "o", color=c, markersize=12, label=f"{row['Location']} (H)")
                ax.hlines(i+0.25, 0, row["Euclidean"], color=c, linewidth=2, linestyle="--")
                ax.plot(row["Euclidean"], i+0.25, "D", color=c, markersize=9)
            ax.set_yticks([i+0.125 for i in range(len(df))])
            ax.set_yticklabels(df["Location"].tolist())
            ax.set_xlabel("Distance from Palla")
            ax.set_title("IFS Distance Analysis — Hamming (●) & Euclidean (◆)", fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            charts.append(("Distance from Palla", mpl_fig_to_bytes(fig)))

        # 5. CCME F-components
        if state.ccme_df is not None:
            d = state.ccme_df.groupby("Location")[["F1_Scope","F2_Frequency","F3_Amplitude"]].mean()
            locs = [l for l in LOCATION_ORDER if l in d.index] + [l for l in d.index if l not in LOCATION_ORDER]
            d = d.reindex(locs)
            x = range(len(locs))
            w = 0.25
            fig, ax = plt.subplots(figsize=(9,4))
            ax.bar([i-w for i in x], d["F1_Scope"],   width=w, label="F1 Scope",     color="#3B82F6")
            ax.bar([i   for i in x], d["F2_Frequency"],width=w,label="F2 Frequency",  color="#F59E0B")
            ax.bar([i+w for i in x], d["F3_Amplitude"],width=w,label="F3 Amplitude",  color="#EF4444")
            ax.set_xticks(list(x)); ax.set_xticklabels(locs, rotation=15)
            ax.set_title("CCME WQI Components F1, F2, F3 by Location", fontweight="bold")
            ax.legend(); ax.set_ylabel("Value")
            plt.tight_layout()
            charts.append(("CCME F-Components", mpl_fig_to_bytes(fig)))

        return charts

    def export_pdf(self, state: PipelineState, out_dir="/tmp") -> str:
        BUS.publish(self.NAME, "Coordinator", "REQUEST", "export PDF with charts")
        path = os.path.join(out_dir, f"yamuna_report_{state.run_id}.pdf")

        if not RL_OK:
            path = path.replace(".pdf",".txt")
            with open(path,"w") as f: f.write(self._text(state))
            return path

        doc    = SimpleDocTemplate(path, pagesize=A4,
                                   leftMargin=1.5*cm, rightMargin=1.5*cm,
                                   topMargin=2*cm,    bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        H1 = ParagraphStyle("H1",parent=styles["Heading1"],fontSize=16,
                             textColor=rl_colors.HexColor("#0369A1"),spaceAfter=8)
        H2 = ParagraphStyle("H2",parent=styles["Heading2"],fontSize=13,
                             textColor=rl_colors.HexColor("#0284C7"),spaceAfter=6)
        BD = ParagraphStyle("BD",parent=styles["Normal"],fontSize=10,leading=14,spaceAfter=3)
        SM = ParagraphStyle("SM",parent=styles["Normal"],fontSize=8,textColor=rl_colors.grey)

        def make_table(df, header_color="#0369A1"):
            cols = df.columns.tolist()
            data = [cols] + [[str(round(v,3)) if isinstance(v,float) else str(v)
                              for v in row] for row in df.values.tolist()]
            col_w = [A4[0]*0.85/len(cols)] * len(cols)
            t = Table(data, colWidths=col_w, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor(header_color)),
                ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
                ("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor("#F8FAFC")]),
                ("GRID",(0,0),(-1,-1),0.4,rl_colors.HexColor("#CBD5E1")),
                ("ALIGN",(1,1),(-1,-1),"CENTER"),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            return t

        story = []
        story += [
            Paragraph("Yamuna River Water Quality Assessment Report", H1),
            Paragraph(f"Run ID: {state.run_id}  |  Generated: {state.run_ts[:19]}", SM),
            Spacer(1,0.2*cm),
            HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor("#0369A1")),
            Spacer(1,0.3*cm),
        ]

        # Tables
        if state.wqi_summary is not None:
            story += [Paragraph("NSF WQI Summary", H2),
                      make_table(state.wqi_summary[["Location","Season","WQI_Mean","WQI_Class"]].round(3)),
                      Spacer(1,0.4*cm)]
        if state.ccme_df is not None:
            story += [Paragraph("CCME WQI Summary", H2),
                      make_table(state.ccme_df[["Location","Season","CCME_WQI","CCME_Class",
                                                "F1_Scope","F2_Frequency","F3_Amplitude"]]),
                      Spacer(1,0.4*cm)]
        if state.ifs_t is not None:
            story += [Paragraph("IFS Relation T — Composition", H2),
                      make_table(state.ifs_t[["Season","Location","Mu_T","Nu_T","Pi_T","IFS"]]),
                      Spacer(1,0.4*cm)]
        if state.dist_df is not None:
            story += [Paragraph("Location Rankings (Distance from Palla)", H2),
                      make_table(state.dist_df.reset_index()),
                      Spacer(1,0.4*cm)]

        # Alerts
        if state.decisions:
            story.append(Paragraph("Decision Alerts", H2))
            for d in state.decisions:
                col = rl_colors.HexColor(ALERT_COLORS.get(d["level"],"#64748B"))
                story.append(Paragraph(
                    f"<b>[{d['level']}]</b> {d['location']} — {d['param']}: {d['message']}", BD))
                story.append(Paragraph(f"  Action: {d['action']}", SM))
                if d.get("ai_explanation"):
                    story.append(Paragraph(f"  AI Analysis: {d['ai_explanation']}", SM))
            story.append(Spacer(1,0.4*cm))

        # Charts
        story.append(PageBreak())
        story.append(Paragraph("Analysis Charts", H1))
        story.append(Spacer(1,0.3*cm))

        charts = self._make_mpl_charts(state)
        for title, png_bytes in charts:
            story.append(Paragraph(title, H2))
            story.append(RLImage(png_bytes, width=16*cm, height=7*cm))
            story.append(Spacer(1,0.5*cm))

        # Footer
        story += [
            HRFlowable(width="100%",thickness=0.5,color=rl_colors.HexColor("#CBD5E1")),
            Paragraph("Water Quality Assessment Platform v3.0"
                  , SM),
        ]

        doc.build(story)
        BUS.publish(self.NAME, "Coordinator", "RESULT", f"PDF: {path}")
        return path

    def _text(self, state):
        lines = ["YAMUNA WATER QUALITY REPORT\n" + "="*60]
        if state.wqi_summary is not None: lines.append("\nNSF WQI:\n"+state.wqi_summary.to_string())
        if state.ccme_df is not None:     lines.append("\nCCME WQI:\n"+state.ccme_df.to_string())
        if state.dist_df is not None:     lines.append("\nDistances:\n"+state.dist_df.to_string())
        if state.decisions:
            lines.append("\nAlerts:")
            for d in state.decisions: lines.append(f"  [{d['level']}] {d['location']}: {d['message']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 21. COORDINATOR AGENT  (dynamic pipeline with retry/skip + memory)
# ─────────────────────────────────────────────────────────────────────────────

OFFICIAL_ONLY = {"run_pipeline","upload_data","delete_record","set_threshold"}

class CoordinatorAgent:
    NAME = "CoordinatorAgent"

    def __init__(self, persist_memory: bool = True):
        self.persist_memory    = persist_memory
        self.preprocess_agent  = PreprocessingAgent()
        self.wqi_agent         = WQIAgent()
        self.ccme_agent        = CCMEAgent()
        self.ifs_agent         = IFSAgent()
        self.inference_agent   = InferenceAgent()
        self.distance_agent    = DistanceAgent()
        self.decision_agent    = LLMDecisionAgent()
        self.alert_agent       = AlertNotificationAgent()
        self.viz_agent         = VisualizationAgent()
        self.report_agent      = ReportAgent()
        self._cache: Optional[PipelineState] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=POLICY_CONFIG["max_workers"])

    def check_perm(self, action: str, session: Optional[UserSession]) -> bool:
        if session is None: return False
        if action in OFFICIAL_ONLY and session.role != "official": return False
        return True

    def run_pipeline(self, df: pd.DataFrame, config: AnalysisConfig,
                     session: UserSession,
                     official_emails: Optional[List[str]] = None) -> PipelineState:

        if not self.check_perm("run_pipeline", session):
            s = PipelineState()
            s.errors.append("ACCESS_DENIED: Only Officials can run the pipeline.")
            return s

        BUS.publish(self.NAME, "ALL", "REQUEST",
                    f"Starting pipeline | user={session.username} | rows={len(df)}")
        state = PipelineState(raw_df=df, config=config)

        # ── Step 1: Preprocess (tracks imputed_mask for IFS missingness π) ─────
        for attempt in range(2):
            res = self.preprocess_agent.run(df, config)
            if res.status == "OK":
                state.cleaned_df, state.imputed_mask, state.validation_log = res.data
                break
            elif res.status == "WARN" and attempt == 0:
                BUS.publish(self.NAME, self.preprocess_agent.NAME, "RETRY_REQUEST",
                            "Retrying with relaxed date filter")
                continue
            else:
                state.errors.append(f"Preprocessing: {res.message}")
                return state

        # ── Step 2–6: Adaptive DAG-Pipelined Execution (Vectorized + Persistent Pool) ──
        # Branch A (IFS -> Inference -> Distance) executes concurrently with Branch B (WQI) & Branch C (CCME)
        def _run_ifs_dag():
            r_ifs = self.ifs_agent.run(state.cleaned_df, config, state.imputed_mask)
            if r_ifs.status != "OK":
                return r_ifs, None, None
            r_inf = self.inference_agent.run(r_ifs.data[2], config)
            if r_inf.status != "OK":
                return r_ifs, r_inf, None
            r_dist = self.distance_agent.run(r_inf.data, config)
            return r_ifs, r_inf, r_dist

        if len(state.cleaned_df) >= config.parallel_threshold:
            f_wqi  = self._executor.submit(self.wqi_agent.run, state.cleaned_df, config, state.imputed_mask)
            f_ccme = self._executor.submit(self.ccme_agent.run, state.cleaned_df, state.imputed_mask)
            f_dag  = self._executor.submit(_run_ifs_dag)
            res_wqi, res_ccme = f_wqi.result(), f_ccme.result()
            res_ifs, res_inf, res_dist = f_dag.result()
        else:
            res_wqi  = self.wqi_agent.run(state.cleaned_df, config, state.imputed_mask)
            res_ccme = self.ccme_agent.run(state.cleaned_df, state.imputed_mask)
            res_ifs, res_inf, res_dist = _run_ifs_dag()

        if res_wqi.status == "OK":
            state.wqi_df, state.wqi_summary = res_wqi.data
        else:
            state.errors.append(f"WQI: {res_wqi.message}")

        if res_ccme.status == "OK":
            state.ccme_df = res_ccme.data
        else:
            state.errors.append(f"CCME: {res_ccme.message}")

        if res_ifs.status != "OK":
            state.errors.append(f"IFS: {res_ifs.message}")
            return state
        state.ifs_r, state.ifs_x, state.ifs_df = res_ifs.data

        if res_inf is None or res_inf.status != "OK":
            state.errors.append(f"Inference: {res_inf.message if res_inf else 'aborted'}")
            return state
        state.ifs_t = res_inf.data

        if res_dist is not None and res_dist.status == "OK":
            state.dist_df = res_dist.data
        else:
            state.errors.append(f"Distance: {res_dist.message if res_dist else 'aborted'}")

        # ── Step 7: Decisions (LLM-enhanced) ─────────────────────────────────
        if all(x is not None for x in (state.wqi_summary, state.ifs_t, state.dist_df)):
            res = self.decision_agent.run(state, MEMORY)
            if res.status == "OK":
                state.decisions = res.data

        # ── Step 8: Email critical alerts to all officials ────────────────────
        if state.decisions and official_emails:
            critical = [d for d in state.decisions if d.get("level") == "CRITICAL"]
            if critical:
                BUS.publish(self.NAME, self.alert_agent.NAME, "ALERT",
                            f"{len(critical)} CRITICAL alerts — notifying {len(official_emails)} officials")
                self.alert_agent.send_to_all_officials(state.decisions, official_emails)

        # ── Store in memory (skipped during synthetic stress/sensitivity runs) ──
        if self.persist_memory:
            MEMORY.store(state)
        state.agent_log_snapshot = LOGGER.get_recent(80)
        self._cache = state
        BUS.publish(self.NAME, "ALL", "RESULT",
                    f"Pipeline COMPLETE | run_id={state.run_id} | errors={len(state.errors)}")
        return state

    def get_cached(self) -> Optional[PipelineState]:
        return self._cache


# ─────────────────────────────────────────────────────────────────────────────
# 22. SAMPLE DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(n_per=36) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    profiles = {
        "Palla":         dict(pH=(7.6,0.2),  DO=(7.0,0.8),  BOD=(2.5,0.8),   FC=(200,120)),
        "Wazirabad":     dict(pH=(7.4,0.3),  DO=(5.5,1.0),  BOD=(8.0,2.0),   FC=(2200,700)),
        "ITO Bridge":    dict(pH=(7.2,0.4),  DO=(3.2,1.2),  BOD=(22.0,4.0),  FC=(24000,7000)),
        "Okhla Barrage": dict(pH=(7.0,0.5),  DO=(1.8,0.8),  BOD=(38.0,6.0),  FC=(78000,14000)),
    }
    adj = {
        "Monsoon":      dict(DO=1.5,  BOD=-3.0, FC=8000),
        "Summer":       dict(DO=-1.2, BOD=5.0,  FC=6000),
        "Winter":       dict(DO=0.6,  BOD=-1.0, FC=-1500),
        "Post-Monsoon": dict(DO=0.4,  BOD=-2.0, FC=-2500),
    }
    dates = pd.date_range("2020-01-01", periods=n_per*4, freq="W")
    for loc, p in profiles.items():
        for d in dates:
            s = get_season(d.month); a = adj.get(s,{})
            rows.append({
                "Date":     d.strftime("%Y-%m-%d"),
                "Location": loc,
                "pH":  round(np.clip(np.random.normal(p["pH"][0],    p["pH"][1]),    5.5,9.5), 2),
                "DO":  round(np.clip(np.random.normal(p["DO"][0]+a.get("DO",0),   p["DO"][1]),    0.1,14), 2),
                "BOD": round(np.clip(np.random.normal(p["BOD"][0]+a.get("BOD",0), p["BOD"][1]),   0.5,250), 2),
                "FC":  round(np.clip(np.random.normal(p["FC"][0]+a.get("FC",0),   p["FC"][1]),    0,200000), 0),
            })
    return pd.DataFrame(rows)



# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

auth_agent    = AuthAgent()
coordinator   = CoordinatorAgent()
llm_assistant = LLMAssistantAgent()
_STATE: Optional[PipelineState] = None
_SESSION: Optional[UserSession]  = None

# --- NEW MCP SERVER INITIALIZATION ---
try:
    from mcp.server.mcpserver import MCPServer
    mcp_server = MCPServer("Yamuna_Platform_Server")
except ImportError:
    class _NoOpMCPServer:
        def __init__(self, name): self.name = name
        def resource(self, *a, **kw): return lambda fn: fn
        def tool(self, *a, **kw): return lambda fn: fn
        def prompt(self, *a, **kw): return lambda fn: fn
    mcp_server = _NoOpMCPServer("Yamuna_Platform_Server")

# ─── MCP RESOURCES & TOOLS ────────────────────────────────────────────────────

@mcp_server.resource("memory://historical_context")
def get_historical_memory() -> str:
    """Returns historical pipeline runs and Yamuna river quality trends."""
    return MEMORY.get_llm_context()

@mcp_server.resource("logs://agent_activity")
def get_agent_logs() -> str:
    """Returns recent inter-agent communication logs."""
    return LOGGER.get_log_text(80)

@mcp_server.tool()
def run_water_quality_pipeline(file_path: str = "", start_date: str = "", end_date: str = "") -> str:
    """Runs the multi-agent water quality pipeline (NSF WQI, CCME, IFS) on any river dataset."""
    try:
        df, resolved_path = load_real_dpcc_data(explicit_path=file_path or None, allow_colab_upload=False)
    except Exception as e:
        return f"Failed to load data: {e}"

    config = AnalysisConfig(
        start_date=start_date or None,
        end_date=end_date or None,
        locations=resolve_locations(df),
    )
    mcp_session = UserSession("sys_mcp", "mcp_client", "official", "MCP Automated Client")

    state = coordinator.run_pipeline(df, config, mcp_session)

    if not state.ok():
        return f"Pipeline errors: {state.errors}"

    avg_wqi = state.wqi_summary["WQI_Mean"].mean()
    worst_loc = state.dist_df.sort_values("Hamming", ascending=False).iloc[0]["Location"]
    return f"Success. Average NSF WQI: {avg_wqi:.2f}. Most polluted location: {worst_loc}."

@mcp_server.prompt("evaluate_yamuna_data")
def evaluate_yamuna_prompt() -> str:
    """Triggers expert evaluation of current water quality data."""
    state = coordinator.get_cached()
    if not state:
        return "No cached state available. Run the pipeline tool first."

    data_summary = LLMDecisionAgent._build_data_summary(state)
    history_ctx = MEMORY.get_llm_context()

    return LLM_DECISION_PROMPT.format(
        data_summary=data_summary,
        history_context=history_ctx
    )

# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
/* ── typography ── */
.gradio-container { font-family:'Inter','Segoe UI',sans-serif !important; }

/* ── login page ── */
#login-wrap { min-height:100vh; display:flex; align-items:center; justify-content:center;
              background:linear-gradient(135deg,#0369A1 0%,#0284C7 40%,#075985 100%); }
#login-card { background:#BAE6FD; border-radius:20px; padding:40px 44px; width:420px;
              box-shadow:0 20px 60px rgba(0,0,0,0.18); }
#login-logo { text-align:center; margin-bottom:24px; }
#login-logo h1 { font-size:22px; font-weight:700; color:#0369A1; margin:0 0 4px; }
#login-logo p  { font-size:13px; color:#64748B; margin:0; }
.login-wave { font-size:40px; display:block; text-align:center; margin-bottom:12px; }
#login-card input[type=text], #login-card input[type=password] {
  border:1.5px solid #E2E8F0 !important; border-radius:10px !important;
  font-size:14px !important; }
#login-card input:focus { border-color:#0369A1 !important; box-shadow:0 0 0 3px #0369A120 !important; }
#login-submit { background:linear-gradient(135deg,#0369A1,#0284C7) !important;
                color:white !important; border:none !important; border-radius:10px !important;
                font-size:14px !important; font-weight:600 !important;
                width:100%; height:44px !important; }

/* ── role banner ── */
#role-banner { border-radius:10px; padding:10px 16px; margin-bottom:10px;
               display:flex; align-items:center; gap:10px; }
.role-off { background:#F0FDF4; border:1px solid #86EFAC; }
.role-vw  { background:#EFF6FF; border:1px solid #93C5FD; }
.role-badge { font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px;
              letter-spacing:0.04em; }
.badge-off { background:#16A34A; color:white; }
.badge-vw  { background:#2563EB; color:white; }

/* ── app header strip ── */
#app-strip { background:linear-gradient(135deg,#0369A1,#0284C7);
             border-radius:10px; padding:12px 20px; margin-bottom:12px;
             display:flex; align-items:center; justify-content:space-between; }
#app-strip h2 { color:white; margin:0; font-size:17px; font-weight:600; }
#app-strip p  { color:rgba(255,255,255,0.75); margin:0; font-size:12px; }
#logout-btn { background:rgba(255,255,255,0.15) !important; color:white !important;
              border:1px solid rgba(255,255,255,0.3) !important;
              border-radius:6px !important; font-size:12px !important;
              padding:4px 12px !important; cursor:pointer; }
#logout-btn:hover { background:rgba(255,255,255,0.25) !important; }

/* ── kpi cards ── */
.kpi-row  { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
             gap:10px; margin:8px 0; }
.kpi      { background:#BAE6FD; border-radius:10px; border:1px solid #E2E8F0;
             padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.kpi-lbl  { font-size:10px; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; }
.kpi-val  { font-size:22px; font-weight:700; color:white; margin-top:2px; }
.kpi-sub  { font-size:11px; color:#94A3B8; margin-top:1px; }

/* ── section headers ── */
.sec-hd { font-size:13px; font-weight:600; color:white;
           border-bottom:2px solid #0369A1; padding-bottom:5px; margin-bottom:10px; }

/* ── alert cards ── */
.alert-card { background:#BAE6FD; border-radius:0 8px 8px 0; margin:5px 0;
               padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }

/* ── run button ── */
#run-btn > button { background:linear-gradient(135deg,#0369A1,#0284C7) !important;
                    color:white !important; border:none !important;
                    border-radius:8px !important; font-weight:600 !important;
                    font-size:14px !important; }
#run-btn > button:hover { opacity:0.9; }

/* ── chat sidebar ── */
#chat-sidebar { background:#BAE6FD; border-radius:12px; border:1px solid #E2E8F0;
                padding:14px; height:fit-content; position:sticky; top:0; }
#chat-header { font-size:13px; font-weight:600; color:#0369A1; margin-bottom:10px;
               display:flex; align-items:center; gap:6px; }
#send-btn > button { background:#0369A1 !important; color:white !important;
                     border:none !important; border-radius:8px !important;
                     font-size:13px !important; font-weight:600 !important; }

/* ── log panel ── */
#log-area textarea { font-family:'JetBrains Mono','Courier New',monospace !important;
                     font-size:11px !important; background:white !important;
                     color:#94A3B8 !important; border-radius:8px !important; }

/* ── interpretation box ── */
.interp { background:#F0FDF4; border-left:3px solid #0369A1; border-radius:0 8px 8px 0;
          padding:10px 14px; font-size:12px; color:black !important; margin:8px 0; }
.interp-warn { background:#FFFBEB; border-left-color:#D97706; color:black !important; }
.interp-crit { background:#FEF2F2; border-left-color:#DC2626; color:black !important; }
.interp b, .interp-warn b, .interp-crit b { color:black !important; }
"""

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def kpi_html(state: PipelineState, role: str) -> str:
    if state is None or state.wqi_summary is None:
        return ""
    avg_wqi   = state.wqi_summary["WQI_Mean"].mean()
    best_loc  = state.wqi_summary.groupby("Location")["WQI_Mean"].mean().idxmax()
    n_crit    = sum(1 for d in state.decisions if d.get("level") == "CRITICAL")
    n_warn    = sum(1 for d in state.decisions if d.get("level") == "WARNING")
    lbl, col  = nsf_class(avg_wqi)
    memory_n  = MEMORY.count()
    trend_str = MEMORY.detect_trend(best_loc) if memory_n > 1 else "—"
    wc = "#DC2626" if n_crit > 0 else "#16A34A"
    ww = "#D97706" if n_warn  > 0 else "#16A34A"

    avg_ccme = state.ccme_df["CCME_WQI"].mean() if state.ccme_df is not None else 0
    clbl, _ = ccme_class(avg_ccme)

    def c(lbl, val, sub="", col="white"):
        return f'<div class="kpi"><div class="kpi-lbl">{lbl}</div><div class="kpi-val" style="color:{col}">{val}</div><div class="kpi-sub">{sub}</div></div>'

    return (
        f'<div class="kpi-row">'
        + c("NSF WQI", round(avg_wqi,1), lbl, col)
        + c("CCME WQI", round(avg_ccme,1), clbl, "#0284C7")
        + c("Best Site", best_loc, f"Trend: {trend_str}", "#16A34A")
        + c("Critical", n_crit, "alerts", wc)
        + c("Warnings", n_warn, "alerts", ww)
        + (c("Historical Runs", memory_n, "in memory", "#7C3AED") if role=="official" else "")
        + "</div>"
    )

def build_ifs_display(rel_df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    out = pd.DataFrame({index_col: rel_df[index_col]})
    for p in PARAMS:
        out[p] = rel_df.apply(
            lambda r: fmt_ifs(
                r[f"mu_{p}"],
                r[f"nu_{p}"],
                r[f"pi_{p}"] if f"pi_{p}" in r.index else max(0.0, 1.0 - r[f"mu_{p}"] - r[f"nu_{p}"]),
            ),
            axis=1,
        )
    return out

def val_html(log: dict) -> str:
    if not log: return ""
    iss  = log.get("issues", [])
    items = "".join(f'<li style="color:#D97706;font-size:12px">{i}</li>' for i in iss) or \
            '<li style="color:#16A34A;font-size:12px">No issues — clean data.</li>'
    return (f'<div style="background:#BAE6FD;border-radius:8px;border:1px solid #E2E8F0;padding:12px;font-size:12px">'
            f'<b>Validation Report</b> — {log.get("original_rows",0)} rows → {log.get("final_rows",0)} after cleaning'
            f'<ul style="margin:4px 0 0 16px">{items}</ul>'
            f'<div style="color:#64748B;margin-top:4px">Seasons: {log.get("season_dist",{})} | Locations: {log.get("location_dist",{})}</div></div>')

def ccme_interp_html(ccme_df: pd.DataFrame) -> str:
    if ccme_df is None: return ""
    parts = []
    for loc in LOCATION_ORDER:
        rows = ccme_df[ccme_df["Location"]==loc]
        if rows.empty: continue
        for _, row in rows.iterrows():
            interp = CCMEWQIEngine.interpretation(row)
            cls_   = row["CCME_Class"]
            css = "interp-crit" if cls_ in ("Poor",) else ("interp-warn" if cls_ in ("Marginal","Fair") else "interp")
            parts.append(f'<div class="{css}"><b>{loc} — {row["Season"]}</b>: {interp}</div>')
    return "\n".join(parts)


# ─── BUILD GRADIO APP ─────────────────────────────────────────────────────────

def build_app():
    global _STATE, _SESSION
    import inspect

    _theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate")
    _blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    _blocks_kwargs = {"title": "Yamuna WQI Platform v3.0"}
    if "css" in _blocks_params:
        _blocks_kwargs["css"] = CSS
    if "theme" in _blocks_params:
        _blocks_kwargs["theme"] = _theme

    with gr.Blocks(**_blocks_kwargs) as app:

        session_st  = gr.State(None)
        chat_hist   = gr.State([])

        # ══════════════════════════════════════════════════════════════════
        #  LOGIN SCREEN  (full width, shown before login)
        # ══════════════════════════════════════════════════════════════════
        with gr.Group(visible=True, elem_id="login-wrap") as login_screen:
            with gr.Row():
                with gr.Column(scale=1): pass
                with gr.Column(scale=2, min_width=420, elem_id="login-card"):
                    gr.HTML("""
<div id="login-logo">
  <span class="login-wave">🌊</span>
  <h1>River Water Quality Platform</h1>
  <p>Multi-Agent Assessment System  v3.0</p>
</div>""")
                    login_user = gr.Textbox(label="Username",  placeholder="official  or  viewer")
                    login_pass = gr.Textbox(label="Password",  placeholder="password", type="password")
                    login_btn  = gr.Button("Sign In", elem_id="login-submit", variant="primary")
                    login_msg  = gr.HTML()
                    gr.HTML("""
<div style="margin-top:20px;padding:12px;background: #40E0D0;border-radius:10px;
            font-size:12px;color:black;line-height:1.9">
  <b>Demo credentials</b><br>
  <code>official</code> / <code>official123</code>  — full access<br>
  <code>viewer</code>   / <code>viewer123</code>    — read-only (charts &amp; chat)
</div>""")
                with gr.Column(scale=1): pass

        # ══════════════════════════════════════════════════════════════════
        #  MAIN APP  (hidden until login, left+right layout)
        # ══════════════════════════════════════════════════════════════════
        with gr.Group(visible=False) as main_app:

            # ── header strip + logout ─────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=5):
                    app_header = gr.HTML()
                with gr.Column(scale=1, min_width=120):
                    logout_btn = gr.Button("⬅  Logout", size="sm", elem_id="logout-btn")

            # ── two-column layout: tabs (left) + chat (right) ─────────────
            with gr.Row(equal_height=False):

                # ── LEFT: role-adaptive tabs ──────────────────────────────
                with gr.Column(scale=3):
                    role_banner = gr.HTML()

                    with gr.Tabs() as main_tabs:

                        # ── Dashboard (both roles) ─────────────────────────
                        with gr.Tab("Dashboard"):
                            dash_kpi    = gr.HTML()
                            dash_alerts = gr.HTML()
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.HTML('<div class="sec-hd">Monitoring Stations — Delhi Stretch</div>')
                                    dash_map = gr.Plot(label="", show_label=False)
                                with gr.Column(scale=2):
                                    gr.HTML('<div class="sec-hd">Active Alerts Summary</div>')
                                    dash_alert_detail = gr.HTML(
                                        '<div style="color:#94A3B8;font-size:13px;padding:20px">Run analysis to see alerts.</div>'
                                    )
                            with gr.Row():
                                with gr.Column():
                                    gr.HTML('<div class="sec-hd">NSF WQI Trend</div>')
                                    dash_trend = gr.Plot()
                                with gr.Column():
                                    gr.HTML('<div class="sec-hd">CCME WQI by Season</div>')
                                    dash_ccme_bar = gr.Plot()

                        # ── Data & Run  (official only) ────────────────────
                        with gr.Tab("Data & Run") as tab_data:
                            # Viewer sees lock notice; official sees full panel
                            with gr.Group(visible=True) as data_locked:
                                gr.HTML('''<div style="padding:20px;text-align:center;background:#FEF3C7;
                                    border-radius:10px;border:1px solid #FCD34D;margin:12px 0">
                                    <div style="font-size:28px;margin-bottom:8px">🔒</div>
                                    <div style="font-weight:600;color:#92400E;font-size:15px">Official Access Required</div>
                                    <div style="color:#78350F;font-size:13px;margin-top:4px">
                                    Login as <b>official / official123</b> to upload data and run analysis.</div>
                                    </div>''')
                            with gr.Group(visible=False) as data_panel:
                                gr.HTML('<div class="sec-hd">Upload Water Quality Data</div>')
                                gr.HTML('<div style="font-size:12px;color:#64748B;margin-bottom:10px">Required columns: <code>Date, Location, pH, DO, BOD, FC</code></div>')
                                with gr.Row():
                                    file_input  = gr.File(label="📂  Upload Excel / CSV",
                                                          file_types=[".xlsx",".csv"])
                                with gr.Row():
                                    sample_btn  = gr.Button("📂  Auto-Load Real DPCC All-Sites Data (Drive / River_Data)",
                                                            variant="secondary", size="lg")
                                gr.HTML('<div style="font-size:11px;color:#64748B;margin:4px 0 10px">↑ Automatically detects real DPCC Excel/CSV in /content/ and fills exact date range (synthetic fallback disabled).</div>')
                                with gr.Row():
                                    start_date  = gr.Textbox(label="Start Date", value="2024-01-01", placeholder="YYYY-MM-DD")
                                    end_date    = gr.Textbox(label="End Date",   value="2026-09-30", placeholder="YYYY-MM-DD")
                                loc_check = gr.CheckboxGroup(choices=LOCATION_ORDER, value=LOCATION_ORDER,
                                                             label="Monitoring Locations")
                                with gr.Accordion("⚙️  Parameter Weights (NSF-WQI)", open=False):
                                    gr.HTML('<div style="font-size:12px;color:#64748B;margin-bottom:6px">Weights must sum to 1.0. NSF defaults shown.</div>')
                                    with gr.Row():
                                        w_do  = gr.Slider(0.05,0.50,value=0.309,step=0.001,label="DO weight")
                                        w_fc  = gr.Slider(0.05,0.50,value=0.291,step=0.001,label="FC weight")
                                    with gr.Row():
                                        w_ph  = gr.Slider(0.05,0.40,value=0.200,step=0.001,label="pH weight")
                                        w_bod = gr.Slider(0.05,0.40,value=0.200,step=0.001,label="BOD weight")
                                    wsum_html = gr.HTML('<div style="font-size:12px;color:#16A34A">Weight sum: 1.000 ✓</div>')
                                gr.HTML('<div style="margin:12px 0 6px;font-size:12px;color:#475569"></div>')
                                run_btn    = gr.Button("▶  Run Full Analysis Pipeline",
                                                       variant="primary", size="lg", elem_id="run-btn")
                                run_status = gr.HTML()
                                data_val   = gr.HTML()

                        # ── IFS Analysis  (both roles) ─────────────────────
                        with gr.Tab("IFS Analysis"):
                            gr.HTML('<div class="sec-hd">Relation R — Season × Parameters</div>')
                            gr.HTML('<div style="font-size:12px;color: white;margin-bottom:8px">Average IFS (μ, ν, π) per season. μ close to 1 = good quality.</div>')
                            with gr.Row():
                                r_table   = gr.DataFrame(label="Relation R", interactive=False)
                                r_heatmap = gr.Plot()
                            r_interp = gr.HTML()

                            gr.HTML('<div class="sec-hd" style="margin-top:16px">Relation X — Location × Parameters</div>')
                            gr.HTML('<div style="font-size:12px;color:white;margin-bottom:8px">Average IFS (μ, ν, π) per location. Shows upstream→downstream gradient.</div>')
                            with gr.Row():
                                x_table = gr.DataFrame(label="Relation X", interactive=False)
                                x_radar = gr.Plot()
                            x_interp = gr.HTML()

                            gr.HTML('<div class="sec-hd" style="margin-top:16px">Relation T — Max-Min Composition  (T = R ∘ X)</div>')
                            gr.HTML('<div style="font-size:12px;color: white;margin-bottom:8px">Weighted max-min composition gives final spatiotemporal IFS assessment.</div>')
                            with gr.Row():
                                t_table   = gr.DataFrame(label="Relation T", interactive=False)
                                t_heatmap = gr.Plot()
                            t_interp = gr.HTML()

                            gr.HTML('<div class="sec-hd" style="margin-top:16px">IFS Space Explorer</div>')
                            ifs_scatter = gr.Plot()

                            gr.HTML('<div class="sec-hd" style="margin-top:16px">Distance Analysis — Deviation from Palla</div>')
                            with gr.Row():
                                with gr.Column(scale=1):
                                    dist_table  = gr.DataFrame(label="Rankings", interactive=False)
                                    dist_interp = gr.HTML()
                                with gr.Column(scale=2):
                                    dist_chart  = gr.Plot()

                        # ── CCME WQI  (both roles) ─────────────────────────
                        with gr.Tab("CCME WQI"):
                            gr.HTML('<div class="sec-hd">Canadian Council of Ministers of Environment WQI</div>')
                            gr.HTML("""<div style="font-size:12px;color:white;margin-bottom:8px">
CCME WQI = 100 − √(F1²+F2²+F3²) / 1.732 &nbsp;|&nbsp;
<b>F1</b>: % variables failing &nbsp;|&nbsp; <b>F2</b>: % tests failing &nbsp;|&nbsp; <b>F3</b>: amplitude of failures<br>
Objectives: pH 6.5–8.5 | DO ≥ 6.0 mg/L | BOD ≤ 5 mg/L | FC ≤ 200 CFU/100mL
</div>""")
                            ccme_table   = gr.DataFrame(label="CCME WQI Results", interactive=False)
                            with gr.Row():
                                ccme_hmap    = gr.Plot(label="CCME Heatmap")
                                ccme_bar_plt = gr.Plot(label="CCME by Season")
                            with gr.Row():
                                ccme_f_bar   = gr.Plot(label="F1, F2, F3 Components")
                                ccme_compare = gr.Plot(label="NSF vs CCME Comparison")
                            gr.HTML('<div class="sec-hd" style="margin-top:12px">Interpretations</div>')
                            ccme_interp_out = gr.HTML()

                        # ── Alerts  (both roles, filtered) ────────────────
                        with gr.Tab("Alerts"):
                            gr.HTML('<div class="sec-hd">Decision Alerts (AI-Enhanced)</div>')
                            gr.HTML('<div style="font-size:12px;color:white;margin-bottom:8px">Rule-based. Critical alerts automatically emailed to all officials.</div>')
                            alert_html_out = gr.HTML()
                            alert_df_out   = gr.DataFrame(label="Alert Summary", interactive=False)

                        # ── Reports  (both roles) ─────────────────────────
                        with gr.Tab("Reports"):
                            gr.HTML('<div class="sec-hd">Export Analysis Results</div>')
                            with gr.Row():
                                with gr.Column():
                                    pdf_btn  = gr.Button("Generate PDF Report (with charts)", variant="primary")
                                    pdf_file = gr.File(label="PDF Download")
                                with gr.Column():
                                    csv_btn  = gr.Button("Generate CSV Export", variant="primary")
                                    csv_file = gr.File(label="CSV Download")
                            rep_status = gr.HTML()
                            gr.HTML("""<div style="font-size:12px;color:#64748B;margin-top:8px">
PDF includes all tables (NSF WQI, CCME WQI, IFS R/X/T, distances, alerts) plus 5 embedded charts.
CSV includes all computed tables combined.
</div>""")

                        # ── Agent Logs  (official only) ────────────────────
                        with gr.Tab("Agent Logs") as tab_logs:
                            with gr.Group(visible=True) as logs_locked:
                                gr.HTML('''<div style="padding:16px;text-align:center;background:#FEF3C7;
                                    border-radius:8px;border:1px solid #FCD34D">
                                    <b style="color:#92400E">🔒 Officials only</b>
                                    </div>''')
                            with gr.Group(visible=False) as logs_panel:
                                gr.HTML('<div class="sec-hd">Live Agent Communication Log</div>')
                                gr.HTML('<div style="font-size:12px;color:#475569;margin-bottom:8px">All inter-agent messages logged in real time. Also written to <code>/tmp/yamuna_agents.log</code>.</div>')
                                log_refresh = gr.Button("Refresh Logs", size="sm")
                                log_area    = gr.Textbox(label="", lines=25, interactive=False,
                                                         elem_id="log-area",
                                                         placeholder="Logs appear here after running the pipeline...")
                                log_stats   = gr.HTML()

                        # ── Settings  (official only) ──────────────────────
                        with gr.Tab("Settings") as tab_settings:
                            with gr.Group(visible=True) as settings_locked:
                                gr.HTML('''<div style="padding:16px;text-align:center;background:#FEF3C7;border-radius:8px;border:1px solid #FCD34D"><b style="color:#92400E">🔒 Officials only</b></div>''')
                            with gr.Group(visible=False) as settings_panel:
                                gr.HTML('<div class="sec-hd">LLM API Keys</div>')
                                gr.HTML('<div style="font-size:12px;color:#64748B;margin-bottom:8px">Keys stored in memory only. <a href="https://console.groq.com" target="_blank" style="color:#0369A1">Get Groq key (free)</a>  |  <a href="https://aistudio.google.com" target="_blank" style="color:#0369A1">Get Gemini key (free)</a></div>')
                                groq_key_in   = gr.Textbox(label="Groq API Key",   type="password", placeholder="gsk_...")
                                gemini_key_in = gr.Textbox(label="Gemini API Key", type="password", placeholder="AIza...")
                                save_keys_btn = gr.Button("Save & Activate Keys", size="sm", variant="primary")
                                keys_status   = gr.HTML()

                                gr.HTML('<div class="sec-hd" style="margin-top:20px">Email Alert Settings (SMTP)</div>')
                                gr.HTML('<div style="font-size:12px;color:#64748B;margin-bottom:8px">When CRITICAL alerts are detected, emails are automatically sent to ALL registered official accounts.</div>')
                                with gr.Row():
                                    smtp_host = gr.Textbox(label="SMTP Host", value="smtp.gmail.com")
                                    smtp_port = gr.Textbox(label="Port",      value="465")
                                smtp_user  = gr.Textbox(label="Gmail Address")
                                smtp_pass  = gr.Textbox(label="App Password", type="password")
                                smtp_btn   = gr.Button("Save SMTP & Test", size="sm")
                                smtp_status = gr.HTML()

                                gr.HTML('<div class="sec-hd" style="margin-top:20px">Manage Official Alert Emails</div>')
                                gr.HTML('<div style="font-size:12px;color:#64748B;margin-bottom:8px">All emails listed here receive CRITICAL alerts automatically when analysis is run.</div>')
                                email_list_html  = gr.HTML()
                                email_refresh_btn = gr.Button("Refresh Email List", size="sm")
                                gr.HTML('<div style="font-size:12px;color:#0369A1;margin-top:8px;margin-bottom:4px">Add email to existing user:</div>')
                                with gr.Row():
                                    email_target_user  = gr.Textbox(label="Username", placeholder="official")
                                    email_new_addr     = gr.Textbox(label="Email Address", placeholder="officer@yamuna.gov.in")
                                add_email_btn   = gr.Button("Update Email", size="sm")
                                email_upd_status = gr.HTML()

                                gr.HTML('<div class="sec-hd" style="margin-top:20px">Add New User</div>')
                                with gr.Row():
                                    new_user  = gr.Textbox(label="Username")
                                    new_pass  = gr.Textbox(label="Password", type="password")
                                with gr.Row():
                                    new_role  = gr.Dropdown(choices=["official","viewer"], value="viewer", label="Role")
                                    new_name  = gr.Textbox(label="Full Name")
                                new_email = gr.Textbox(label="Email (for alerts)")
                                add_user_btn = gr.Button("Add User", size="sm")
                                add_user_status = gr.HTML()

                # ── RIGHT: persistent AI chat ─────────────────────────────
                with gr.Column(scale=1, min_width=320, elem_id="chat-sidebar"):
                    gr.HTML("""<div id="chat-header">
  🤖 <span>Water Quality Assistant</span>
</div>
<div style="font-size:11px;color:#64748B;margin-bottom:8px">
  Ask anything about the analysis, IFS theory, or CCME/NSF WQI.
  Run the pipeline first for data-specific answers.
</div>""")
                    _cb_params = inspect.signature(gr.Chatbot.__init__).parameters
                    _cb_kwargs = {
                        "height": 520,
                        "show_label": False,
                        "placeholder": "Run analysis, then ask: *Which location is worst?*",
                    }
                    if "type" in _cb_params:
                        _cb_kwargs["type"] = "messages"
                    if "bubble_full_width" in _cb_params:
                        _cb_kwargs["bubble_full_width"] = False
                    chatbot_widget = gr.Chatbot(**_cb_kwargs)
                    chat_input = gr.Textbox(
                        placeholder="Ask a question... (Enter to send)",
                        show_label=False,
                        lines=2,
                        max_lines=4,
                    )
                    with gr.Row():
                        clear_chat = gr.Button("Clear", size="sm")
                        send_btn   = gr.Button("Send ↑", variant="primary", size="sm", elem_id="send-btn")

                    gr.HTML("""<div style="margin-top:10px;font-size:11px;color:#94A3B8">
Suggested questions:<br>
• Which location is most polluted?<br>
• Explain CCME WQI vs NSF WQI<br>
• What does π hesitation mean?<br>
• What actions for critical alerts?<br>
• Compare Palla and Okhla data
</div>""")

        # ══════════════════════════════════════════════════════════════════
        #  EVENT HANDLERS
        # ══════════════════════════════════════════════════════════════════

        # ── login ──────────────────────────────────────────────────────────
        def handle_login(username, password):
            global _SESSION
            session, msg = auth_agent.login(username, password)
            _SESSION = session
            if session is None:
                err_html = (f'<div style="color:#DC2626;padding:8px;background:#FEE2E2;'
                            f'border-radius:8px;font-size:13px;margin-top:8px">{msg}</div>')
                return (session, err_html,
                        gr.update(visible=True),  gr.update(visible=False),
                        gr.update(value=""),       gr.update(value=""),
                        gr.update(visible=True),   gr.update(visible=False),
                        gr.update(visible=True),   gr.update(visible=False),
                        gr.update(visible=True),   gr.update(visible=False))

            is_off = session.role == "official"
            hdr = (f'<div id="app-strip">'
                   f'<div><h2>Multi-Agent AI System  v3.0</h2>'
                   f'<p>River Water Quality Platform</p></div>'
                   f'<div style="color:rgba(255,255,255,0.7);font-size:12px">'
                   f'{session.name}  |  {session.role.upper()}</div></div>')
            cls   = "role-off" if is_off else "role-vw"
            badge = "badge-off" if is_off else "badge-vw"
            perms = "Full access: upload, run, configure, export, manage alerts" if is_off else \
                    "Read-only access: view analysis, IFS, CCME, alerts, chat"
            banner = (f'<div id="role-banner" class="{cls}">'
                      f'<span class="role-badge {badge}">{"Official" if is_off else "Viewer"}</span>'
                      f'<span style="font-size:12px;color:#475569">{session.name}  —  {perms}</span>'
                      f'</div>')
            login_ok_html = (f'<div style="color:#16A34A;padding:8px;background:#F0FDF4;'
                             f'border-radius:8px;font-size:13px;margin-top:8px">{msg}</div>')
            return (session, login_ok_html,
                    gr.update(visible=False),   # login_screen
                    gr.update(visible=True),    # main_app
                    gr.update(value=hdr),       # app_header
                    gr.update(value=banner),    # role_banner
                    gr.update(visible=not is_off), gr.update(visible=is_off),   # data locked/panel
                    gr.update(visible=not is_off), gr.update(visible=is_off),   # logs locked/panel
                    gr.update(visible=not is_off), gr.update(visible=is_off),   # settings locked/panel
                    )

        _LOGIN_OUTPUTS = [session_st, login_msg, login_screen, main_app,
                          app_header, role_banner,
                          data_locked, data_panel,
                          logs_locked, logs_panel,
                          settings_locked, settings_panel]
        login_btn.click(handle_login,   inputs=[login_user, login_pass], outputs=_LOGIN_OUTPUTS)
        login_pass.submit(handle_login, inputs=[login_user, login_pass], outputs=_LOGIN_OUTPUTS)

        # ── logout ────────────────────────────────────────────────────────
        def handle_logout(session):
            global _SESSION
            _SESSION = None
            LOGGER.info("AuthAgent", f"User {getattr(session,'username','?')} logged out")
            return (None, "",
                    gr.update(visible=True),   # login_screen
                    gr.update(visible=False),  # main_app
                    gr.update(value=""),       # app_header
                    gr.update(value=""),       # role_banner
                    gr.update(visible=True),   gr.update(visible=False),  # data locked/panel
                    gr.update(visible=True),   gr.update(visible=False),  # logs locked/panel
                    gr.update(visible=True),   gr.update(visible=False),  # settings locked/panel
                    )
        logout_btn.click(handle_logout, inputs=[session_st], outputs=_LOGIN_OUTPUTS)

        # ── weight sum ────────────────────────────────────────────────────
        def upd_wsum(a,b,c,d):
            t = round(a+b+c+d, 4)
            col = "#16A34A" if abs(t-1)<0.01 else "#DC2626"
            return f'<div style="font-size:12px;color:{col}">Weight sum: <b>{t}</b> {"✓" if abs(t-1)<0.01 else "— must equal 1.0"}</div>'
        for sl in [w_do,w_fc,w_ph,w_bod]:
            sl.change(upd_wsum,[w_do,w_fc,w_ph,w_bod],[wsum_html])

        # ── auto-load real DPCC data (no synthetic fallback) ───────────────
        def load_real_for_ui():
            try:
                df_real, path = load_real_dpcc_data(allow_colab_upload=False)
                dates = pd.to_datetime(df_real["Date"], errors="coerce").dropna()
                sd = dates.min().strftime("%Y-%m-%d") if not dates.empty else "2024-09-01"
                ed = dates.max().strftime("%Y-%m-%d") if not dates.empty else "2025-08-31"
                return path, sd, ed, f'<div style="color:#16A34A;font-size:12px">✅ Loaded real DPCC dataset: <code>{os.path.basename(path)}</code> ({len(df_real)} rows, {sd} → {ed})</div>'
            except FileNotFoundError as e:
                return None, "2024-09-01", "2025-08-31", f'<div style="color:#DC2626;font-size:12px">⚠️ {e}</div>'
        sample_btn.click(load_real_for_ui, outputs=[file_input, start_date, end_date, run_status])

        # ── MAIN PIPELINE RUN ──────────────────────────────────────────────
        def run_analysis(session, file_obj, sd, ed, locs, wd, wf, wp, wb):
            global _STATE
            if session is None:
                return ([gr.update()]*29
                        + ['<div style="color:#DC2626">Please login first.</div>', ""])
            if session.role != "official":
                return ([gr.update()]*29
                        + ['<div style="color:#DC2626">Officials only can run the pipeline.</div>', ""])
            if file_obj is None:
                return ([gr.update()]*29
                        + ['<div style="color:#D97706">Upload a file or click "Use Sample Data".</div>', ""])

            wt = wd+wf+wp+wb
            weights = {"DO":round(wd/wt,4),"FC":round(wf/wt,4),"pH":round(wp/wt,4),"BOD":round(wb/wt,4)}
            fp = file_obj if isinstance(file_obj, str) else file_obj.name
            try:
                df = pd.read_excel(fp) if fp.endswith(".xlsx") else pd.read_csv(fp)
            except Exception as e:
                return ([gr.update()]*29
                        + [f'<div style="color:#DC2626">File error: {e}</div>', ""])

            config = AnalysisConfig(
                start_date=sd, end_date=ed,
                locations=list(locs) if locs else LOCATION_ORDER,
                weights=weights,
            )
            official_emails = auth_agent.all_official_emails()

            # Run pipeline
            state = coordinator.run_pipeline(df, config, session, official_emails)
            _STATE = state

            # Feed LLM context
            llm_assistant.set_context(state, MEMORY)

            if not state.ok():
                errs = "<br>".join(state.errors)
                return ([gr.update()]*29
                        + [f'<div style="color:#DC2626">Pipeline errors:<br>{errs}</div>', ""])

            viz  = coordinator.viz_agent
            role = session.role

            # ── Relation R ─────────────────────────────────────────────────
            r_disp = build_ifs_display(state.ifs_r, "Season")
            r_heat = viz.heatmap_R(state.ifs_r)
            best_s = state.ifs_r.set_index("Season")[[f"mu_{p}" for p in PARAMS]].mean(axis=1).idxmax()
            r_int  = f'<div class="interp"><b>{best_s}</b> — highest average membership. Best seasonal water quality.</div>'

            # ── Relation X ─────────────────────────────────────────────────
            x_disp = build_ifs_display(state.ifs_x, "Location")
            x_rdr  = viz.radar(state.ifs_x)
            fc_avg = state.ifs_x["mu_FC"].mean()
            x_int  = f'<div class="interp interp-warn">Fecal Coliform avg membership: <b>{fc_avg:.3f}</b> — consistently the worst parameter across all locations.</div>'

            # ── Relation T ─────────────────────────────────────────────────
            t_disp = state.ifs_t[["Season","Location","Mu_T","Nu_T","Pi_T","E_T","IFS"]]
            t_heat = viz.heatmap_T(state.ifs_t)
            best_l = state.ifs_t.groupby("Location")["E_T"].mean().idxmax()
            t_int  = f'<div class="interp"><b>{best_l}</b> — highest μ_T in composition. Confirmed cleanest location.</div>'

            # ── IFS scatter ────────────────────────────────────────────────
            scatter = viz.ifs_scatter(state.ifs_t)

            # ── Distance ───────────────────────────────────────────────────
            d_chart = viz.lollipop(state.dist_df)
            worst   = state.dist_df.sort_values("Hamming",ascending=False).iloc[0]["Location"]
            d_int   = f'<div class="interp interp-crit">Most polluted: <b>{worst}</b> — largest IFS distance from Palla.</div>'

            # ── CCME ───────────────────────────────────────────────────────
            ccme_disp = state.ccme_df[["Location","Season","CCME_WQI","CCME_Class",
                                       "F1_Scope","F2_Frequency","F3_Amplitude",
                                       "Failed_Variables","Failed_Tests","Total_Tests"]]
            ccme_hm   = viz.ccme_heatmap(state.ccme_df)
            ccme_bar_ = viz.ccme_bar(state.ccme_df)
            ccme_fc   = viz.f_components_bar(state.ccme_df)
            ccme_cmp  = viz.wqi_compare(state.wqi_summary, state.ccme_df)
            ccme_int  = ccme_interp_html(state.ccme_df)

            # ── Alerts ─────────────────────────────────────────────────────
            a_html = coordinator.alert_agent.alert_html(state.decisions, role)
            a_df   = coordinator.alert_agent.to_dataframe(state.decisions, role)

            # ── Dashboard ──────────────────────────────────────────────────
            dk = kpi_html(state, role)
            da = coordinator.alert_agent.alert_html(state.decisions[:4], role)
            dt = viz.wqi_trend(state.wqi_df) if state.wqi_df is not None else go.Figure()
            dc = viz.ccme_bar(state.ccme_df)
            dm = viz.location_map(state.wqi_summary, state.ccme_df)

            # ── Log snapshot ───────────────────────────────────────────────
            log_text = LOGGER.get_log_text(60)
            n_msgs   = len(BUS.get_history())
            log_stats_html = (f'<div style="font-size:12px;color:#475569;margin-top:6px">'
                              f'Total messages: <b>{n_msgs}</b> | '
                              f'Memory runs: <b>{MEMORY.count()}</b> | '
                              f'Log file: <code>{LOG_FILE}</code></div>')

            # ── Validation ─────────────────────────────────────────────────
            v_html = val_html(state.validation_log)
            status = (f'<div style="color:#16A34A;padding:8px;background:#F0FDF4;'
                      f'border-radius:6px;font-size:13px">'
                      f'✅ Analysis complete — {state.validation_log.get("final_rows",0)} rows | '
                      f'Run ID: {state.run_id}</div>')

            return [
                # Dashboard (6)
                dk, da, dm, da, dt, dc,
                # Relation R (3)
                r_disp, r_heat, r_int,
                # Relation X (3)
                x_disp, x_rdr, x_int,
                # Relation T (3)
                t_disp, t_heat, t_int,
                # IFS scatter (1)
                scatter,
                # Distance (3)
                state.dist_df.reset_index(), d_int, d_chart,
                # CCME (6)
                ccme_disp, ccme_hm, ccme_bar_, ccme_fc, ccme_cmp, ccme_int,
                # Alerts (2)
                a_html, a_df,
                # Logs (2)
                log_text, log_stats_html,
                # Status + validation (2)
                status, v_html,
            ]

        run_btn.click(
            run_analysis,
            inputs=[session_st, file_input, start_date, end_date, loc_check,
                    w_do, w_fc, w_ph, w_bod],
            outputs=[
                dash_kpi, dash_alerts, dash_map, dash_alert_detail, dash_trend, dash_ccme_bar,
                r_table, r_heatmap, r_interp,
                x_table, x_radar,   x_interp,
                t_table, t_heatmap, t_interp,
                ifs_scatter,
                dist_table, dist_interp, dist_chart,
                ccme_table, ccme_hmap, ccme_bar_plt, ccme_f_bar, ccme_compare, ccme_interp_out,
                alert_html_out, alert_df_out,
                log_area, log_stats,
                run_status, data_val,
            ],
        )

        # ── chat send ─────────────────────────────────────────────────────
        def handle_chat(message, history):
            if not message.strip():
                return history, ""
            new_hist, _ = llm_assistant.chat(message, history)
            return new_hist, ""

        send_btn.click(handle_chat, [chat_input, chat_hist], [chatbot_widget, chat_input])
        chat_input.submit(handle_chat, [chat_input, chat_hist], [chatbot_widget, chat_input])
        clear_chat.click(lambda: ([], ""), outputs=[chatbot_widget, chat_input])

        # Sync chat history state
        chatbot_widget.change(lambda h: h, inputs=[chatbot_widget], outputs=[chat_hist])

        # ── log refresh ───────────────────────────────────────────────────
        def refresh_logs():
            text  = LOGGER.get_log_text(80)
            n     = len(BUS.get_history())
            stats = (f'<div style="font-size:12px;color:#475569;margin-top:6px">'
                     f'Messages: <b>{n}</b> | Memory runs: <b>{MEMORY.count()}</b> | '
                     f'Log file: <code>{LOG_FILE}</code></div>')
            return text, stats
        log_refresh.click(refresh_logs, outputs=[log_area, log_stats])

        # ── save API keys ─────────────────────────────────────────────────
        def save_keys(gk, mk, session):
            if session is None or session.role != "official":
                return '<div style="color:#DC2626;font-size:12px">Officials only</div>'
            llm_assistant.init(gk, mk)
            coordinator.decision_agent.init_llm(gk, mk)
            parts = (["Groq key active"] if gk else []) + (["Gemini key active"] if mk else [])
            msg   = " | ".join(parts) if parts else "No keys provided"
            col   = "#16A34A" if parts else "#D97706"
            LOGGER.info("Settings", f"LLM keys updated by {session.username}")
            return f'<div style="color:{col};font-size:12px;padding:6px">{msg}</div>'
        save_keys_btn.click(save_keys, [groq_key_in, gemini_key_in, session_st], [keys_status])

        # ── SMTP save & test ──────────────────────────────────────────────
        def save_smtp(h, p, u, pw, session):
            if session is None or session.role != "official":
                return '<div style="color:#DC2626;font-size:12px">Officials only</div>'
            coordinator.alert_agent.smtp = {"host":h,"port":p,"user":u,"password":pw}
            ok, msg = coordinator.alert_agent._send_one(
                [{"level":"INFO","location":"Test","param":"System","wqi":0,"ccme_wqi":0,
                  "message":"SMTP test from Yamuna Platform v3.0","action":"None",
                  "ai_explanation":"","likely_cause":"",
                  "timestamp":datetime.now().isoformat(),"show_viewer":True}],
                u)
            col = "#16A34A" if ok else "#DC2626"
            LOGGER.info("Settings", f"SMTP test: {msg}")
            return f'<div style="color:{col};font-size:12px;padding:6px">SMTP test: {msg}</div>'
        smtp_btn.click(save_smtp, [smtp_host, smtp_port, smtp_user, smtp_pass, session_st], [smtp_status])

        # ── add user ──────────────────────────────────────────────────────
        def add_user_fn(uname, upass, role, name, email, session):
            if session is None or session.role != "official":
                return '<div style="color:#DC2626;font-size:12px">Officials only</div>'
            if not uname or not upass:
                return '<div style="color:#D97706;font-size:12px">Username and password required</div>'
            auth_agent.add_user(uname, upass, role, name, email)
            return f'<div style="color:#16A34A;font-size:12px">User <b>{uname}</b> ({role}) added successfully</div>'
        add_user_btn.click(add_user_fn,
            [new_user, new_pass, new_role, new_name, new_email, session_st],
            [add_user_status])

        # ── email list & update ────────────────────────────────────────────
        def get_email_list_html():
            rows = []
            for u in auth_agent._users.values():
                role_col = "#16A34A" if u.get("role") == "official" else "#2563EB"
                email_val = u.get("email", "") or '<span style="color:#94A3B8;font-style:italic">not set</span>'
                rows.append(
                    f'<tr>'
                    f'<td style="padding:5px 10px;font-weight:600">{u["username"]}</td>'
                    f'<td style="padding:5px 10px"><span style="background:{role_col}20;color:{role_col};'
                    f'padding:2px 8px;border-radius:4px;font-size:11px">{u.get("role","")}</span></td>'
                    f'<td style="padding:5px 10px;font-size:12px">{u.get("name","")}</td>'
                    f'<td style="padding:5px 10px;font-family:monospace;font-size:12px">{email_val}</td>'
                    f'</tr>'
                )
            rows_html = "\n".join(rows)
            officials = auth_agent.all_official_emails()
            return (
                f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
                f'<thead><tr style="background:#F1F5F9;font-size:11px;text-transform:uppercase;color:#64748B">'
                f'<th style="padding:6px 10px;text-align:left">Username</th>'
                f'<th style="padding:6px 10px;text-align:left">Role</th>'
                f'<th style="padding:6px 10px;text-align:left">Name</th>'
                f'<th style="padding:6px 10px;text-align:left">Alert Email</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table>'
                f'<div style="font-size:11px;color:#0369A1;margin-top:6px">'
                f'🚨 Critical alerts sent to: <b>{", ".join(officials) or "none configured"}</b></div>'
            )

        def refresh_email_list(session):
            if session is None or session.role != "official":
                return '<div style="color:#DC2626;font-size:12px">Officials only</div>'
            return get_email_list_html()

        def update_user_email(uname, email, session):
            if session is None or session.role != "official":
                return '<div style="color:#DC2626;font-size:12px">Officials only</div>', ""
            uname = uname.strip()
            if uname not in auth_agent._users:
                return f'<div style="color:#DC2626;font-size:12px">User "{uname}" not found</div>', ""
            auth_agent._users[uname]["email"] = email.strip()
            LOGGER.info("Settings", f"Email updated for {uname}: {email}")
            return (f'<div style="color:#16A34A;font-size:12px">Email for <b>{uname}</b> updated to <b>{email}</b></div>',
                    get_email_list_html())

        email_refresh_btn.click(refresh_email_list, [session_st], [email_list_html])
        add_email_btn.click(update_user_email,
            [email_target_user, email_new_addr, session_st],
            [email_upd_status, email_list_html])

        # ── show default map on dashboard immediately at startup ───────────
        app.load(
            fn=lambda: coordinator.viz_agent.location_map(),
            inputs=None,
            outputs=[dash_map],
        )

        # ── PDF export ────────────────────────────────────────────────────
        def do_pdf(session):
            global _STATE
            if session is None: return None,'<div style="color:#DC2626;font-size:12px">Login required</div>'
            if _STATE  is None: return None,'<div style="color:#D97706;font-size:12px">Run analysis first</div>'
            path = coordinator.report_agent.export_pdf(_STATE)
            return path, f'<div style="color:#16A34A;font-size:12px">PDF ready: {os.path.basename(path)}</div>'
        pdf_btn.click(do_pdf, [session_st], [pdf_file, rep_status])

        # ── CSV export ────────────────────────────────────────────────────
        def do_csv(session):
            global _STATE
            if session is None: return None,'<div style="color:#DC2626;font-size:12px">Login required</div>'
            if _STATE  is None: return None,'<div style="color:#D97706;font-size:12px">Run analysis first</div>'
            path = coordinator.report_agent.export_csv(_STATE)
            return path, f'<div style="color:#16A34A;font-size:12px">CSV ready: {os.path.basename(path)}</div>'
        csv_btn.click(do_csv, [session_st], [csv_file, rep_status])

    return app


# ─── LAUNCH ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Gradio UI + MCP Server Launch ─────────────────────────────────────────────
    # Defaults to localhost (127.0.0.1) and share=False; set env vars only if remote access is needed.
    import socket, threading

    MCP_HOST     = os.environ.get("MCP_HOST", "127.0.0.1")
    MCP_PORT     = int(os.environ.get("MCP_PORT", "8070"))
    _in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    _default_share = "true" if _in_colab else "false"
    GRADIO_SHARE = os.environ.get("GRADIO_SHARE", _default_share).lower() == "true"

    def _port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def run_mcp():
        try:
            mcp_server.run(transport="sse", host=MCP_HOST, port=MCP_PORT)
        except Exception as e:
            print(f"MCP Server Error: {e}")

    if _port_in_use(MCP_HOST, MCP_PORT):
        print(f"ℹ️ MCP Server already running on {MCP_HOST}:{MCP_PORT} — skipping duplicate bind.")
    else:
        threading.Thread(target=run_mcp, daemon=True).start()
    app = build_app()
    import inspect
    _launch_params = inspect.signature(app.launch).parameters
    _launch_kwargs = {
        "share": GRADIO_SHARE,
        "server_port": 7870,
        "show_error": True,
        "quiet": False,
    }
    if "css" in _launch_params:
        _launch_kwargs["css"] = CSS
    if "theme" in _launch_params:
        _launch_kwargs["theme"] = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate")
    app.launch(**_launch_kwargs)
