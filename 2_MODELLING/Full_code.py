# === Amaya & Filbien (2015) — ECB Text Similarity Replication (Integrated 1999–2023) ===
# Includes:
#  - Robust SEs (White/HAC) [HAC default]
#  - Statement-frequency ΔMRO construction
#  - Winsorized |CAR| option
#  - Standardization (z-scores) + VIF diagnostics
#  - President & crisis/pandemic dummies
#  - Split-sample outputs (1999–2013 vs 2014–2023)
#  - Event-window robustness (±1/±3/±5/±7)
#  - Optional MRO surprise if surprise_mro.csv is present
#
# Required local files (same names you used):
#   - final_df.csv                (must include 'date' + 'content')
#   - Loughran-McDonald_MasterDictionary_1993-2021.csv
#   - MSCI EURO.xlsx              (Date, Price)
#   - Inflation & interest.xlsx   (Date monthly; Inflation; optional MRO)
#   - Q GDP LEVEL.xlsx            (quarterly GDP level, e.g., 'Q1/2005' or 2005Q1)
#   - ecb data.csv                (fallback for MRO/STR if not in the inflation file)
#   - surprise_mro.csv            (OPTIONAL: Date, MRO_surprise)

import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas import Timestamp
from datetime import datetime
from itertools import tee

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------ Config ------------------
# Sample window (configurable)
START_DATE = pd.Timestamp("1999-01-01")
END_DATE   = pd.Timestamp("2023-12-31")  # or None for max available

# Errors: HAC (Newey–West) by default
COV_TYPE  = "hac"   # "hac" (NW), "white"(HC0), "hc1","hc2","hc3", or "none"
HAC_LAGS  = 6       # Newey–West lags (try 4–8)

# Event-study windows to compute
EVENT_WINDOWS = [1, 3, 5, 7]     # ±days around announcement

# Winsorization level for |CAR|
WINSOR_P = 0.01                  # 1%

# Optional WLS (off by default)
USE_WLS   = False
WLS_SCHEME = None                # e.g., "inv_time" to downweight early obs

# Rescale heuristics (auto-detect if series look like decimals)
AUTO_RESCALE_RATES = True  # DeltaMRO and Inflation to percent if they look like decimals

# -------------------------------------------

# ---- NLTK assets ----
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

EN_STOPS = set(stopwords.words("english"))
STEM = PorterStemmer()

# ======================= Helper Functions =======================

def extract_introductory_statement(text: str) -> str:
    if not isinstance(text, str):
        return ""
    txt = text.replace("Jump to the transcript", "")
    end_markers = [
        "We are now ready to take your questions",
        "We are now at your disposal for questions",
        "We are now at your disposal, should you have any questions",
        "Transcript of the questions asked and the answers given by",
        "Q&A", "CONTACT", "Related topics", "You may also be interested"
    ]
    start_idx = 0
    for s in ["Introductory statement", "Introductory Statement"]:
        ix = txt.find(s)
        if ix != -1:
            start_idx = ix
            break
    cut = len(txt)
    for m in end_markers:
        ix = txt.find(m)
        if ix != -1:
            cut = min(cut, ix)
    return txt[start_idx:cut].strip()

def tokenize_clean_plain(text: str):
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    toks = word_tokenize(t)
    return [w for w in toks if w.isalpha() and w not in EN_STOPS]

def tokenize_clean_stem(text: str):
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    toks = word_tokenize(t)
    toks = [w for w in toks if w.isalpha() and w not in EN_STOPS]
    toks = [STEM.stem(w) for w in toks]
    return toks

def make_bigrams(tokens):
    a, b = tee(tokens); next(b, None)
    return set(zip(a, b))

def jaccard_bigrams(b1, b2):
    inter = b1.intersection(b2); uni = b1.union(b2)
    return (len(inter) / len(uni)) if len(uni) > 0 else np.nan

def pessimism_from_tokens(tokens_plain, neg_set, pos_set):
    neg = sum(1 for t in tokens_plain if t in neg_set)
    pos = sum(1 for t in tokens_plain if t in pos_set)
    total = len(tokens_plain)
    return np.nan if total == 0 else (neg - pos) / total

def nearest_trading_date(d, trading_index):
    if d in trading_index: return d
    future = trading_index[trading_index >= d]
    if len(future) > 0: return future[0]
    past = trading_index[trading_index <= d]
    return past[-1] if len(past) > 0 else None

def _to_quarter_start(dt: pd.Timestamp) -> pd.Timestamp:
    qm = 3 * ((dt.month - 1) // 3) + 1
    return Timestamp(dt.year, qm, 1)

def parse_quarter_any(s) -> pd.Timestamp:
    """Accepts 'Q1/2005', '2005/Q1', '2005Q1', 'Q1 2005', or month-year like '09/2023', '2023-09'."""
    if pd.isna(s): return pd.NaT
    t = str(s).strip()
    # Month-year forms
    for fmt, pat in [("%m/%Y", r"^\d{1,2}/\d{4}$"), ("%Y-%m", r"^\d{4}-\d{1,2}$"), ("%Y/%m", r"^\d{4}/\d{1,2}$")]:
        if re.match(pat, t):
            dt = pd.to_datetime(t, format=fmt); return _to_quarter_start(dt)
    # Generic date-like
    try:
        dt = pd.to_datetime(t, errors="raise")
        return _to_quarter_start(dt)
    except Exception:
        pass
    # Excel serial
    if isinstance(s, (int, float)) and not pd.isna(s):
        try:
            dt = pd.to_datetime(s, unit="D", origin="1899-12-30")
            return _to_quarter_start(dt)
        except Exception:
            pass
    # Quarter-labeled
    u = t.upper().replace("\\", "/").replace("-", "/").replace("_", "/").replace(" ", "/")
    m = (re.search(r'(?:Q)?([1-4])/([12]\d{3})', u) or
         re.search(r'([12]\d{3})/(?:Q)?([1-4])', u) or
         re.search(r'([12]\d{3})Q([1-4])', u) or
         re.search(r'Q([1-4])([12]\d{3})', u))
    if m:
        g1, g2 = m.group(1), m.group(2)
        if len(g1) == 4: year, q = int(g1), int(g2)
        elif len(g2) == 4: q, year = int(g1), int(g2)
        else: q, year = int(g1), int(g2)
        return Timestamp(year=year, month=(3 * (q - 1) + 1), day=1)
    raise ValueError(f"Unrecognized quarter format: {s}")

def zscore(df, cols):
    Z = df[cols].astype(float).copy()
    Z[:] = StandardScaler().fit_transform(Z)
    return Z

def vif_report(dfX):
    X = sm.add_constant(dfX.astype(float))
    v = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns, name="VIF")
    return v.round(2)

def fit_reg(dep, exog, add_const=True, cov_type=COV_TYPE, hac_lags=HAC_LAGS, weights=None, label=""):
    X = exog.copy()
    if add_const:
        X = sm.add_constant(X)
    model = sm.WLS(dep, X, weights=weights, missing="drop") if (USE_WLS and weights is not None) else sm.OLS(dep, X, missing="drop")
    cov_kwds = {}
    ct = None
    if cov_type is None or cov_type.lower() == "none":
        res = model.fit()
    else:
        ct_map = {"white":"HC0","hc0":"HC0","hc1":"HC1","hc2":"HC2","hc3":"HC3","hac":"HAC"}
        ct = ct_map.get(cov_type.lower(), "HC0")
        if ct == "HAC":
            cov_kwds["maxlags"] = hac_lags
            cov_kwds["use_correction"] = True
        res = model.fit(cov_type=ct, cov_kwds=cov_kwds)
    print(f"\n=== {label} ===")
    if ct:
        print(f"[cov_type = {ct}{('; maxlags='+str(hac_lags)) if ct=='HAC' else ''}]")
    print("[estimator = WLS]" if (USE_WLS and weights is not None) else "[estimator = OLS]")
    print(res.summary())
    return res

def cars_with_window(px_df, events_idx, car_w=5, est_L=250, est_G=50):
    def _car(ev):
        if ev not in px_df["Date"].values: return np.nan
        idx = px_df.index[px_df["Date"] == ev][0]
        est_start, est_end = idx - est_L, idx - est_G
        if est_start < 0 or est_end <= est_start: return np.nan
        mu = px_df.loc[est_start:est_end, "ret"].mean()
        win_start, win_end = max(0, idx - car_w), min(len(px_df)-1, idx + car_w)
        return (px_df.loc[win_start:win_end, "ret"] - mu).sum()
    out = pd.DataFrame({"Date": events_idx, "CAR": [_car(ev) for ev in events_idx]})
    out["absCAR"] = out["CAR"].abs()
    return out

def winsorize_s(x: pd.Series, p=WINSOR_P):
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lo, hi)

# ======================= 1) Statements =======================

df_stmt = pd.read_csv("final_df.csv")
# Normalize date column
date_cols = [c for c in df_stmt.columns if c.lower() in ("date", "dates")]
if not date_cols:
    raise ValueError("final_df.csv must include a 'date' column.")
content_col = "content" if "content" in df_stmt.columns else None
if content_col is None:
    raise ValueError("final_df.csv must include a 'content' column.")

df_stmt = df_stmt.rename(columns={date_cols[0]: "Date"})
df_stmt["Date"] = pd.to_datetime(df_stmt["Date"])  # let pandas infer formats

# Clip to configured window
if END_DATE is None:
    end_ = df_stmt["Date"].max()
else:
    end_ = END_DATE
mask_period = (df_stmt["Date"] >= START_DATE) & (df_stmt["Date"] <= end_)
df_stmt = (df_stmt.loc[mask_period]
                  .sort_values("Date")
                  .drop_duplicates(subset=["Date"])  # keep one per meeting
                  .reset_index(drop=True))

# Extract text & NLP
df_stmt["intro"]        = df_stmt[content_col].apply(extract_introductory_statement)
df_stmt["tokens_plain"] = df_stmt["intro"].apply(tokenize_clean_plain)  # LM dict
df_stmt["tokens_stem"]  = df_stmt["intro"].apply(tokenize_clean_stem)   # similarity

# ======================= 2) Similarity & Pessimism =======================

lm = pd.read_csv("Loughran-McDonald_MasterDictionary_1993-2021.csv")
neg_set = set(lm[lm["Negative"] != 0]["Word"].str.lower())
pos_set = set(lm[lm["Positive"] != 0]["Word"].str.lower())

df_stmt["pessimism"] = df_stmt["tokens_plain"].apply(lambda toks: pessimism_from_tokens(toks, neg_set, pos_set))
df_stmt["bigrams"]   = df_stmt["tokens_stem"].apply(make_bigrams)

sim = [np.nan] + [jaccard_bigrams(df_stmt.loc[i, "bigrams"], df_stmt.loc[i-1, "bigrams"]) for i in range(1, len(df_stmt))]
df_stmt["similarity"] = sim

first_day = df_stmt["Date"].min()
df_stmt["Time_days"]  = (df_stmt["Date"] - first_day).dt.days.clip(lower=1)
df_stmt["Time_count"] = np.arange(1, len(df_stmt) + 1)

df_stmt["n_tokens"]   = df_stmt["tokens_plain"].apply(len)

EPS = 1e-9
df_stmt["logSimilarity"] = np.log(df_stmt["similarity"].clip(lower=EPS))
df_stmt["logTime_days"]  = np.log(df_stmt["Time_days"])
df_stmt["logTime_count"] = np.log(df_stmt["Time_count"])

# ======================= 3) Controls =======================

# HICP monthly -> daily ffill
hicp = pd.read_excel("Inflation & interest.xlsx")
if "Unnamed: 0" in hicp.columns and "Date" not in hicp.columns:
    hicp = hicp.rename(columns={"Unnamed: 0": "Date"})
if "Date" not in hicp.columns:
    raise ValueError("Inflation & interest.xlsx must include a Date column.")
hicp_all = hicp.copy()
try:
    hicp["Date"] = pd.to_datetime(hicp["Date"], format="%m/%Y")
except Exception:
    hicp["Date"] = pd.to_datetime(hicp["Date"])

infl_col = None
for cand in ["Inflation", "HICP", "HICP YoY", "Inflation rate"]:
    if cand in hicp.columns:
        infl_col = cand; break
if infl_col is None:
    raise ValueError("Inflation column not found in Inflation & interest.xlsx")

hicp = (hicp[["Date", infl_col]].dropna()
        .sort_values("Date").rename(columns={infl_col: "Inflation"}))
hicp_daily = hicp.set_index("Date").resample("D").ffill().reset_index()

# Output gap proxy from quarterly GDP
gdp_q = pd.read_excel("Q GDP LEVEL.xlsx")
gdp_q["Date"] = gdp_q["Date"].apply(parse_quarter_any)

# pick GDP column
gdp_col = None
for cand in ["GDPL", "GDP", "GDP Level", "Real GDP"]:
    if cand in gdp_q.columns:
        gdp_col = cand; break
if gdp_col is None:
    gdp_col = [c for c in gdp_q.columns if c != "Date"][0]

gdp_q = gdp_q[["Date", gdp_col]].rename(columns={gdp_col: "GDPL"}).set_index("Date").sort_index()
gdp_d = gdp_q.resample("D").interpolate()
X_tr = np.arange(len(gdp_d)).reshape(-1, 1)
y_tr = gdp_d["GDPL"].values
lin = LinearRegression().fit(X_tr, y_tr)
pot = lin.predict(X_tr)
ogap_df = pd.DataFrame({"Date": gdp_d.index, "OutputGap": (y_tr - pot)})

# MRO level & statement-frequency DeltaMRO
mro_daily = None
if any(c.lower().startswith("mro") for c in hicp_all.columns):
    mro_cols = [c for c in hicp_all.columns if c.lower().startswith("mro")]
    tmp = hicp_all[["Date", mro_cols[0]]].rename(columns={mro_cols[0]: "MRO"})
    try:
        tmp["Date"] = pd.to_datetime(tmp["Date"], format="%m/%Y")
    except Exception:
        tmp["Date"] = pd.to_datetime(tmp["Date"])
    mro_daily = tmp.set_index("Date").resample("D").ffill().reset_index()
elif os.path.exists("ecb data.csv"):
    ecb = pd.read_csv("ecb data.csv")
    ecb = ecb.rename(columns={c: c.strip() for c in ecb.columns})
    dcol = "DATE" if "DATE" in ecb.columns else ("Date" if "Date" in ecb.columns else None)
    if dcol is None: raise ValueError("ecb data.csv missing DATE/Date")
    ecb["Date"] = pd.to_datetime(ecb[dcol])
    rate_col = None
    for cand in ["MRO","mro","STR","refi","Main refinancing operations"]:
        if cand in ecb.columns:
            rate_col = cand; break
    if rate_col is None:
        num = ecb.select_dtypes(include=[np.number]).columns.tolist()
        if not num: raise ValueError("ecb data.csv has no numeric rate column")
        rate_col = num[0]
    mro_daily = (ecb[["Date", rate_col]].rename(columns={rate_col: "MRO"})
                 .set_index("Date").resample("D").ffill().reset_index())

if mro_daily is None:
    raise ValueError("No MRO found. Add MRO to Inflation & interest.xlsx or provide ecb data.csv")

mro_daily = mro_daily.sort_values("Date").reset_index(drop=True)

# Align MRO to statement dates; DeltaMRO between consecutive statements
stmt_mro = pd.merge_asof(
    df_stmt[["Date"]].sort_values("Date"),
    mro_daily[["Date","MRO"]],
    on="Date", direction="forward"
)
stmt_mro["DeltaMRO"] = stmt_mro["MRO"].diff()
stmt_mro = stmt_mro[["Date","MRO","DeltaMRO"]]

# Optional Surprise MRO
surprise = None
if os.path.exists("surprise_mro.csv"):
    surprise = pd.read_csv("surprise_mro.csv")
    surprise["Date"] = pd.to_datetime(surprise["Date"])
    s_col = "MRO_surprise" if "MRO_surprise" in surprise.columns else [c for c in surprise.columns if c != "Date"][0]
    surprise = surprise[["Date", s_col]].rename(columns={s_col: "SurpriseMRO"})

# ======================= 4) Market data & CAR =======================

px = pd.read_excel("MSCI EURO.xlsx")["Date Price".split()].dropna()
try:
    px["Date"] = pd.to_datetime(px["Date"], format="%d/%m/%Y")
except Exception:
    px["Date"] = pd.to_datetime(px["Date"])
px = px.sort_values("Date").reset_index(drop=True)
px["ret"] = np.log(px["Price"]).diff()
px = px.dropna().reset_index(drop=True)
trading = pd.DatetimeIndex(px["Date"])

# Announcement dates aligned to nearest trading day on/after
events = []
for d in df_stmt["Date"]:
    td = nearest_trading_date(d, trading)
    if td is not None:
        events.append(td)
events = pd.DatetimeIndex(events)

# Compute CAR for all requested windows and store in dict of DataFrames
cars_dict = {}
for W in EVENT_WINDOWS:
    cars_dict[W] = cars_with_window(px, events, car_w=W)

# Use ±5 as the main |CAR|
df_cars_main = cars_dict[5]

# ======================= 5) Merge master statement-level DataFrame =======================

controls = hicp_daily.merge(ogap_df, on="Date", how="outer")
if surprise is not None:
    controls = controls.merge(surprise, on="Date", how="left")
controls = controls.sort_values("Date").ffill()

stmt = (df_stmt[["Date","pessimism","similarity","logSimilarity","logTime_days","logTime_count","n_tokens"]]
        .merge(controls, on="Date", how="left")
        .merge(stmt_mro, on="Date", how="left")
        .merge(df_cars_main[["Date","absCAR"]], on="Date", how="left"))

print(f"Statements in sample: {len(stmt)}; Date range: {stmt['Date'].min().date()} → {stmt['Date'].max().date()}")
print("Var(DeltaMRO) at statement freq:", np.nanvar(stmt["DeltaMRO"]))

# President regimes (approximate handover dates)
stmt["pres_duisenberg"] = (stmt["Date"] <  pd.Timestamp("2003-11-01")).astype(int)
stmt["pres_trichet"]    = ((stmt["Date"] >= pd.Timestamp("2003-11-01")) & (stmt["Date"] < pd.Timestamp("2011-11-01"))).astype(int)
stmt["pres_draghi"]     = ((stmt["Date"] >= pd.Timestamp("2011-11-01")) & (stmt["Date"] < pd.Timestamp("2019-11-01"))).astype(int)
stmt["pres_lagarde"]    = (stmt["Date"] >= pd.Timestamp("2019-11-01")).astype(int)

# Crisis & pandemic dummies
stmt["crisis_08_12"] = ((stmt["Date"] >= pd.Timestamp("2008-09-01")) & (stmt["Date"] <= pd.Timestamp("2012-12-31"))).astype(int)
stmt["pandemic_20_21"] = ((stmt["Date"] >= pd.Timestamp("2020-03-01")) & (stmt["Date"] <= pd.Timestamp("2021-12-31"))).astype(int)

# Optional auto-rescale to percent units for ΔMRO and Inflation
if AUTO_RESCALE_RATES:
    scaled_msgs = []
    if stmt["DeltaMRO"].abs().quantile(0.99) < 0.05:
        stmt["DeltaMRO"] = stmt["DeltaMRO"] * 100.0
        scaled_msgs.append("DeltaMRO×100 (to p.p.)")
    if stmt["Inflation"].abs().quantile(0.99) < 0.2:
        stmt["Inflation"] = stmt["Inflation"] * 100.0
        scaled_msgs.append("Inflation×100 (to %)")
    if scaled_msgs:
        print("Auto-rescaled:", ", ".join(scaled_msgs))

# Add winsorized |CAR|
stmt["absCAR_w"] = winsorize_s(stmt["absCAR"], WINSOR_P)

# Create alternative |CAR| for other windows too (optional; dict)
for W in EVENT_WINDOWS:
    stmt[f"absCAR_w{W}"] = np.nan
    if W in cars_dict:
        tmp = cars_dict[W][["Date","absCAR"]].copy()
        tmp["absCAR"] = winsorize_s(tmp["absCAR"], WINSOR_P)
        stmt = stmt.drop(columns=[f"absCAR_w{W}"], errors="ignore").merge(tmp.rename(columns={"absCAR":f"absCAR_w{W}"}), on="Date", how="left")

# ======================= 6) Regression blocks =======================

def run_table3(stmt_df, label_suffix=""):
    print(f"\n===== TABLE 3 — {label_suffix} =====")
    # Basic sets
    tbl3  = stmt_df.dropna(subset=["logSimilarity","Inflation","OutputGap"]).copy()
    tbl3c = tbl3.dropna(subset=["DeltaMRO"]).copy()

    # (1) logSimilarity ~ logTime_days
    _ = fit_reg(tbl3["logSimilarity"], tbl3[["logTime_days"]], label=f"T3(1){label_suffix}: logSimilarity ~ logTime_days")

    # (2) + controls
    _ = fit_reg(tbl3c["logSimilarity"], tbl3c[["logTime_days","OutputGap","Inflation","DeltaMRO"]], label=f"T3(2){label_suffix}: + controls")

    # (4) logTime_count + controls
    _ = fit_reg(tbl3c["logSimilarity"], tbl3c[["logTime_count","OutputGap","Inflation","DeltaMRO"]], label=f"T3(4){label_suffix}: logTime_count + controls")

    # + Length & FE (omit one president to avoid dummy trap)
    X_len_fe = tbl3c[["logTime_days","OutputGap","Inflation","DeltaMRO","n_tokens","crisis_08_12","pandemic_20_21","pres_trichet","pres_draghi","pres_lagarde"]]
    _ = fit_reg(tbl3c["logSimilarity"], X_len_fe, label=f"T3(2){label_suffix}: + length + crisis/pandemic + president FE")

    # Standardized X versions & VIF
    Xz = zscore(tbl3c, ["logTime_days","OutputGap","Inflation","DeltaMRO","n_tokens"])  # keep length too
    _  = fit_reg(tbl3c["logSimilarity"], Xz, label=f"T3 z-scored X{label_suffix} (incl. length)")
    print("\nVIF — T3(2) controls (+length):")
    print(vif_report(tbl3c[["logTime_days","OutputGap","Inflation","DeltaMRO","n_tokens"]]))


def run_table4(stmt_df, label_suffix=""):
    print(f"\n===== TABLE 4 — {label_suffix} =====")
    # Use winsorized |CAR| at ±5d main window
    tbl4  = stmt_df.dropna(subset=["absCAR_w","pessimism","similarity","Inflation","OutputGap"]).copy()
    tbl4["interaction"] = tbl4["pessimism"] * tbl4["similarity"]
    tbl4c = tbl4.dropna(subset=["DeltaMRO"]).copy()

    # (1) |CAR| ~ Pessimism
    _ = fit_reg(tbl4["absCAR_w"], tbl4[["pessimism"]], label=f"T4(1){label_suffix}: |CAR|_w ~ Pessimism")

    # (3) |CAR| ~ Pessimism×Similarity
    _ = fit_reg(tbl4["absCAR_w"], tbl4[["interaction"]], label=f"T4(3){label_suffix}: |CAR|_w ~ Pessimism×Similarity")

    # (4) + controls
    _ = fit_reg(tbl4c["absCAR_w"], tbl4c[["interaction","OutputGap","Inflation","DeltaMRO"]], label=f"T4(4){label_suffix}: |CAR|_w ~ interaction + controls")

    # z-scored X version
    Xz = zscore(tbl4c, ["interaction","OutputGap","Inflation","DeltaMRO"])  
    _  = fit_reg(tbl4c["absCAR_w"], Xz, label=f"T4(4){label_suffix} z-scored X")

    print("\nVIF — T4(4) controls:")
    print(vif_report(tbl4c[["interaction","OutputGap","Inflation","DeltaMRO"]]))

    # Event-window robustness: ±1, ±3, ±7 (winsorized)
    for W in [w for w in EVENT_WINDOWS if w != 5]:
        col = f"absCAR_w{W}"
        if col in stmt_df.columns:
            t4W = stmt_df.dropna(subset=[col,"pessimism","similarity","Inflation","OutputGap","DeltaMRO"]).copy()
            t4W["interaction"] = t4W["pessimism"] * t4W["similarity"]
            XW = zscore(t4W, ["interaction","OutputGap","Inflation","DeltaMRO"]) 
            print(f"\n— T4(4){label_suffix} with ±{W} day window —")
            _ = fit_reg(t4W[col], XW, label=f"T4(4){label_suffix} ±{W}d z-scored X")

    # Surprise MRO (optional)
    if "SurpriseMRO" in stmt_df.columns and stmt_df["SurpriseMRO"].notna().any():
        t45_df = tbl4c.dropna(subset=["SurpriseMRO"]).copy()
        if len(t45_df) > 0:
            X45 = zscore(t45_df, ["interaction","OutputGap","Inflation","DeltaMRO","SurpriseMRO"]) 
            _ = fit_reg(t45_df["absCAR_w"], X45, label=f"T4(5){label_suffix}: + Surprise MRO (z-scored X)")
        else:
            print(f"[Note] SurpriseMRO present but all NA in this subsample {label_suffix}.")

# ======================= 7) Run — Full sample and splits =======================

# Full sample
run_table3(stmt, "FULL 1999–2023")
run_table4(stmt, "FULL 1999–2023")

# Split samples
pre_mask  = (stmt["Date"] <= pd.Timestamp("2013-12-31"))
post_mask = (stmt["Date"] >= pd.Timestamp("2014-01-01"))

stmt_pre  = stmt.loc[pre_mask].copy()
stmt_post = stmt.loc[post_mask].copy()

run_table3(stmt_pre,  "PRE 1999–2013")
run_table4(stmt_pre,  "PRE 1999–2013")

run_table3(stmt_post, "POST 2014–2023")
run_table4(stmt_post, "POST 2014–2023")

print("\nDone. Integrated replication with HAC errors, winsorized |CAR|, FE, and split samples.")
