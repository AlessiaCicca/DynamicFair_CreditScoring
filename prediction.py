

# -*- coding: utf-8 -*-
"""
Models evaluated (10 total):
    M1     : Logistic Regression (no longitudinal marker, static + time + cats)
    M1-Joint   : Dynamic Logistic (static + age_norm + m_hat_obs + cats)
    M1-LM     : Dynamic Logistic + Landmark one-hot
    M1-LMISO  : Dynamic Logistic + Landmark one-hot + Isotonic Calibration
    M1-TD     : Time-Decay Logistic
    M1-IW     : Importance-Weighted Logistic
    Cox       : Cox PH survival model
    XGB       : XGBoost
    HAT       : Hoeffding Adaptive Tree
    ARF       : Adaptive Random Forest

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore", category=FutureWarning)

SVC_PATH  = Path("/content/Incorporating-data-drift-to-perform-survival-analysis-on-credit-risk/svc.csv")
ORIG_PATH = Path("/content/Incorporating-data-drift-to-perform-survival-analysis-on-credit-risk/orig.csv")
OUT_DIR   = Path("/content/Incorporating-data-drift-to-perform-survival-analysis-on-credit-risk/output")  # FIX: directory reale
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
HORIZON_MONTHS = 12
# LANDMARKS = [12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]  # full dataset
LANDMARKS = [1, 2, 3]  # toy dataset
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
 
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
 
 
def parse_yyyymm(S):
    return pd.to_datetime(S.astype(str).str[:6] + "01",
                          format="%Y%m%d", errors="coerce")
def custom_loss(y_true, y_pred):
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad = p - y_true
    hess = p * (1 - p)
    return grad, hess

# fairness loss con closure
def make_fairness_loss(sensitive, lam=1.0):
    def loss(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        grad_base = p - y_true
        hess_base = p * (1 - p)
        mask0 = (sensitive == 0)
        mask1 = (sensitive == 1)
        n0 = mask0.sum() + 1e-6
        n1 = mask1.sum() + 1e-6
        gap = p[mask1].sum()/n1 - p[mask0].sum()/n0
        dp = np.zeros_like(p)
        dp[mask1] =  2 * gap / n1
        dp[mask0] = -2 * gap / n0
        grad_fair = dp * p * (1 - p)
        hess_fair = dp * (1 - 2*p) * p * (1 - p)
        return grad_base + lam*grad_fair, hess_base + lam*np.abs(hess_fair)
    return loss

def scheduled_balance(orig_upb, r, N, a):
    if any(pd.isna(x) for x in [orig_upb, r, N, a]):
        return np.nan
    try:
        orig_upb = float(orig_upb); r = float(r)
        N = int(float(N)); a = float(a)
    except:
        return np.nan
 
    if r > 0:
        r /= 100.0
    if r < 0 or N <= 0 or N > 1000:
        return np.nan
 
    rm = r / 12.0
    if abs(rm) < 1e-10:
        P = orig_upb / N
        return max(0.0, orig_upb - P * a)
 
    a = np.clip(a, 0, N)
    try:
        num = (1 + rm)**N - (1 + rm)**a
        den = (1 + rm)**N - 1
        if den == 0:
            return np.nan
        return max(0.0, orig_upb * num / den)
    except OverflowError:
        return np.nan
 
 
def compute_bd_pct(cur_upb, sched):
    if pd.isna(cur_upb) or pd.isna(sched) or sched <= 1:
        return np.nan
    return (cur_upb - sched) / sched
 
 
def is_default(x):
    try:
        return 0 if float(x) == 0 else 1
    except:
        return 0 if str(x).strip() in {"", "0", "00"} else 1
 
 
def metrics_all(y_true, p, th=0.5):
    p = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, p)
    y_pred = (p >= th)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(AUC=auc, Brier=brier, F1=f1)
 
 
def agg_mean_sd(list_of_dicts):
    out = {}
    keys = list(list_of_dicts[0].keys())
    for k in keys:
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"]   = float(np.nanstd(vals))
    return out
 
 
def time_decay_weights(landmarks, gamma=0.20):
    Lmax = landmarks.max()
    return np.exp(-gamma * (Lmax - landmarks) / max(HORIZON_MONTHS, 1))
 
 
def importance_weights_shift(Xtr, Xte):
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([Xtr, Xte])
    y = np.hstack([np.zeros(len(Xtr)), np.ones(len(Xte))])
    clf = LogisticRegression(max_iter=3000, solver="newton-cholesky")
    clf.fit(X, y)
    s = clf.predict_proba(Xtr)[:, 1]
    w = s / np.maximum(1 - s, 1e-6)
    cap = 10 * np.median(w)
    return np.clip(w, 1e-3, cap)
 
 
def isotonic_calibrate(p_tr, y_tr, p_te):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    return iso.predict(p_te)
 
 
def row_to_dict(vec):
    return {f"f{i}": float(vec[i]) for i in range(len(vec))}
 
 
svc = pd.read_csv(SVC_PATH)
orig = pd.read_csv(ORIG_PATH)
svc.columns = svc.columns.str.strip()
orig.columns = orig.columns.str.strip()
 
svc["MonRepPer"] = parse_yyyymm(svc["MonRepPer"])
svc = svc.sort_values(["LoanSeqNum", "MonRepPer"])
first_m = svc.groupby("LoanSeqNum")["MonRepPer"].transform("min")
svc["LoanAge"] = ((svc["MonRepPer"] - first_m).dt.days // 30).astype(float)
 
df = svc.merge(orig, on="LoanSeqNum", how="inner")
 
NUM_COLS = [
    "OrigUPB", "OrigLTV", "DTI", "CreditScore",
    "OrigInterestRate", "OrigLoanTerm", "NumBorrowers",
    "CurAct_UPB", "CurIntRate", "ELTV"
]
 
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
 
df["B_sch"] = df.apply(lambda r: scheduled_balance(
    r["OrigUPB"], r["OrigInterestRate"], r["OrigLoanTerm"], r["LoanAge"]), axis=1)
 
df["BD_pct"] = df.apply(lambda r: compute_bd_pct(
    r["CurAct_UPB"], r["B_sch"]), axis=1)
 
df["Default"] = df["CurLoanDel"].apply(is_default)
 
fd_age = df[df["Default"] == 1].groupby("LoanSeqNum")["LoanAge"].min()
df = df.merge(fd_age.rename("FirstDefaultAge"), on="LoanSeqNum", how="left")
 
df["age_norm"] = df["LoanAge"] / df["OrigLoanTerm"]
 
fit = df[["LoanSeqNum", "age_norm", "BD_pct"]].dropna()
fit["xx"] = fit["age_norm"]**2
fit["xy"] = fit["age_norm"] * fit["BD_pct"]
agg = fit.groupby("LoanSeqNum").agg(
    n=("age_norm", "size"),
    sx=("age_norm", "sum"),
    sy=("BD_pct", "sum"),
    sxx=("xx", "sum"),
    sxy=("xy", "sum")).reset_index()
 
den = agg["n"] * agg["sxx"] - agg["sx"]**2 + 1e-3
b1 = np.where(agg["n"] >= 2, (agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]) / den, 0)
b0 = np.where(agg["n"] >= 2, (agg["sy"] - b1 * agg["sx"]) / agg["n"], 0)
coef = pd.DataFrame({"LoanSeqNum": agg["LoanSeqNum"], "b0": b0, "b1": b1})
df = df.merge(coef, on="LoanSeqNum", how="left")
df["m_hat_obs"] = df["b0"] + df["b1"] * df["age_norm"]
 
lm_rows = []
for L in LANDMARKS:
    snap = df[np.isclose(df["LoanAge"], L, atol=0.5)].copy()
    if len(snap) == 0:
        continue
 
    last_age = df.groupby("LoanSeqNum")["LoanAge"].max()
    snap = snap.merge(last_age.rename("LastAge"), on="LoanSeqNum")
 
    tte = np.where(
        snap["FirstDefaultAge"].notna(),
        np.maximum(0, snap["FirstDefaultAge"] - L),
        np.maximum(0, snap["LastAge"] - L))
 
    snap["lm_time"]  = np.minimum(HORIZON_MONTHS, tte)
 
    # FIX: use lm_event as target (excludes defaults already occurred before L)
    snap["lm_event"] = (
        snap["FirstDefaultAge"].notna() &
        (snap["FirstDefaultAge"] - L > 0) &
        (snap["FirstDefaultAge"] - L <= HORIZON_MONTHS)
    ).astype(int)
 
    snap["landmark"] = L
    lm_rows.append(snap)
 
landmark_df = pd.concat(lm_rows, ignore_index=True)
 # gruppo sensibile — adatta alla tua variabile
sensitive = landmark_df["Occupancy"].map({"O": 0, "S": 1, "I": 0}).fillna(0).to_numpy()

 
STATIC_COLS = ["CreditScore", "DTI", "OrigLTV", "OrigInterestRate", "OrigLoanTerm", "NumBorrowers"]
CAT_COLS    = ["Occupancy", "LoanPurpose"]
 
TV_COLS = [
    "CurAct_UPB",
    "CurIntRate",
    "ELTV"
]
 
enc_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cats = enc_cat.fit_transform(landmark_df[CAT_COLS])
cat_names = enc_cat.get_feature_names_out(CAT_COLS)
 
enc_lmk = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
lmk_oh = enc_lmk.fit_transform(landmark_df[["landmark"]])
lmk_names = enc_lmk.get_feature_names_out(["landmark"])
 
tv_filled = landmark_df[TV_COLS].astype(float)
tv_filled = tv_filled.fillna(tv_filled.median())
 
X_lr = np.hstack([
    landmark_df[STATIC_COLS].fillna(landmark_df[STATIC_COLS].median()).to_numpy(),
    tv_filled.to_numpy(),
    landmark_df[["age_norm"]].fillna(0).to_numpy(),
    cats
])
 
X_base = np.hstack([
    landmark_df[STATIC_COLS].fillna(landmark_df[STATIC_COLS].median()).to_numpy(),
    landmark_df[["age_norm"]].fillna(0).to_numpy(),
    landmark_df[["m_hat_obs"]].fillna(0).to_numpy(),
    cats
])
 
X_lm = np.hstack([X_base, lmk_oh])
 
# FIX: use lm_event as target (correct forward-looking definition)
y = landmark_df["lm_event"].to_numpy()
groups = landmark_df["LoanSeqNum"].to_numpy()
 
cox_df = pd.concat([
    landmark_df[["LoanSeqNum", "lm_time", "lm_event", "age_norm", "m_hat_obs"] + STATIC_COLS],
    pd.DataFrame(cats, columns=cat_names),
    pd.DataFrame(lmk_oh, columns=lmk_names)
], axis=1)
 
sel = VarianceThreshold(1e-8)
tmp = cox_df.drop(columns=["LoanSeqNum", "lm_time", "lm_event"])
tmp = tmp.loc[:, sel.fit(tmp).get_support()]
cox_df = pd.concat([cox_df[["LoanSeqNum", "lm_time", "lm_event"]], tmp], axis=1)
 
 
model_names = ["M1-Joint", "LGB-Fair"]

 
metrics_by_model = {m: [] for m in model_names}
times_by_model   = {m: [] for m in model_names}
 
gkf = GroupKFold(n_splits=5)
 
for tr_idx, te_idx in gkf.split(X_base, y, groups):
 
    Xtr_base, Xte_base = X_base[tr_idx], X_base[te_idx]
    ytr, yte           = y[tr_idx],      y[te_idx]
 
    # M1-Joint: Dynamic Logistic (static + age_norm + m_hat_obs + cats)
    t0 = time.perf_counter()
    lr_base = LogisticRegression(max_iter=3000, class_weight="balanced", solver="newton-cholesky")
    lr_base.fit(Xtr_base, ytr)
    p_base = lr_base.predict_proba(Xte_base)[:, 1]
    times_by_model["M1-Joint"].append(time.perf_counter() - t0)
    metrics_by_model["M1-Joint"].append(metrics_all(yte, p_base))

    sensitive_tr = sensitive[tr_idx]

    print(f"Xtr_base shape: {Xtr_base.shape}, ytr shape: {ytr.shape}")
    print(f"Classi nel fold: {np.unique(ytr, return_counts=True)}")

    if Xtr_base.shape[0] == 0 or Xtr_base.shape[1] == 0:
        print("Fold vuoto — skip LGB-Fair")
        continue

    t0 = time.perf_counter()
    lgb_fair = lgb.LGBMClassifier(
        objective=make_fairness_loss(sensitive_tr, lam=1.0),
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1
    )
    lgb_fair.fit(Xtr_base, ytr)
    p_lgb_fair = lgb_fair.predict_proba(Xte_base)[:, 1]
    times_by_model["LGB-Fair"].append(time.perf_counter() - t0)
    metrics_by_model["LGB-Fair"].append(metrics_all(yte, p_lgb_fair))
        
 
rows = []
for m in model_names:
    agg = agg_mean_sd(metrics_by_model[m])
    agg["Model"] = m
    agg["Time_Mean_sec"] = float(np.mean(times_by_model[m]))
    agg["Time_SD_sec"]   = float(np.std(times_by_model[m]))
    rows.append(agg)
 
summary = pd.DataFrame(rows)[[
    "Model",
    "AUC_Mean", "AUC_SD",
    "Brier_Mean", "Brier_SD",
    "F1_Mean", "F1_SD",
    "Time_Mean_sec", "Time_SD_sec"
]]
 
out_path = OUT_DIR / "results.csv"
summary.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
