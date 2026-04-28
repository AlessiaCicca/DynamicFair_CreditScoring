import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time, gc, os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── UNICO PARAMETRO DA CAMBIARE ────────────────────────────────────────────────
DATA_PATH = filepath # <-- sostituisci con il tuo path
OUT_DIR   = Path("/content/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_MONTHS = 12
LANDMARKS      = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]



import wandb

FAIR_ALPHA = 1000
FAIR_ATTR  = "RACE"

wandb.init(
    entity="alessia-ciccaglione02-",
    # Set the wandb project where this run will be logged.
    project="ThesisFairness",
    name    = f"LR_{FAIR_ATTR}_alpha{FAIR_ALPHA}",
    config  = {
        "model_type":     "LogisticRegression",
        "fair_attr":      FAIR_ATTR,
        "fair_alpha":     FAIR_ALPHA,
        "apply_fair":     FAIR_ALPHA > 0,
        "lr":             1e-3,
        "n_epochs":       500,
        "pos_weight_cap": 110.0,
        "n_folds":        5,
        "horizon_months": HORIZON_MONTHS,
        "n_knots":        5,
    }
)


import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons

class EqualiedOddsLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.y_classes = [0, 1]
        self.n_class = len(sensitive_classes)
        self.n_y_class = len(self.y_classes)
        super(EqualiedOddsLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        # K:  number of constraint : (|A| x |Y| x {+, -})
        self.n_constraints = self.n_class * self.n_y_class * 2
        # J : dim of conditions  : ((|A|+1) x |Y|)
        self.dim_condition = self.n_y_class * (self.n_class + 1)
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        # make M (K * J): (|A| x |Y| x {+, -})  *   (|A|+1) x |Y|) )
        self.c = torch.zeros(self.n_constraints)
        element_K_A = self.sensitive_classes + [None]
        for i_a, a_0 in enumerate(self.sensitive_classes):
            for i_y, y_0 in enumerate(self.y_classes):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y_1 in enumerate(self.y_classes):
                        for j_a, a_1 in enumerate(element_K_A):
                            i = i_a * (2 * self.n_y_class) + i_y * 2 + i_s
                            j = j_y + self.n_y_class * j_a
                            self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                idx_true = (y == v) * (sensitive == u)  # torch.bool
                expected_values_list.append(out[idx_true].mean())
        # sensitive is star
        for v in self.y_classes:
            idx_true = y == v
            expected_values_list.append(out[idx_true].mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y):
        return super(EqualiedOddsLoss, self).forward(X, out, sensitive, y=y)


# ── Helpers ────────────────────────────────────────────────────────────────────
def is_default_vec(s):
    num = pd.to_numeric(s, errors="coerce")
    return (num.notna() & (num != 0)).astype(np.int8)

def scheduled_balance(orig_upb, r, N, a):
    """
    Saldo teorico al mese a secondo il piano di ammortamento originale.
    r = tasso annuo (percentuale o decimale), N = mesi totali, a = mese corrente.
    """
    try:
        orig_upb = float(orig_upb); r = float(r)
        N = int(float(N));          a = float(a)
    except:
        return np.nan
    if any(np.isnan(x) for x in [orig_upb, r, N, a]):
        return np.nan
    if r > 1:
        r /= 100.0
    if r < 0 or N <= 0 or N > 1000:
        return np.nan
    rm = r / 12.0
    if abs(rm) < 1e-10:
        return max(0.0, orig_upb - (orig_upb / N) * a)
    a = np.clip(a, 0, N)
    try:
        num = (1 + rm)**N - (1 + rm)**a
        den = (1 + rm)**N - 1
        return max(0.0, orig_upb * num / den) if den != 0 else np.nan
    except OverflowError:
        return np.nan

def compute_bd_pct(cur_upb, sched):
    """
    Balance Deviation %: (saldo_corrente - saldo_teorico) / saldo_teorico.
    Positivo = mutuatario in ritardo rispetto al piano (segnale di rischio).
    Negativo = ha pagato più del previsto.
    """
    if pd.isna(cur_upb) or pd.isna(sched) or sched <= 1:
        return np.nan
    return (cur_upb - sched) / sched

def find_best_threshold(y_true, p, max_th_quantile=0.90):
    """Soglia ottimale su F1, limitata al 90° percentile degli score."""
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    max_th    = np.quantile(p, max_th_quantile)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    f1_scores[thresholds > max_th] = 0
    return thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

def metrics_all(y_true, p, threshold=0.5):
    p   = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(
        AUC   = auc,
        Brier = brier_score_loss(y_true, p),
        F1    = f1_score(y_true, (p >= threshold), zero_division=0),
        Th    = threshold
    )

def agg_mean_sd(list_of_dicts):
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"]   = float(np.nanstd(vals))
    return out

def check_array(name, arr, y=None):
    print(f"\n[CHECK] {name}:")
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    print(f"  NaN={nan_count}  Inf={inf_count}")
    if nan_count > 0 or inf_count > 0:
        print(f"  *** ATTENZIONE: valori non finiti presenti! ***")
    if y is not None:
        print(f"  label: n={len(y)}  pos={y.sum()}  prev={y.mean():.4f}")
    return nan_count == 0 and inf_count == 0

def fit_splines(df_in, cols, n_knots=5, degree=3):
    data  = df_in[cols].fillna(df_in[cols].median())
    tfm   = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    arr   = tfm.fit_transform(data).astype(np.float32)
    n_out = n_knots + degree - 2
    names = [f"{c}_sp{i}" for c in cols for i in range(n_out)]
    return arr, tfm, names





def train_logreg(Xtr, ytr, Xte, yte,  sensitive_tr=None,model_name=""):
    assert np.isfinite(Xtr).all(), f"[{model_name}] Xtr ha NaN/Inf!"
    assert np.isfinite(Xte).all(), f"[{model_name}] Xte ha NaN/Inf!"
    assert np.isin(ytr, [0, 1]).all(), f"[{model_name}] ytr fuori da {{0,1}}!"

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr).astype(np.float32)
    Xte_s  = scaler.transform(Xte).astype(np.float32)

    if not np.isfinite(Xtr_s).all():
        Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=5.0, neginf=-5.0)
        Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=5.0, neginf=-5.0)

    X_train = torch.tensor(Xtr_s, device=DEVICE)
    y_train = torch.tensor(ytr.astype(np.float32), device=DEVICE)
    X_test  = torch.tensor(Xte_s, device=DEVICE)
    sens_train = (
    torch.tensor(sensitive_tr.astype(np.float32), device=DEVICE)
    if sensitive_tr is not None else None
    )


    n_pos = (ytr == 1).sum()
    n_neg = (ytr == 0).sum()
    ratio = n_neg / max(n_pos, 1)

    # Cap a 110: copre il ratio reale ~102 del dataset PP con margine
    pw    = float(np.clip(ratio, 1.0, 110.0))
    pos_w = torch.tensor([pw], dtype=torch.float32, device=DEVICE)
    print(f"  [{model_name}] pos_weight={pw:.1f}x  (ratio reale={ratio:.1f}x)")

    # Bias init sulla prevalenza reale del fold → evita collasso iniziale
    prev      = n_pos / (n_pos + n_neg)
    bias_init = float(np.log(prev / (1 - prev + 1e-9)))
    model     = nn.Sequential(nn.Linear(X_train.shape[1], 1)).to(DEVICE)
    with torch.no_grad():
        model[0].bias.fill_(bias_init)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dp_loss = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=FAIR_ALPHA)


    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(X_train).view(-1)
        loss = criterion(logits, y_train)
        if model_name == "static" and sens_train is not None:
            loss += dp_loss(X_train, logits, sens_train, y_train)
        if not torch.isfinite(loss):
            print(f"  [WARN] Loss non finita all'epoch {epoch}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_test)).view(-1).cpu().numpy()
        p_tr = torch.sigmoid(model(X_train)).view(-1).cpu().numpy()

    bias = model[0].bias.item()
    print(f"  [{model_name}] bias={bias:.4f} → sigmoid={torch.sigmoid(torch.tensor(bias)).item():.4f}")
    print(f"  [{model_name}] p_train: mean={p_tr.mean():.4f}  p_test: mean={p_te.mean():.4f}")
    print(f"  [{model_name}] target  train_prev={ytr.mean():.4f}  test_prev={yte.mean():.4f}")

    return p_te, p_tr, model, scaler

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, usecols=[
    "loan_sequence_number", "loan_age", "loan_term",
    "current_upb", "current_interest_rate", "estimated_ltv",
    "current_loan_delinquency_status", "loan_amount",
    "original_ltv", "original_dti", "credit_score",
    "interest_rate", "num_borrowers",
    "occupancy_status_orig", "loan_purpose_orig",
    "applicant_sex", "derived_race", "applicant_age"
], low_memory=False)

print(f"  Righe caricate: {len(df):,}")
print(f"  Loan unici:     {df['loan_sequence_number'].nunique():,}")

NUM_COLS = [
    "loan_age", "loan_term", "current_upb", "current_interest_rate",
    "estimated_ltv", "loan_amount", "original_ltv", "original_dti",
    "credit_score", "interest_rate", "num_borrowers"
]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

df["occupancy_status_orig"] = df["occupancy_status_orig"].astype("category")
df["loan_purpose_orig"]     = df["loan_purpose_orig"].astype("category")

# ── Demografiche ───────────────────────────────────────────────────────────────
sex_map = {1: 0, 2: 1}
df["sex_bin"] = df["applicant_sex"].map(sex_map)
print(f"  applicant_sex valid: {df['sex_bin'].notna().sum():,} ({df['sex_bin'].notna().mean():.1%})")

def race_map(x):
    if not isinstance(x, str): return np.nan
    x = x.strip().lower()
    if x in ["white", "asian"]: return 0
    if x in ["black or african american", "american indian or alaska native",
              "native hawaiian or other pacific islander", "2 or more races", "other"]: return 1
    return np.nan
df["race_bin"] = df["derived_race"].apply(race_map)
print(f"  applicant_race valid: {df['race_bin'].notna().sum():,} ({df['race_bin'].notna().mean():.1%})")

age_map = {"<25": 1, "25-34": 0, "35-44": 0, "45-54": 0, "55-64": 0, "65-74": 0, ">74": 0}
df["age_bin"] = df["applicant_age"].map(age_map)
print(f"  applicant_age valid: {df['age_bin'].notna().sum():,} ({df['age_bin'].notna().mean():.1%})")

# ── Feature engineering ────────────────────────────────────────────────────────
df["Default"] = is_default_vec(df["current_loan_delinquency_status"])
df.drop(columns=["current_loan_delinquency_status"], inplace=True)

fd_age = (df[df["Default"] == 1]
          .groupby("loan_sequence_number")["loan_age"].min()
          .rename("FirstDefaultAge"))
df = df.merge(fd_age, on="loan_sequence_number", how="left")
df.drop(columns=["Default"], inplace=True)

n_loans      = df["loan_sequence_number"].nunique()
n_defaulters = df.groupby("loan_sequence_number")["FirstDefaultAge"].first().notna().sum()
print(f"\n[CHECK] Composizione dataset:")
print(f"  Loan totali:    {n_loans:,}")
print(f"  Defaulters:     {n_defaulters:,} ({n_defaulters/n_loans:.1%})")
print(f"  Non-defaulters: {n_loans - n_defaulters:,} ({(n_loans-n_defaulters)/n_loans:.1%})")

for col, name in [("sex_bin","sex_bin_loan"),("race_bin","race_bin_loan"),("age_bin","age_bin_loan")]:
    per_loan = df.groupby("loan_sequence_number")[col].first().rename(name)
    df = df.merge(per_loan, on="loan_sequence_number", how="left")

df = df.sort_values(["loan_sequence_number", "loan_age"])

# ── BD_pct: balance deviation vs schedule teorico ──────────────────────────────
print("\nComputing BD_pct...")
df["b_sched"] = df.apply(
    lambda r: scheduled_balance(r["loan_amount"], r["interest_rate"],
                                r["loan_term"],   r["loan_age"]), axis=1
).astype("float32")

df["bd_pct"] = df.apply(
    lambda r: compute_bd_pct(r["current_upb"], r["b_sched"]), axis=1
).astype("float32")

df["bd_pct"] = df["bd_pct"].replace([np.inf, -np.inf], np.nan).clip(-2, 2)
print(f"  bd_pct: NaN={df['bd_pct'].isna().sum():,}  "
      f"mean={df['bd_pct'].mean():.4f}  std={df['bd_pct'].std():.4f}")

# ── Altre TVC ──────────────────────────────────────────────────────────────────
df["upb_change"]  = df.groupby("loan_sequence_number")["current_upb"].diff().fillna(0)
df["rate_change"] = df.groupby("loan_sequence_number")["current_interest_rate"].diff().fillna(0)
df["ltv_change"]  = df.groupby("loan_sequence_number")["estimated_ltv"].diff().fillna(0)

df["upb_pct_change"]  = df.groupby("loan_sequence_number")["current_upb"].pct_change().fillna(0).clip(-5, 5)
df["rate_pct_change"] = df.groupby("loan_sequence_number")["current_interest_rate"].pct_change().fillna(0).clip(-5, 5)
df["ltv_pct_change"]  = df.groupby("loan_sequence_number")["estimated_ltv"].pct_change().fillna(0).clip(-5, 5)

df["ltv_ratio"]  = df["estimated_ltv"] / df["original_ltv"].replace(0, np.nan)
df["rate_ratio"] = df["current_interest_rate"] / df["interest_rate"].replace(0, np.nan)
df["upb_ratio"]  = df["current_upb"] / df["loan_amount"].replace(0, np.nan)

for c in ["ltv_ratio", "rate_ratio", "upb_ratio"]:
    n_inf = np.isinf(df[c]).sum()
    n_nan = df[c].isna().sum()
    print(f"  {c}: Inf={n_inf}  NaN={n_nan}")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

df["cs_lt_620"] = (df["credit_score"]  <  620).astype(np.float32)
df["cs_lt_660"] = (df["credit_score"]  <  660).astype(np.float32)
df["ltv_gt_80"] = (df["estimated_ltv"] >   80).astype(np.float32)
df["ltv_gt_95"] = (df["estimated_ltv"] >   95).astype(np.float32)
df["dti_gt_43"] = (df["original_dti"]  >   43).astype(np.float32)

DUMMY_COLS = ["cs_lt_620", "cs_lt_660", "ltv_gt_80", "ltv_gt_95", "dti_gt_43"]
print(f"\nDummy features — mean values:")
for c in DUMMY_COLS:
    print(f"  {c}: {df[c].mean():.3f}")

df["cs_x_ltv"]  = (df["credit_score"]  / 850.0) * (df["estimated_ltv"]  / 100.0)
df["ltv_x_dti"] = (df["estimated_ltv"] / 100.0) * (df["original_dti"]   / 100.0)
INTERACTION_COLS = ["cs_x_ltv", "ltv_x_dti"]

# bd_pct è time-varying → va nelle TVC, non nelle STATIC
TVC_COLS = [
    "current_upb",
    "current_interest_rate",
    "estimated_ltv",
    "upb_ratio",
    "bd_pct",
]

STATIC_COLS = [
    "credit_score",
    "original_dti",
    "original_ltv",
    "interest_rate",
    "loan_term",
    "num_borrowers",
]

CAT_COLS = ["occupancy_status_orig", "loan_purpose_orig"]

SPLINE_COLS = ["credit_score", "estimated_ltv", "original_dti", "upb_ratio"]
# ↑ tolto current_interest_rate (importanza 0.228, quasi zero)

print(f"\nFitting splines on {SPLINE_COLS}...")
spline_arr_full, spline_tfm, spline_names = fit_splines(df, SPLINE_COLS, n_knots=5, degree=3)
spline_df_full = pd.DataFrame(spline_arr_full, columns=spline_names, index=df.index)
df = pd.concat([df, spline_df_full], axis=1)
del spline_arr_full, spline_df_full
gc.collect()
print(f"  Spline features aggiunte: {len(spline_names)}")

# ── Person-period dataset ──────────────────────────────────────────────────────
print("\nBuilding PERSON-PERIOD dataset...")
df_pp = df.sort_values(["loan_sequence_number", "loan_age"]).copy()
df_pp["is_default_now"] = (
    df_pp["FirstDefaultAge"].notna() &
    (df_pp["loan_age"] == df_pp["FirstDefaultAge"])
).astype(np.int8)
df_pp["default_next"] = (
    df_pp.groupby("loan_sequence_number")["is_default_now"]
    .shift(-1).fillna(0).astype(np.int8)
)
last_mask = df_pp["loan_age"] == df_pp.groupby("loan_sequence_number")["loan_age"].transform("max")
df_pp = df_pp[~last_mask]
df_pp = df_pp[df_pp["FirstDefaultAge"].isna() | (df_pp["loan_age"] < df_pp["FirstDefaultAge"])]
pp_df = df_pp.copy()
del df_pp
gc.collect()
print(f"  Righe: {len(pp_df):,} | Default: {pp_df['default_next'].sum()} ({pp_df['default_next'].mean():.2%})")

enc_cat_pp = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
cats_pp    = enc_cat_pp.fit_transform(pp_df[CAT_COLS])
medians_pp = pp_df[STATIC_COLS + TVC_COLS].median()

print(f"\n[CHECK] Mediane imputation PP (prime 5):")
print(medians_pp.head(5).to_string())

spline_pp = pp_df[spline_names].to_numpy(dtype=np.float32)

# Spline temporale 3 knots (meno overfitting sulla discretizzazione mensile)
spline_arr_time, spline_tfm_time, spline_names_time = fit_splines(
    pp_df, ["loan_age"], n_knots=3, degree=3
)
# log1p(loan_age): feature monotona esplicita, coeff atteso negativo
log_age_pp = np.log1p(pp_df["loan_age"].to_numpy(dtype=np.float32)).reshape(-1, 1)

X_pp = np.hstack([
    pp_df[STATIC_COLS].fillna(medians_pp[STATIC_COLS]).to_numpy(dtype=np.float32),
    pp_df[TVC_COLS].fillna(medians_pp[TVC_COLS]).to_numpy(dtype=np.float32),
    cats_pp,
    spline_pp,
    spline_arr_time,
    log_age_pp,
])
y_pp    = pp_df["default_next"].to_numpy(dtype=np.int8)
grp_pp  = pp_df["loan_sequence_number"].to_numpy()
pp_ages = pp_df["loan_age"].to_numpy()
sex_pp  = pp_df["sex_bin_loan"].to_numpy()
age_pp  = pp_df["age_bin_loan"].to_numpy()
race_pp = pp_df["race_bin_loan"].to_numpy()
nb_pp   = pp_df["num_borrowers"].to_numpy()

check_array("X_pp", X_pp, y_pp)
del pp_df, cats_pp, spline_pp
gc.collect()

# ── Static dataset ─────────────────────────────────────────────────────────────
print("\nBuilding STATIC dataset (t=0)...")
static_df = df.sort_values("loan_age").groupby("loan_sequence_number").first().reset_index()
static_df["target_static"] = (
    static_df["FirstDefaultAge"].notna() &
    (static_df["FirstDefaultAge"] <= HORIZON_MONTHS)
).astype(np.int8)

enc_cat_s = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
cats_s    = enc_cat_s.fit_transform(static_df[CAT_COLS])
medians_s = static_df[STATIC_COLS].median()

SPLINE_COLS_STATIC  = [c for c in SPLINE_COLS if c in STATIC_COLS]
spline_names_static = [n for n in spline_names if any(n.startswith(c) for c in SPLINE_COLS_STATIC)]
spline_s = static_df[spline_names_static].to_numpy(dtype=np.float32)

X_static = np.hstack([
    static_df[STATIC_COLS].fillna(medians_s).to_numpy(dtype=np.float32),
    cats_s,
    spline_s,
])
y_static    = static_df["target_static"].to_numpy(dtype=np.int8)
grp_static  = static_df["loan_sequence_number"].to_numpy()
sex_static  = static_df["sex_bin_loan"].to_numpy()
race_static = static_df["race_bin_loan"].to_numpy()
age_static  = static_df["age_bin_loan"].to_numpy()
print(f"  Righe: {len(X_static):,} | Default: {y_static.sum()} ({y_static.mean():.2%})")
check_array("X_static", X_static, y_static)

# ── Dynamic dataset ────────────────────────────────────────────────────────────
print("\nBuilding DYNAMIC dataset (landmarks + TVC)...")
KEEP = (["loan_sequence_number", "future_default", "landmark"] +
        STATIC_COLS + TVC_COLS + CAT_COLS + spline_names +
        ["loan_age", "sex_bin_loan", "race_bin_loan", "age_bin_loan"])

lm_rows = []
for L in LANDMARKS:
    snap = df[df["loan_age"] == L].copy()
    if len(snap) == 0: continue
    snap = snap[snap["FirstDefaultAge"].isna() | (snap["FirstDefaultAge"] > L)].copy()
    snap["future_default"] = (
        snap["FirstDefaultAge"].notna() &
        (snap["FirstDefaultAge"] > L) &
        (snap["FirstDefaultAge"] <= L + HORIZON_MONTHS)
    ).astype(np.int8)
    snap["landmark"] = np.int8(L)
    lm_rows.append(snap[[c for c in KEEP if c in snap.columns]])

landmark_df = pd.concat(lm_rows, ignore_index=True)
del lm_rows, df
gc.collect()
print(f"  Righe: {len(landmark_df):,} | Default: {landmark_df['future_default'].sum()} ({landmark_df['future_default'].mean():.2%})")

enc_cat_d = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
cats_d    = enc_cat_d.fit_transform(landmark_df[CAT_COLS])
enc_lmk   = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
lmk_oh    = enc_lmk.fit_transform(landmark_df[["landmark"]])
medians_d = landmark_df[STATIC_COLS + TVC_COLS].median()
spline_d  = landmark_df[spline_names].to_numpy(dtype=np.float32)

X_dynamic = np.hstack([
    landmark_df[STATIC_COLS].fillna(medians_d[STATIC_COLS]).to_numpy(dtype=np.float32),
    landmark_df[TVC_COLS].fillna(medians_d[TVC_COLS]).to_numpy(dtype=np.float32),
    cats_d,
    spline_d,
    lmk_oh,
])
y_dynamic    = landmark_df["future_default"].to_numpy(dtype=np.int8)
grp_dynamic  = landmark_df["loan_sequence_number"].to_numpy()
lmk_vals     = landmark_df["landmark"].to_numpy()
sex_dynamic  = landmark_df["sex_bin_loan"].to_numpy()
race_dynamic = landmark_df["race_bin_loan"].to_numpy()
age_dynamic  = landmark_df["age_bin_loan"].to_numpy()
check_array("X_dynamic", X_dynamic, y_dynamic)
del cats_d, lmk_oh, spline_d
gc.collect()

# ── Cross-validation ───────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=5)

metrics_static  = []; times_static  = []
metrics_dynamic = []; times_dynamic = []
metrics_pp      = []; times_pp      = []

# ── Static model ───────────────────────────────────────────────────────────────
print("\nTraining STATIC model...")
static_oof_preds = np.zeros(len(y_static))

for fold, (tr, te) in enumerate(gkf.split(X_static, y_static, grp_static)):
    t0 = time.perf_counter()
    p_test, p_train, model, _ = train_logreg(
        X_static[tr], y_static[tr], X_static[te], y_static[te],race_static[tr],
        model_name="static"
    )
    best_th = find_best_threshold(y_static[tr], p_train)
    static_oof_preds[te] = p_test
    metrics_static.append(metrics_all(y_static[te].astype(int), p_test, threshold=best_th))
    times_static.append(time.perf_counter() - t0)
    print(f"  Fold {fold+1} — AUC: {metrics_static[-1]['AUC']:.4f}")
    # Static loop — dopo print fold AUC
    wandb.log({
        "static/fold": fold+1,
        "static/AUC":  metrics_static[-1]['AUC'],
        "static/Brier": metrics_static[-1]['Brier'],
        "static/F1":   metrics_static[-1]['F1'],
        "static/time": times_static[-1],
    })

del X_static
gc.collect()

# ── Dynamic model ──────────────────────────────────────────────────────────────
print("\nTraining DYNAMIC model...")
dynamic_oof_preds      = np.zeros(len(y_dynamic))
metrics_dynamic_by_lmk = {L: [] for L in LANDMARKS}

for fold, (tr, te) in enumerate(gkf.split(X_dynamic, y_dynamic, grp_dynamic)):
    t0 = time.perf_counter()
    p_test, p_train, model, _ = train_logreg(
        X_dynamic[tr], y_dynamic[tr], X_dynamic[te], y_dynamic[te],
        model_name="dynamic"
    )
    dynamic_oof_preds[te] = p_test
    best_th = find_best_threshold(y_dynamic[tr], p_train)
    metrics_dynamic.append(metrics_all(y_dynamic[te].astype(int), p_test, threshold=best_th))
    times_dynamic.append(time.perf_counter() - t0)
    print(f"  Fold {fold+1} — AUC: {metrics_dynamic[-1]['AUC']:.4f}")


    # Dynamic loop — dopo print fold AUC
    wandb.log({
        "dynamic/fold": fold+1,
        "dynamic/AUC":  metrics_dynamic[-1]['AUC'],
        "dynamic/Brier": metrics_dynamic[-1]['Brier'],
        "dynamic/F1":   metrics_dynamic[-1]['F1'],
        "dynamic/time": times_dynamic[-1],
    })


    if fold == 4:
        for L in LANDMARKS:
            mask = lmk_vals[te] == L
            if mask.sum() > 10 and len(np.unique(y_dynamic[te][mask])) > 1:
                metrics_dynamic_by_lmk[L].append(
                    metrics_all(y_dynamic[te][mask].astype(int), p_test[mask], threshold=best_th)
                )
del X_dynamic
gc.collect()

# ── Person-period model ────────────────────────────────────────────────────────
print("\nTraining PERSON-PERIOD model...")
pp_oof_preds  = np.zeros(len(y_pp))
model_pp_last = None

for fold, (tr, te) in enumerate(gkf.split(X_pp, y_pp, grp_pp)):
    t0 = time.perf_counter()
    p_test, p_train, model, _ = train_logreg(
        X_pp[tr], y_pp[tr], X_pp[te], y_pp[te],
        model_name="person_period"
    )
    pp_oof_preds[te] = p_test
    best_th = find_best_threshold(y_pp[tr], p_train)
    metrics_pp.append(metrics_all(y_pp[te].astype(int), p_test, threshold=best_th))
    times_pp.append(time.perf_counter() - t0)
    print(f"  Fold {fold+1} — AUC: {metrics_pp[-1]['AUC']:.4f}  best_th={best_th:.5f}")
    # PP loop — dopo print fold AUC
    wandb.log({
        "pp/fold":  fold+1,
        "pp/AUC":   metrics_pp[-1]['AUC'],
        "pp/Brier": metrics_pp[-1]['Brier'],
        "pp/F1":    metrics_pp[-1]['F1'],
        "pp/time":  times_pp[-1],
    })
    if fold == 4:
        model_pp_last = model
del X_pp
gc.collect()

# ── Calibrazione OOF ──────────────────────────────────────────────────────────
print("\n[CHECK] Calibrazione OOF — real_prev vs pred_mean:")
for name, y_true, y_pred in [
    ("STATIC",  y_static,  static_oof_preds),
    ("DYNAMIC", y_dynamic, dynamic_oof_preds),
    ("PP",      y_pp,      pp_oof_preds),
]:
    ratio_cal = y_pred.mean() / max(y_true.mean(), 1e-9)
    print(f"  {name}: real={y_true.mean():.4f}  pred={y_pred.mean():.4f}  ratio={ratio_cal:.2f}x")

pp_diag = pd.DataFrame({"loan_age": pp_ages, "pred": pp_oof_preds, "label": y_pp})
diag = pp_diag.groupby(pd.cut(pp_diag["loan_age"], bins=10)).agg(
    real_prev=("label","mean"), pred_mean=("pred","mean"), n=("label","count")
).round(4)
print("\nDiagnostica calibrazione PP per bucket età:")
print(diag)

print("\n[CHECK] Separazione score PP:")
pos_scores = pp_oof_preds[y_pp == 1]
neg_scores = pp_oof_preds[y_pp == 0]
print(f"  Defaulters:     mean={pos_scores.mean():.4f}  median={np.median(pos_scores):.4f}")
print(f"  Non-defaulters: mean={neg_scores.mean():.4f}  median={np.median(neg_scores):.4f}")
print(f"  Differenza media: {pos_scores.mean() - neg_scores.mean():.4f}")

# ── Plots diagnostica ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
pp_diag2 = pd.DataFrame({"loan_age": pp_ages, "pred": pp_oof_preds,
                          "label": y_pp, "num_borrowers": nb_pp})
for nb in sorted(pp_diag2["num_borrowers"].dropna().unique()):
    mask = pp_diag2["num_borrowers"] == nb
    axes[0].scatter(pp_diag2.loc[mask, "loan_age"], pp_diag2.loc[mask, "pred"],
                    alpha=0.05, s=2, label=f"nb={nb:.0f}")
axes[0].set_title("Pred h(t) per num_borrowers")
axes[0].set_xlabel("Loan Age"); axes[0].set_ylabel("Predicted h(t)")
axes[0].legend(markerscale=5)

axes[1].hist(pp_oof_preds[y_pp == 0], bins=50, alpha=0.6, label="No default", density=True)
axes[1].hist(pp_oof_preds[y_pp == 1], bins=50, alpha=0.6, label="Default",    density=True)
axes[1].axvline(pp_oof_preds.mean(), color="black", linestyle="--",
                label=f"mean={pp_oof_preds.mean():.3f}")
axes[1].set_title("Distribuzione pred per label — PP")
axes[1].set_xlabel("Predicted h(t)"); axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "diag_calibrazione_pp.png", dpi=150)
plt.show()

# ── Feature importance ─────────────────────────────────────────────────────────
pp_feature_names = (
    STATIC_COLS + TVC_COLS +
    list(enc_cat_pp.get_feature_names_out(CAT_COLS)) +
    spline_names + spline_names_time + ["log1p_loan_age"]
)
coef = model_pp_last[0].weight.detach().cpu().numpy()[0]

if len(pp_feature_names) != len(coef):
    print(f"[ERROR] Feature names={len(pp_feature_names)} vs coef={len(coef)} — MISMATCH!")
else:
    coef_series = pd.Series(np.abs(coef), index=pp_feature_names).sort_values(ascending=False)
    raw_coef    = pd.Series(coef, index=pp_feature_names)
    print("\nTop-15 features by |coefficient| (M_PP):")
    print(coef_series.head(15).to_string())

    print(f"\n[CHECK] log1p_loan_age raw coef = {raw_coef.get('log1p_loan_age', np.nan):.6f}  (atteso: negativo)")
    print(f"[CHECK] bd_pct          raw coef = {raw_coef.get('bd_pct', np.nan):.6f}   (atteso: positivo)")

    def aggregate_spline_importance(cs, snames, orig_cols):
        agg = {}
        for c in orig_cols:
            keys = [n for n in snames if n.startswith(c + "_sp")]
            agg[f"{c}_SPLINE"] = cs[keys].sum()
        return pd.Series(agg).sort_values(ascending=False)

    spline_agg = aggregate_spline_importance(coef_series, spline_names, SPLINE_COLS)
    print("\nSpline importance aggregata:")
    print(spline_agg.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    coef_series.head(15).plot(kind="bar", ax=axes[0], title="Top-15 feature |coef| — M_PP")
    spline_agg.plot(kind="bar", ax=axes[1], title="Spline importance — M_PP")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_pp.png", dpi=150)
    plt.show()

# ── Hazard plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
mask_pos = (y_pp == 1); mask_neg = (y_pp == 0)
axes[0].scatter(pp_ages[mask_neg], pp_oof_preds[mask_neg], alpha=0.03, s=2, color="steelblue")
axes[0].set_title("Non-default"); axes[0].set_xlabel("Loan Age"); axes[0].set_ylabel("h(t)")
axes[1].scatter(pp_ages[mask_pos], pp_oof_preds[mask_pos], alpha=0.3,  s=5, color="crimson")
axes[1].set_title("Default"); axes[1].set_xlabel("Loan Age")
plt.suptitle("Hazard by Loan Age — M_PP (split by outcome)")
plt.tight_layout()
plt.savefig(OUT_DIR / "hazard_by_age_split.png", dpi=150)
plt.show()

pp_df_tmp = pd.DataFrame({"age_bin": pd.cut(pp_ages, bins=10), "pred": pp_oof_preds, "label": y_pp})
pp_df_tmp.groupby(["age_bin", "label"])["pred"].mean().unstack().plot(
    kind="bar", figsize=(12, 4), title="Mean predicted h(t) by age bucket and label — M_PP"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "hazard_boxplot.png", dpi=150)
plt.show()

# ── Summary ────────────────────────────────────────────────────────────────────
def make_row(name, metric_list, time_list):
    row = agg_mean_sd(metric_list)
    row["Model"]         = name
    row["Time_Mean_sec"] = float(np.mean(time_list))
    row["Time_SD_sec"]   = float(np.std(time_list))
    return row

summary = pd.DataFrame([
    make_row("M_STATIC",  metrics_static,  times_static),
    make_row("M_DYNAMIC", metrics_dynamic, times_dynamic),
    make_row("M_PP",      metrics_pp,      times_pp),
])[[
    "Model","AUC_Mean","AUC_SD","Brier_Mean","Brier_SD",
    "F1_Mean","F1_SD","Time_Mean_sec","Time_SD_sec"
]]
print("\n=== RISULTATI FINALI ===")
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "comparison_static_vs_dynamic.csv", index=False)

# ── AUC per landmark ───────────────────────────────────────────────────────────
lmk_rows = []
for L in LANDMARKS:
    if metrics_dynamic_by_lmk[L]:
        r = agg_mean_sd(metrics_dynamic_by_lmk[L])
        r["Landmark"] = L
        lmk_rows.append(r)
if lmk_rows:
    lmk_summary = pd.DataFrame(lmk_rows)[["Landmark","AUC_Mean","Brier_Mean","F1_Mean"]]
    print("\n=== DINAMICO — AUC PER LANDMARK ===")
    print(lmk_summary.to_string(index=False))
    lmk_summary.to_csv(OUT_DIR / "dynamic_by_landmark.csv", index=False)

# ── Fairness ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FAIRNESS ANALYSIS")
print("="*60)

# ── Soglie globali ─────────────────────────────────────────────────────────────
def compute_threshold(y_true, y_pred):
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return thr[np.argmax(f1)] if len(thr) > 0 else 0.5

th_static  = compute_threshold(y_static,  static_oof_preds)
th_dynamic = compute_threshold(y_dynamic, dynamic_oof_preds)
th_pp      = compute_threshold(y_pp,      pp_oof_preds)

# Questi devono avere la stessa lunghezza di y_static, y_dynamic, y_pp
ybin_static  = (static_oof_preds  >= th_static ).astype(int)
ybin_dynamic = (dynamic_oof_preds >= th_dynamic).astype(int)
ybin_pp      = (pp_oof_preds      >= th_pp     ).astype(int)

print(f"Shapes check:")
print(f"  y_static={len(y_static)}  ybin_static={len(ybin_static)}  race_static={len(race_static)}")
print(f"  y_dynamic={len(y_dynamic)}  ybin_dynamic={len(ybin_dynamic)}  race_dynamic={len(race_dynamic)}")
print(f"  y_pp={len(y_pp)}  ybin_pp={len(ybin_pp)}  race_pp={len(race_pp)}")

# ── 1. AGGREGATE fairness (tutti i t insieme) ──────────────────────────────────
print("\n--- AGGREGATE ---")
agg_rows = []
for attr_name, GROUP_NAMES, s_stat, s_dyn, s_pp in [
    ("SEX",  {0:"Male",        1:"Female"},      sex_static,  sex_dynamic,  sex_pp),
    ("RACE", {0:"White/Asian", 1:"Black/Indian"}, race_static, race_dynamic, race_pp),
    ("AGE",  {0:"Old",         1:"Young"},        age_static,  age_dynamic,  age_pp),
]:
   print(f"\n{attr_name}")
   for mname, y_t, y_p, y_b, sens, th in [
    ("M_STATIC",  y_static,  static_oof_preds,  ybin_static,  s_stat, th_static),
    ("M_DYNAMIC", y_dynamic, dynamic_oof_preds, ybin_dynamic, s_dyn,  th_dynamic),
    ("M_PP",      y_pp,      pp_oof_preds,      ybin_pp,      s_pp,   th_pp),
]:
    res = fairness_metrics(y_t, y_p, y_b, sens, GROUP_NAMES, threshold=th)
    print_fairness_report(mname, res, GROUP_NAMES, label="AGGREGATE")
    agg_rows.append(res_to_row(res, GROUP_NAMES, {
        "attr": attr_name, "model": mname, "type": "aggregate"
    }))


df_fair_aggregate = pd.DataFrame(agg_rows)
df_fair_aggregate.to_csv(OUT_DIR / "fairness_aggregate.csv", index=False)
print("\nSalvato: fairness_aggregate.csv")

# ── 2. DYNAMIC per landmark ────────────────────────────────────────────────────
print("\n--- DYNAMIC PER LANDMARK ---")
lmk_rows = []
for attr_name, GROUP_NAMES, s_dyn in [
    ("SEX",  {0:"Male",        1:"Female"},      sex_dynamic),
    ("RACE", {0:"White/Asian", 1:"Black/Indian"}, race_dynamic),
    ("AGE",  {0:"Old",         1:"Young"},        age_dynamic),
]:
    for L in LANDMARKS:
       #Per ogni landmark (0, 3, 6... 48 mesi), crea una maschera booleana che seleziona
       #solo le righe corrispondenti a quel momento temporale
        mask_L = lmk_vals == L
        yt, yp, sn = filter_sensitive(
            y_dynamic[mask_L], dynamic_oof_preds[mask_L], s_dyn[mask_L]
        )
        valid = np.isin(s_dyn[mask_L], [0, 1])
        yb    = ybin_dynamic[mask_L][valid]
        if len(np.unique(yt)) < 2 or len(np.unique(sn)) < 2 or len(yt) < 100:
            continue
        #Calcola le metriche, le stampa con il label che indica landmark e attributo
        res = fairness_metrics(yt, yp, yb, sn, GROUP_NAMES, threshold=th_dynamic)
        lmk_rows.append(res_to_row(res, GROUP_NAMES, {
            "attr": attr_name, "model": "M_DYNAMIC",
            "type": "by_landmark", "landmark": L
        }))

df_fair_dynamic_lmk = pd.DataFrame(lmk_rows)
df_fair_dynamic_lmk.to_csv(OUT_DIR / "fairness_dynamic_by_landmark.csv", index=False)
print("\nSalvato: fairness_dynamic_by_landmark.csv")

# ── 3. PERSON-PERIOD per age bin ───────────────────────────────────────────────
print("\n--- PERSON-PERIOD PER LOAN AGE ---")
max_age = int(np.nanmax(pp_ages))
AGE_BINS = [(m, m+1) for m in range(0, max_age)]
pp_rows = []
for attr_name, GROUP_NAMES, s_pp in [
    ("SEX",  {0:"Male",        1:"Female"},      sex_pp),
    ("RACE", {0:"White/Asian", 1:"Black/Indian"}, race_pp),
    ("AGE",  {0:"Old",         1:"Young"},        age_pp),
]:
    for (age_lo, age_hi) in AGE_BINS:
      #La maschera seleziona tutte le osservazioni del mutuo che cadono
      #in quella fascia temporale
        mask_a = (pp_ages >= age_lo) & (pp_ages < age_hi)
        yt, yp, sn = filter_sensitive(
            y_pp[mask_a], pp_oof_preds[mask_a], s_pp[mask_a]
        )
        valid = np.isin(s_pp[mask_a], [0, 1])
        yb    = ybin_pp[mask_a][valid]
        if len(np.unique(yt)) < 2 or len(np.unique(sn)) < 2 or len(yt) < 100:
            continue
        res = fairness_metrics(yt, yp, yb, sn, GROUP_NAMES, threshold=th_pp)
        pp_rows.append(res_to_row(res, GROUP_NAMES, {
            "attr": attr_name, "model": "M_PP",
            "type": "by_age", "age_lo": age_lo, "age_hi": age_hi,
            "age_bin": f"{age_lo}-{age_hi}m"
        }))

df_fair_pp_age = pd.DataFrame(pp_rows)
df_fair_pp_age.to_csv(OUT_DIR / "fairness_pp_by_age.csv", index=False)
print("\nSalvato: fairness_pp_by_age.csv")


# Plot dynamic per landmark
plot_fairness_over_time(
    df_fair_dynamic_lmk,
    time_col   = "landmark",
    title      = "Fairness evolution — Dynamic model by landmark",
    filename   = "fairness_dynamic_by_landmark.png",
    static_df  = df_fair_aggregate        # ← nuovo parametro
)

# Plot PP per age bin
plot_fairness_over_time(
    df_fair_pp_age,
    time_col   = "age_lo",
    title      = "Fairness evolution — Person-Period model by loan age",
    filename   = "fairness_pp_by_age.png",
    static_df  = df_fair_aggregate        # ← nuovo parametro
)

for attr_name, s_stat, s_dyn, s_pp in [
    ("SEX",  sex_static,  sex_dynamic,  sex_pp),
    ("RACE", race_static, race_dynamic, race_pp),
    ("AGE",  age_static,  age_dynamic,  age_pp),
]:
    res_stat = compute_adTPR_adFPR(y_static,  ybin_static,  s_stat)           # time_points=None
    res_dyn  = compute_adTPR_adFPR(y_dynamic, ybin_dynamic, s_dyn,  lmk_vals) # per landmark
    res_pp   = compute_adTPR_adFPR(y_pp,      ybin_pp,      s_pp,   pp_ages)  # per mese

    print(f"\n{attr_name}")
    print(f"  STATIC  — dTPR={res_stat['adTPR']:.4f}  dFPR={res_stat['adFPR']:.4f}")
    print(f"  DYNAMIC — adTPR={res_dyn['adTPR']:.4f}  adFPR={res_dyn['adFPR']:.4f}")
    print(f"  PP      — adTPR={res_pp['adTPR']:.4f}  adFPR={res_pp['adFPR']:.4f}")


df_auc_comparison = auc_fairness_all_models(
    df_fair_dynamic_lmk,
    df_fair_pp_age,
    df_fair_aggregate,
    time_col_dyn = "landmark",
    time_col_pp  = "age_lo"
)

print("\n=== AUC FAIRNESS COMPARISON ===")
print(df_auc_comparison.to_string(index=False))
df_auc_comparison.to_csv(OUT_DIR / "auc_fairness_comparison.csv", index=False)


# ── Log su W&B ─────────────────────────────────────────────────────────────────
for _, row in df_fair_aggregate.iterrows():
    prefix = f"{row['model'].lower()}/{row['attr']}/aggregate"
    wandb.log({
        f"{prefix}/independence": row.get("independence"),
        f"{prefix}/separation":   row.get("separation"),
        f"{prefix}/sufficiency":  row.get("sufficiency"),
        f"{prefix}/dp_gap":       row.get("dp_gap"),
        f"{prefix}/tpr_gap":      row.get("tpr_gap"),
        f"{prefix}/auc_gap":      row.get("auc_gap"),
    })

# Loga solo le immagini su W&B
wandb.log({
    "fairness_dynamic_landmark_plot": wandb.Image(str(OUT_DIR / "fairness_dynamic_by_landmark.png")),
    "fairness_pp_age_plot":           wandb.Image(str(OUT_DIR / "fairness_pp_by_age.png")),
})


wandb.finish()
