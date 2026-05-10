"""
Intervention 2 — Upgraded Triage with Licensed Staff (Diversion Classifier)

Three public functions — all share the same return schema so you can
swap which one feeds results["intervention_2_triage_diversion"] in analysis.py:

    intervention2(ed, pat_feat, ...)          — Logistic Regression
    intervention2_rf(ed, pat_feat, ...)       — Random Forest
    intervention2_xg(ed, pat_feat, ...)       — XGBoost

All return:
  train_mse, test_mse, auc_roc, sensitivity_recall, specificity,
  precision, f1_score, p_value, correlation_coeff_r,
  feature_importance_table (importances for RF/XG; coefficients for LR),
  projected_ed_reduction_pct, hyperparameters, and bookkeeping counts.

train_mse / test_mse are Brier scores (MSE of predicted probability vs
binary label) — lower is better.
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, mean_squared_error,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TRIAGE_FEATURES = [
    # Original triage features
    "is_weekend", "is_offhours", "approx_age", "is_female",
    "sdoh_unique_domains", "sdoh_financial_resource_strain",
    "sdoh_housing_stability", "sdoh_transportation_needs",
    "encounters_per_month", "pcp_ratio", "mychart_active",
    # Encounter-level: how the patient arrived (encoded in analysis.py onto ed)
    "admission_src_self", "admission_type_non_emergency",
    # Patient-level: demographics + history (computed in analysis.py onto pat_feat)
    "smoking_current", "is_hispanic", "race_white", "race_black", "is_married",
    "comorbidity_count", "prior_preventable", "prior_inpatient_frac",
]

_DEFAULT_DIVERT_RATES = {"conservative": 0.50, "base": 0.70, "optimistic": 0.90}


def intervention2(
    ed: pd.DataFrame,
    pat_feat: pd.DataFrame,
    C: float = 1.0,
    max_iter: int = 500,
    class_weight: str = "balanced",
    solver: str = "lbfgs",
    test_size: float = 0.2,
    random_state: int = 42,
    n_boot: int = 200,
    divert_rates: dict = None,
) -> dict:
    """
    Fit a logistic regression triage diversion classifier.

    Parameters
    ----------
    ed            ED-filtered encounters DataFrame.
                  Required columns: PatientDurableKey, is_avoidable,
                  is_weekend, AdmitHour.
    pat_feat      Patient-level feature DataFrame.
    C             Inverse regularisation strength (default 1.0).
    max_iter      Max solver iterations (default 500).
    class_weight  'balanced' or None (default 'balanced').
    solver        sklearn solver name (default 'lbfgs').
    test_size     Fraction of data held out for evaluation (default 0.2).
    random_state  RNG seed (default 42).
    n_boot        Bootstrap iterations for coefficient p-values (default 200).
    divert_rates  Dict with keys conservative/base/optimistic mapping to the
                  fraction of flagged-avoidable visits successfully diverted.
                  Default: {"conservative": 0.50, "base": 0.70, "optimistic": 0.90}.

    Returns
    -------
    dict with train_mse, test_mse, and all fields expected by
    results["intervention_2_triage_diversion"] in analysis.py.
    """
    if divert_rates is None:
        divert_rates = dict(_DEFAULT_DIVERT_RATES)

    # ── Build encounter-level feature matrix ──────────────────────────────────
    ed_pat = ed.merge(
        pat_feat[[
            "PatientDurableKey", "approx_age", "is_female",
            "sdoh_unique_domains", "sdoh_financial_resource_strain",
            "sdoh_housing_stability", "sdoh_transportation_needs",
            "ed_ratio", "encounters_per_month", "pcp_ratio", "mychart_active",
            # New patient-level features added by analysis.py
            "smoking_current", "is_hispanic", "race_white", "race_black", "is_married",
            "comorbidity_count", "prior_preventable", "prior_inpatient_frac",
        ]],
        on="PatientDurableKey", how="inner",
    )
    ed_pat["is_offhours"] = (
        ed_pat["is_weekend"]
        | ed_pat["AdmitHour"].between(0, 5)
        | ed_pat["AdmitHour"].between(22, 23)
    ).astype(int)

    _feats = [f for f in TRIAGE_FEATURES if f in ed_pat.columns]
    model_df = ed_pat[_feats + ["is_avoidable"]].dropna()
    X = model_df[_feats].values
    y = model_df["is_avoidable"].values

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Fit classifier ────────────────────────────────────────────────────────
    clf = LogisticRegression(
        C=C, max_iter=max_iter, class_weight=class_weight,
        solver=solver, random_state=random_state,
    )
    clf.fit(X_train_sc, y_train)

    # ── MSE (Brier score) on train and test ───────────────────────────────────
    train_prob = clf.predict_proba(X_train_sc)[:, 1]
    test_prob  = clf.predict_proba(X_test_sc)[:, 1]
    train_mse  = float(mean_squared_error(y_train, train_prob))
    test_mse   = float(mean_squared_error(y_test, test_prob))

    # ── Classification metrics on test set ────────────────────────────────────
    y_pred = clf.predict(X_test_sc)
    auc    = float(roc_auc_score(y_test, test_prob))
    recall = float(recall_score(y_test, y_pred))
    prec   = float(precision_score(y_test, y_pred, zero_division=0))
    f1     = float(f1_score(y_test, y_pred))
    cm     = confusion_matrix(y_test, y_pred)
    spec   = float(cm[0, 0] / (cm[0, 0] + cm[0, 1]))

    # ── Refit on full data for stable coefficient reporting ───────────────────
    X_all_sc = scaler.fit_transform(X)
    clf_full = LogisticRegression(
        C=C, max_iter=max_iter, class_weight=class_weight,
        solver=solver, random_state=random_state,
    )
    clf_full.fit(X_all_sc, y)

    # ── Bootstrap coefficient standard errors (index-based, no walrus hack) ───
    rng = np.random.RandomState(random_state)
    boot_coefs = []
    for i in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        clf_b = LogisticRegression(
            C=C, max_iter=300, class_weight=class_weight,
            solver=solver, random_state=i,
        )
        clf_b.fit(scaler.transform(X[idx]), y[idx])
        boot_coefs.append(clf_b.coef_[0])
    boot_coefs = np.array(boot_coefs)

    coef_se = boot_coefs.std(axis=0)
    coef_z  = clf_full.coef_[0] / np.where(coef_se == 0, np.nan, coef_se)
    coef_p  = 2 * (1 - stats.norm.cdf(np.abs(coef_z)))

    coef_table = [
        {
            "feature"    : feat,
            "coefficient": round(float(c), 4),
            "std_error"  : round(float(se), 4),
            "z_score"    : round(float(z), 4),
            "p_value"    : round(float(p), 4),
        }
        for feat, c, se, z, p in zip(
            _feats, clf_full.coef_[0], coef_se, coef_z, coef_p
        )
    ]

    # ── Overall model p-value (likelihood-ratio chi-square approximation) ─────
    null_acc  = max(y.mean(), 1 - y.mean())
    model_acc = float(clf_full.score(X_all_sc, y))
    G2        = 2 * len(y) * max(0.0, model_acc - null_acc)
    p_val     = float(stats.chi2.sf(G2, df=len(_feats)))

    # ── Projected ED reduction ────────────────────────────────────────────────
    avoidable_rate = ed["is_avoidable"].mean()
    sens = {
        k: round(recall * avoidable_rate * v * 100, 2)
        for k, v in divert_rates.items()
    }
    divert_pct_str = "/".join(
        str(int(divert_rates[s] * 100))
        for s in ("conservative", "base", "optimistic")
    )

    return {
        "train_mse"                  : round(train_mse, 6),
        "test_mse"                   : round(test_mse, 6),
        "n_model"                    : len(model_df),
        "n_train"                    : len(y_train),
        "n_test"                     : len(y_test),
        "features_used"              : TRIAGE_FEATURES,
        "auc_roc"                    : round(auc, 4),
        "sensitivity_recall"         : round(recall, 4),
        "specificity"                : round(spec, 4),
        "precision"                  : round(prec, 4),
        "f1_score"                   : round(f1, 4),
        "p_value"                    : p_val,
        "correlation_coeff_r"        : round(auc * 2 - 1, 4),
        "coefficient_table"          : coef_table,
        "projected_ed_reduction_pct" : sens,
        "sensitivity_assumption"     : (
            f"{divert_pct_str} % of triage-flagged avoidable visits "
            "are successfully diverted"
        ),
        "hyperparameters"            : {
            "C"           : C,
            "max_iter"    : max_iter,
            "class_weight": class_weight,
            "solver"      : solver,
            "test_size"   : test_size,
            "random_state": random_state,
            "n_boot"      : n_boot,
        },
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_model_df(ed: pd.DataFrame, pat_feat: pd.DataFrame):
    """Join ED encounters with patient features and add is_offhours flag."""
    # 1. Merge the dataframes
    ed_pat = ed.merge(
        pat_feat[[
            "PatientDurableKey", "approx_age", "is_female",
            "sdoh_unique_domains", "sdoh_financial_resource_strain",
            "sdoh_housing_stability", "sdoh_transportation_needs",
            "ed_ratio", "encounters_per_month", "pcp_ratio", "mychart_active",
            # New patient-level features added by analysis.py
            "smoking_current", "is_hispanic", "race_white", "race_black", "is_married",
            "comorbidity_count", "prior_preventable", "prior_inpatient_frac",
        ]],
        on="PatientDurableKey", how="inner",
    )
    
    ed_pat["is_offhours"] = (
        ed_pat["is_weekend"]
        | ed_pat["AdmitHour"].between(0, 5)
        | ed_pat["AdmitHour"].between(22, 23)
    ).astype(int)

    # encounter-level features (admission_src_self, admission_type_emergency) come
    # from ed; patient-level features come from pat_feat — both land in ed_pat after merge.
    avail = [f for f in TRIAGE_FEATURES if f in ed_pat.columns]
    model_df = ed_pat[avail + ["is_avoidable"]].dropna()
    print(f"[model] rows before dropna: {len(ed_pat):,}  after: {len(model_df):,}  "
          f"avoidable rate: {model_df['is_avoidable'].mean():.3f}  features: {len(avail)}")
    _zero_var = [f for f in avail if model_df[f].std() == 0]
    if _zero_var:
        print(f"[model] WARNING — zero-variance features (will not help): {_zero_var}")
    return model_df, avail


def _classification_metrics(y_true, y_prob, y_pred):
    """Return auc, recall, precision, f1, specificity from arrays."""
    from sklearn.metrics import (
        confusion_matrix, f1_score, mean_squared_error,
        precision_score, recall_score, roc_auc_score,
    )
    auc    = float(roc_auc_score(y_true, y_prob))
    recall = float(recall_score(y_true, y_pred))
    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred))
    cm     = confusion_matrix(y_true, y_pred)
    spec   = float(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
    train_mse = None  # filled by caller
    test_mse  = None
    return auc, recall, prec, f1, spec


def _importance_table(features, importances):
    return [
        {"feature": f, "importance": round(float(imp), 6)}
        for f, imp in sorted(
            zip(features, importances), key=lambda x: -x[1]
        )
    ]


def _permutation_p(clf, X_sc, y, n_perm: int, random_state: int):
    """Permutation test p-value: fraction of shuffled AUCs >= observed AUC."""
    from sklearn.metrics import roc_auc_score
    obs = roc_auc_score(y, clf.predict_proba(X_sc)[:, 1])
    rng = np.random.RandomState(random_state)
    count = sum(
        roc_auc_score(rng.permutation(y), clf.predict_proba(X_sc)[:, 1]) >= obs
        for _ in range(n_perm)
    )
    return float(count / n_perm)


def _sens_and_str(recall, avoidable_rate, divert_rates):
    sens = {
        k: round(recall * avoidable_rate * v * 100, 2)
        for k, v in divert_rates.items()
    }
    pct_str = "/".join(
        str(int(divert_rates[s] * 100))
        for s in ("conservative", "base", "optimistic")
    )
    return sens, pct_str


# ── Random Forest ──────────────────────────────────────────────────────────────

def intervention2_rf(
    ed: pd.DataFrame,
    pat_feat: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = None,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str = "balanced",
    test_size: float = 0.2,
    random_state: int = 42,
    n_permutations: int = 200,
    divert_rates: dict = None,
) -> dict:
    """
    Random Forest triage diversion classifier.

    Parameters
    ----------
    ed, pat_feat      Same DataFrames passed to intervention2().
    n_estimators      Number of trees (default 300).
    max_depth         Max tree depth; None = grow until pure (default None).
    min_samples_leaf  Min samples required at a leaf (default 1).
                      Increase (e.g. 5–20) to regularise / reduce overfitting.
    max_features      Features considered per split: 'sqrt', 'log2', or float
                      fraction (default 'sqrt').
    class_weight      'balanced', 'balanced_subsample', or None (default 'balanced').
    test_size         Held-out fraction (default 0.2).
    random_state      RNG seed (default 42).
    n_permutations    Permutation iterations for overall p-value (default 200).
    divert_rates      conservative/base/optimistic diversion fractions.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if divert_rates is None:
        divert_rates = dict(_DEFAULT_DIVERT_RATES)

    model_df, avail = _build_model_df(ed, pat_feat)
    X = model_df[avail].values
    y = model_df["is_avoidable"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # RF doesn't require scaling but we keep it consistent for pipeline parity
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train)

    train_prob = clf.predict_proba(X_train_sc)[:, 1]
    test_prob  = clf.predict_proba(X_test_sc)[:, 1]
    train_mse  = float(mean_squared_error(y_train, train_prob))
    test_mse   = float(mean_squared_error(y_test,  test_prob))

    y_pred = clf.predict(X_test_sc)
    auc, recall, prec, f1, spec = _classification_metrics(y_test, test_prob, y_pred)

    # p-value via permutation test on full data
    X_all_sc = scaler.fit_transform(X)
    clf_full = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        class_weight=class_weight, random_state=random_state, n_jobs=-1,
    )
    clf_full.fit(X_all_sc, y)
    p_val = _permutation_p(clf_full, X_all_sc, y, n_permutations, random_state)

    imp_table = _importance_table(avail, clf_full.feature_importances_)
    sens, pct_str = _sens_and_str(recall, ed["is_avoidable"].mean(), divert_rates)

    return {
        "train_mse"                  : round(train_mse, 6),
        "test_mse"                   : round(test_mse, 6),
        "n_model"                    : len(model_df),
        "n_train"                    : len(y_train),
        "n_test"                     : len(y_test),
        "features_used"              : TRIAGE_FEATURES,
        "auc_roc"                    : round(auc, 4),
        "sensitivity_recall"         : round(recall, 4),
        "specificity"                : round(spec, 4),
        "precision"                  : round(prec, 4),
        "f1_score"                   : round(f1, 4),
        "p_value"                    : p_val,
        "correlation_coeff_r"        : round(auc * 2 - 1, 4),
        "feature_importance_table"   : imp_table,
        "projected_ed_reduction_pct" : sens,
        "sensitivity_assumption"     : (
            f"{pct_str} % of triage-flagged avoidable visits are successfully diverted"
        ),
        "hyperparameters"            : {
            "model"           : "RandomForest",
            "n_estimators"    : n_estimators,
            "max_depth"       : max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features"    : max_features,
            "class_weight"    : class_weight,
            "test_size"       : test_size,
            "random_state"    : random_state,
            "n_permutations"  : n_permutations,
        },
    }


# ── XGBoost ────────────────────────────────────────────────────────────────────

def intervention2_xg(
    ed: pd.DataFrame,
    pat_feat: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_permutations: int = 200,
    divert_rates: dict = None,
) -> dict:
    """
    XGBoost triage diversion classifier.

    Parameters
    ----------
    ed, pat_feat        Same DataFrames passed to intervention2().
    n_estimators        Number of boosting rounds (default 300).
    max_depth           Max tree depth per round (default 6; lower = less overfit).
    learning_rate       Step size shrinkage / eta (default 0.1).
    subsample           Row sampling ratio per tree (default 0.8).
    colsample_bytree    Feature sampling ratio per tree (default 0.8).
    min_child_weight    Min sum of instance weight in a child (default 1).
                        Higher = more conservative; reduces overfit on small classes.
    gamma               Min loss reduction to split a node (default 0.0).
                        Increase for stronger pruning.
    reg_alpha           L1 regularisation on weights (default 0.0).
    reg_lambda          L2 regularisation on weights (default 1.0).
    scale_pos_weight    Ratio of negative to positive samples for class imbalance.
                        None = auto-computed from training labels (recommended).
    test_size           Held-out fraction (default 0.2).
    random_state        RNG seed (default 42).
    n_permutations      Permutation iterations for overall p-value (default 200).
    divert_rates        conservative/base/optimistic diversion fractions.

    Raises
    ------
    ImportError if xgboost is not installed.
    """
    if not _XGB_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        )

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if divert_rates is None:
        divert_rates = dict(_DEFAULT_DIVERT_RATES)

    model_df, avail = _build_model_df(ed, pat_feat)
    X = model_df[avail].values
    y = model_df["is_avoidable"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Diagnostics: catch inf/nan that survive dropna and break XGBoost silently
    _finite = np.isfinite(X_train_sc).all()
    _stds   = X_train_sc.std(axis=0)
    print(f"[xgb] X_train shape: {X_train_sc.shape}  all-finite: {_finite}  "
          f"min-std: {_stds.min():.4f}  max-std: {_stds.max():.4f}  "
          f"y_train balance: {y_train.mean():.3f}")
    if not _finite:
        bad_cols = [avail[i] for i, ok in enumerate(np.isfinite(X_train_sc).all(axis=0)) if not ok]
        print(f"[xgb] WARNING — non-finite values in: {bad_cols}")
        X_train_sc = np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_sc  = np.nan_to_num(X_test_sc,  nan=0.0, posinf=0.0, neginf=0.0)

    # Auto scale_pos_weight from training labels if not provided
    if scale_pos_weight is None:
        neg = float((y_train == 0).sum())
        pos = float((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    clf.fit(X_train_sc, y_train)

    train_prob = clf.predict_proba(X_train_sc)[:, 1]
    test_prob  = clf.predict_proba(X_test_sc)[:, 1]
    train_mse  = float(mean_squared_error(y_train, train_prob))
    test_mse   = float(mean_squared_error(y_test,  test_prob))

    y_pred = clf.predict(X_test_sc)
    auc, recall, prec, f1, spec = _classification_metrics(y_test, test_prob, y_pred)

    # p-value via permutation test on full data
    X_all_sc = scaler.fit_transform(X)
    clf_full = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight, random_state=random_state,
        eval_metric="logloss", verbosity=0, use_label_encoder=False,
    )
    clf_full.fit(X_all_sc, y)
    p_val = _permutation_p(clf_full, X_all_sc, y, n_permutations, random_state)

    imp_table = _importance_table(avail, clf_full.feature_importances_)
    sens, pct_str = _sens_and_str(recall, ed["is_avoidable"].mean(), divert_rates)

    return {
        "train_mse"                  : round(train_mse, 6),
        "test_mse"                   : round(test_mse, 6),
        "n_model"                    : len(model_df),
        "n_train"                    : len(y_train),
        "n_test"                     : len(y_test),
        "features_used"              : TRIAGE_FEATURES,
        "auc_roc"                    : round(auc, 4),
        "sensitivity_recall"         : round(recall, 4),
        "specificity"                : round(spec, 4),
        "precision"                  : round(prec, 4),
        "f1_score"                   : round(f1, 4),
        "p_value"                    : p_val,
        "correlation_coeff_r"        : round(auc * 2 - 1, 4),
        "feature_importance_table"   : imp_table,
        "projected_ed_reduction_pct" : sens,
        "sensitivity_assumption"     : (
            f"{pct_str} % of triage-flagged avoidable visits are successfully diverted"
        ),
        "hyperparameters"            : {
            "model"            : "XGBoost",
            "n_estimators"     : n_estimators,
            "max_depth"        : max_depth,
            "learning_rate"    : learning_rate,
            "subsample"        : subsample,
            "colsample_bytree" : colsample_bytree,
            "min_child_weight" : min_child_weight,
            "gamma"            : gamma,
            "reg_alpha"        : reg_alpha,
            "reg_lambda"       : reg_lambda,
            "scale_pos_weight" : scale_pos_weight,
            "test_size"        : test_size,
            "random_state"     : random_state,
            "n_permutations"   : n_permutations,
        },
    }
