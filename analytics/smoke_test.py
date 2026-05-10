"""Smoke test for the four-intervention modeling logic."""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, spearmanr, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
warnings.filterwarnings("ignore")

ROOT = Path(r"c:\Users\timch\OneDrive\桌面\DataFest2026")
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
MISSING = ["*Unspecified","*Unknown","*Not Applicable","*Deleted"]
def nm(df): return df.replace(MISSING, pd.NA)

print("Loading...")
patients  = nm(pd.read_csv(ROOT/"DATA/patients.csv", low_memory=False))
pat_feat  = pd.read_csv(ROOT/"DATA_CLEANED/patient_features.csv")
pcp_prov  = pd.read_csv(ROOT/"DATA_CLEANED/providers_pcp.csv")
diag      = pd.read_csv(ROOT/"DATA/diagnosis.csv", low_memory=False)

ENC_COLS = ["EncounterKey","PatientDurableKey","ProviderDurableKey",
            "IsEdVisit","PrimaryDiagnosisKey","Date","AdmitHour"]
enc_chunks = []
for chunk in pd.read_csv(ROOT/"DATA/encounters.csv", usecols=ENC_COLS,
                          chunksize=500_000, low_memory=False):
    enc_chunks.append(chunk)
encounters = pd.concat(enc_chunks, ignore_index=True)
print(f"  encounters: {len(encounters):,}")

# ── NYU classification ────────────────────────────────────────────────────────
NON_EMERGENT  = frozenset(["J00","J01","J02","J03","J04","J05","J06","J20","J21","J22",
    "H60","H61","H65","H66","H67","H10","H11","L00","L01","L02","L03","L04","L05","L08",
    "L20","L21","L22","L23","L24","L25","L30","L50","L70","L73","N30","N34","N39","N76",
    "N77","K00","K01","K02","K03","K04","K05","K06","K08","M54","M79","M25","M47","B34",
    "B99","Z00","Z01","Z02","Z11","Z12","Z13","Z23","Z71","Z76"])
EMERGENT_PC   = frozenset(["A08","A09","K52","K59","R11","R19","K29","K30","K31","R30",
    "R31","R32","R33","R35","M50","M51","M99","S13","S23","S33","S43","S53","S63","S93",
    "G43","G44","R51","L27","L29","T78","D50","I10","H72","H73","H74","J35","J36",
    "F40","F41","L03"])
EMERGENT_PREV = frozenset(["J45","J46","J40","J41","J42","J43","J44","I50","I11","I12",
    "I13","I16","E10","E11","E13","E16","J13","J14","J15","J18","E86","E87","L03","N10",
    "N11","N12","K09","K12","G40","G41","E78","D64"])
EMERGENT_TRUE = frozenset(["I21","I22","I20","I63","I64","G45","I46","I47","I48","I49",
    "I71","I26","J80","J81","J96","A40","A41","K35","K36","K37","K56","K25","K26","K57",
    "K92","O00","O08","O36","O60","O67","O72","N17","N18","N19","G00","G01","G02","G03",
    "G04","E10","E11","E13","T78","T36","T37","T38","T39","T40","T41","T42","T43","T44",
    "T45","T46","T47","T48","T49","T50","S02","S12","S22","S32","S42","S52","S62","S72",
    "S82","S92","S06","S14","S24","S34","T20","T21","T22","T23","T24","T25","T26","T27",
    "T28","T29","T30","T31","T67","T68","T69"])
AVOIDABLE_CATS     = {"Non-Emergent","Emergent/PC-Treatable","Emergent/Preventable"}
NOT_AVOIDABLE_CATS = {"Emergent/True-ED","Injury","Psychiatric","Alcohol/Drug"}

def classify_nyu_ed(icd10):
    if pd.isna(icd10) or not str(icd10).strip(): return "Unclassified"
    code = str(icd10).strip().upper().replace(".", "")
    p1, p3 = code[0], code[:3]
    if p1 == "S" or (p1 == "T" and p3 not in EMERGENT_TRUE): return "Injury"
    if p3 in {"F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"}: return "Alcohol/Drug"
    if p1 == "F": return "Psychiatric"
    if p3 in EMERGENT_TRUE: return "Emergent/True-ED"
    if p3 in EMERGENT_PREV: return "Emergent/Preventable"
    if p3 in EMERGENT_PC:   return "Emergent/PC-Treatable"
    if p3 in NON_EMERGENT:  return "Non-Emergent"
    return "Unclassified"

def avoidability(cat):
    if cat in AVOIDABLE_CATS:     return "Avoidable"
    if cat in NOT_AVOIDABLE_CATS: return "Not Avoidable"
    return "Unknown"

diag["nyu_ed_category"] = diag["DiagnosisValue"].map(classify_nyu_ed)
diag_nyu = diag[["DiagnosisKey","nyu_ed_category"]].drop_duplicates("DiagnosisKey")

# ── attach avoidability to encounters ─────────────────────────────────────────
encounters["PrimaryDiagnosisKey"] = pd.to_numeric(
    encounters["PrimaryDiagnosisKey"].replace(-1, pd.NA), errors="coerce")
ed_mask = encounters["IsEdVisit"] == 1
ed_enc  = (encounters.loc[ed_mask, ["EncounterKey","PrimaryDiagnosisKey"]]
           .merge(diag_nyu, left_on="PrimaryDiagnosisKey",
                  right_on="DiagnosisKey", how="left"))
ed_enc["ed_avoidability"] = ed_enc["nyu_ed_category"].fillna("Unclassified").map(avoidability)
encounters = encounters.merge(ed_enc[["EncounterKey","ed_avoidability"]],
                               on="EncounterKey", how="left")
encounters.loc[~ed_mask, "ed_avoidability"] = pd.NA

ed = encounters[
    (encounters["IsEdVisit"] == 1) &
    encounters["ed_avoidability"].notna() &
    (encounters["ed_avoidability"] != "Unknown")
].copy()
ed["is_avoidable"] = (ed["ed_avoidability"] == "Avoidable").astype(int)
ed["is_weekend"]   = pd.to_datetime(ed["Date"], errors="coerce").dt.dayofweek.isin([5,6]).astype(int)
ed["AdmitHour"]    = pd.to_numeric(ed["AdmitHour"], errors="coerce")

n_ed_raw        = int((encounters["IsEdVisit"] == 1).sum())
n_ed_classified = len(ed)
n_avoidable     = int(ed["is_avoidable"].sum())
n_not_avoidable = int((ed["ed_avoidability"] == "Not Avoidable").sum())
n_unknown       = n_ed_raw - n_ed_classified
avoidable_rate  = n_avoidable / n_ed_classified

# ── patient features ──────────────────────────────────────────────────────────
mychart_map = (patients[["DurableKey","MyChartStatus"]]
               .assign(mychart_active=lambda d: (d["MyChartStatus"]=="Activated").astype(int))
               [["DurableKey","mychart_active"]])
pat_feat = pat_feat.merge(mychart_map, left_on="PatientDurableKey",
                           right_on="DurableKey", how="left")
pat_feat["mychart_active"] = pat_feat["mychart_active"].fillna(0).astype(int)

_pcp_keys = set(pcp_prov["DurableKey"].values)
_pcp_enc  = (encounters.groupby("PatientDurableKey")
             .apply(lambda g: pd.Series({
                 "total_enc": len(g),
                 "pcp_enc"  : g["ProviderDurableKey"].isin(_pcp_keys).sum()}))
             .reset_index())
_pcp_enc["pcp_ratio"] = _pcp_enc["pcp_enc"] / _pcp_enc["total_enc"].clip(lower=1)
pat_feat = pat_feat.merge(_pcp_enc[["PatientDurableKey","pcp_ratio"]],
                           on="PatientDurableKey", how="left")
pat_feat["pcp_ratio"] = pat_feat["pcp_ratio"].fillna(0)

print("Running models...")
results = {}

# ── INT 1 ─────────────────────────────────────────────────────────────────────
_wkd  = ed[ed["is_weekend"] == 1]
_wkdy = ed[ed["is_weekend"] == 0]
_ct   = pd.crosstab(ed["is_weekend"], ed["is_avoidable"])
_chi2_i1, _p_i1, _, _ = chi2_contingency(_ct)
_cramers_v = float(np.sqrt(_chi2_i1 / (len(ed) * (min(_ct.shape) - 1))))
_r_i1, _pr_i1 = pointbiserialr(ed["is_weekend"], ed["is_avoidable"])
_n_avoid_wkd  = int(_wkd["is_avoidable"].sum())
_sens_i1 = {k: round(_n_avoid_wkd * v / n_ed_classified * 100, 2)
            for k, v in {"conservative":0.25,"base":0.45,"optimistic":0.65}.items()}

results["intervention_1_extended_hours"] = {
    "name"                       : "Extended / Weekend Clinic Hours",
    "hypothesis"                 : "Avoidable ED visits spike on weekends due to PCP unavailability",
    "n_classified_ed"            : n_ed_classified,
    "n_weekend_ed"               : len(_wkd),
    "n_avoidable_weekend"        : _n_avoid_wkd,
    "avoidable_rate_weekend_pct" : round(float(_wkd["is_avoidable"].mean()) * 100, 2),
    "avoidable_rate_weekday_pct" : round(float(_wkdy["is_avoidable"].mean()) * 100, 2),
    "rate_difference_pct"        : round((float(_wkd["is_avoidable"].mean()) - float(_wkdy["is_avoidable"].mean())) * 100, 2),
    "chi2_statistic"             : round(_chi2_i1, 4),
    "p_value"                    : float(_p_i1),
    "cramers_v"                  : round(_cramers_v, 4),
    "correlation_coeff_r"        : round(float(_r_i1), 4),
    "correlation_p_value"        : float(_pr_i1),
    "projected_ed_reduction_pct" : _sens_i1,
    "sensitivity_assumption"     : "% of avoidable weekend ED visits diverted to extended-hours clinics",
}
print(f"INT1 done — chi2={_chi2_i1:.2f}, p={_p_i1:.2e}, V={_cramers_v:.4f}")

# ── INT 2 ─────────────────────────────────────────────────────────────────────
TRIAGE_FEATS = ["is_weekend","is_offhours","approx_age","is_female",
                "sdoh_unique_domains","sdoh_financial_resource_strain",
                "sdoh_housing_stability","sdoh_transportation_needs",
                "encounters_per_month","pcp_ratio","mychart_active"]

_ed_pat = ed.merge(pat_feat[["PatientDurableKey","approx_age","is_female",
    "sdoh_unique_domains","sdoh_financial_resource_strain","sdoh_housing_stability",
    "sdoh_transportation_needs","encounters_per_month","pcp_ratio","mychart_active"]],
    on="PatientDurableKey", how="inner")
_ed_pat["is_offhours"] = (
    _ed_pat["is_weekend"] |
    _ed_pat["AdmitHour"].between(0, 5) |
    _ed_pat["AdmitHour"].between(22, 23)
).astype(int)

_mdf = _ed_pat[TRIAGE_FEATS + ["is_avoidable"]].dropna()
_X   = StandardScaler().fit_transform(_mdf[TRIAGE_FEATS].values)
_y   = _mdf["is_avoidable"].values
_clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced").fit(_X, _y)
_yp  = _clf.predict(_X)
_ypr = _clf.predict_proba(_X)[:, 1]
_auc = float(roc_auc_score(_y, _ypr))
_rec = float(recall_score(_y, _yp))
_cm  = confusion_matrix(_y, _yp)
_spc = float(_cm[0,0] / (_cm[0,0] + _cm[0,1]))

# bootstrap SEs for coefficient p-values
from sklearn.utils import resample as _resample
_N_BOOT = 300
_n_samp = min(len(_mdf), 5000)   # cap for speed
_boot_coefs = []
for i in range(_N_BOOT):
    _idx = np.random.RandomState(i).choice(len(_mdf), _n_samp, replace=True)
    _Xb, _yb = _X[_idx], _y[_idx]
    _c = LogisticRegression(max_iter=200, random_state=i, class_weight="balanced").fit(_Xb, _yb).coef_[0]
    _boot_coefs.append(_c)
_boot_coefs = np.array(_boot_coefs)
_coef_se = _boot_coefs.std(axis=0)
_coef_z  = _clf.coef_[0] / np.where(_coef_se == 0, np.nan, _coef_se)
_coef_p  = 2 * (1 - stats.norm.cdf(np.abs(_coef_z)))

_coef_table = [{"feature": f, "coefficient": round(float(c),4),
                "std_error": round(float(se),4), "z_score": round(float(z),4),
                "p_value": round(float(p),4)}
               for f,c,se,z,p in zip(TRIAGE_FEATS,_clf.coef_[0],_coef_se,_coef_z,_coef_p)]

_null_acc  = max(_y.mean(), 1 - _y.mean())
_G2_i2     = 2 * len(_mdf) * max(0, float(_clf.score(_X, _y)) - _null_acc)
_p_i2      = float(stats.chi2.sf(_G2_i2, df=len(TRIAGE_FEATS)))
_divert_b  = _rec * avoidable_rate
_sens_i2   = {"conservative": round(_divert_b*0.50*100,2),
              "base":          round(_divert_b*0.70*100,2),
              "optimistic":    round(_divert_b*0.90*100,2)}

results["intervention_2_triage_diversion"] = {
    "name"                       : "Upgraded Triage with Licensed Staff",
    "hypothesis"                 : "Patient features at triage predict avoidable ED use without diagnosis",
    "n_model"                    : len(_mdf),
    "features_used"              : TRIAGE_FEATS,
    "auc_roc"                    : round(_auc, 4),
    "sensitivity_recall"         : round(_rec, 4),
    "specificity"                : round(_spc, 4),
    "p_value"                    : _p_i2,
    "correlation_coeff_r"        : round(_auc * 2 - 1, 4),
    "coefficient_table"          : _coef_table,
    "projected_ed_reduction_pct" : _sens_i2,
    "sensitivity_assumption"     : "50/70/90% of triage-flagged avoidable visits are diverted",
}
print(f"INT2 done — AUC={_auc:.4f}, recall={_rec:.4f}, spec={_spc:.4f}")

# ── INT 3 ─────────────────────────────────────────────────────────────────────
_mc_on  = pat_feat[pat_feat["mychart_active"]==1]["ed_ratio"].dropna()
_mc_off = pat_feat[pat_feat["mychart_active"]==0]["ed_ratio"].dropna()
_u_i3, _p_i3   = mannwhitneyu(_mc_on, _mc_off, alternative="two-sided")
_r_i3, _pr_i3  = pearsonr(pat_feat["mychart_active"].dropna(),
                            pat_feat.loc[pat_feat["mychart_active"].notna(),"ed_ratio"].fillna(0))
_pool_sd = np.sqrt((_mc_on.std()**2 + _mc_off.std()**2) / 2)
_d_i3    = float((_mc_on.mean() - _mc_off.mean()) / _pool_sd)
_diff_pct = (_mc_on.mean() - _mc_off.mean()) / _mc_off.mean() * 100
_n_unact  = int((pat_feat["mychart_active"]==0).sum())
_ed_eff   = _mc_on.mean() - _mc_off.mean()
_sens_i3  = {k: round(abs(_ed_eff)*_n_unact*v/len(pat_feat)*100,2)
             for k,v in {"conservative":0.10,"base":0.20,"optimistic":0.35}.items()}

results["intervention_3_mychart_sdoh"] = {
    "name"                        : "MyChart Activation + SDOH Resource Referral",
    "hypothesis"                  : "MyChart activation is associated with substantially lower ED utilization",
    "n_activated"                 : len(_mc_on),
    "n_not_activated"             : len(_mc_off),
    "ed_ratio_activated_mean"     : round(float(_mc_on.mean()),4),
    "ed_ratio_not_activated_mean" : round(float(_mc_off.mean()),4),
    "ed_ratio_diff_pct"           : round(float(_diff_pct),2),
    "cohens_d"                    : round(_d_i3,4),
    "mann_whitney_u"              : round(float(_u_i3),2),
    "p_value"                     : float(_p_i3),
    "correlation_coeff_r"         : round(float(_r_i3),4),
    "correlation_p_value"         : float(_pr_i3),
    "projected_ed_reduction_pct"  : _sens_i3,
    "sensitivity_assumption"      : "% of currently unactivated patients who gain MyChart access",
}
print(f"INT3 done — p={_p_i3:.2e}, r={_r_i3:.4f}, d={_d_i3:.4f}, diff={_diff_pct:.1f}%")

# ── INT 4 ─────────────────────────────────────────────────────────────────────
_pcp_df = pat_feat[["pcp_ratio","ed_ratio"]].dropna()
_r_i4,  _pr_i4   = pearsonr(_pcp_df["pcp_ratio"], _pcp_df["ed_ratio"])
_rho_i4, _prho_i4 = spearmanr(_pcp_df["pcp_ratio"], _pcp_df["ed_ratio"])
_slope, _, _, _p_ols, _se_slope = stats.linregress(_pcp_df["pcp_ratio"], _pcp_df["ed_ratio"])

_prov_load = (encounters.groupby("ProviderDurableKey")["EncounterKey"]
              .count().rename("annual_enc").reset_index())
_p25_load  = _prov_load["annual_enc"].quantile(0.25)
_n_idle    = int((_prov_load["annual_enc"] <= _p25_load).sum())
_med_load  = float(_prov_load["annual_enc"].median())
_sens_i4 = {}
for k,rate in {"conservative":0.15,"base":0.30,"optimistic":0.50}.items():
    _new_pcp_enc   = _n_idle * rate * _med_load * 0.5
    _delta_pcp_r   = _new_pcp_enc / max(len(encounters), 1)
    _delta_ed_r    = _slope * _delta_pcp_r
    _sens_i4[k]    = round(abs(_delta_ed_r)*100, 2)

results["intervention_4_specialist_reallocation"] = {
    "name"                        : "Idle Specialist Reallocation as Part-time PCP",
    "hypothesis"                  : "Higher PCP encounter ratio negatively correlates with ED utilization",
    "n_patients"                  : len(_pcp_df),
    "pearson_r"                   : round(float(_r_i4),4),
    "pearson_p_value"             : float(_pr_i4),
    "spearman_rho"                : round(float(_rho_i4),4),
    "spearman_p_value"            : float(_prho_i4),
    "correlation_coeff_r"         : round(float(_r_i4),4),
    "ols_slope"                   : round(float(_slope),6),
    "ols_slope_se"                : round(float(_se_slope),6),
    "ols_p_value"                 : float(_p_ols),
    "p_value"                     : float(_pr_i4),
    "n_idle_providers"            : _n_idle,
    "idle_threshold_p25_enc"      : round(float(_p25_load),1),
    "projected_ed_reduction_pct"  : _sens_i4,
    "sensitivity_assumption"      : "% of idle (<=P25 load) providers redeployed as 0.5 FTE PCP",
}
print(f"INT4 done — r={_r_i4:.4f}, p={_pr_i4:.2e}, slope={_slope:.6f}, idle={_n_idle}")

# ── OVERALL ───────────────────────────────────────────────────────────────────
_ps = [_p_i1, _p_i2, float(_p_i3), float(_pr_i4)]
_fisher_chi2 = -2 * sum(np.log([max(p, 1e-300) for p in _ps]))
_fisher_p    = float(stats.chi2.sf(_fisher_chi2, df=2*len(_ps)))

_multi_df = pat_feat[["ed_ratio","mychart_active","pcp_ratio",
                       "sdoh_unique_domains","encounters_per_month"]].dropna()
_mX = _multi_df[["mychart_active","pcp_ratio","sdoh_unique_domains","encounters_per_month"]].values
_my = _multi_df["ed_ratio"].values
_corr_mat  = np.corrcoef(np.column_stack([_mX, _my]).T)
_r_with_y  = _corr_mat[:-1, -1]
_R_sq      = float(np.dot(_r_with_y,
    np.linalg.lstsq(_corr_mat[:-1,:-1], _r_with_y, rcond=None)[0]))
_R_sq      = max(0.0, min(1.0, _R_sq))
_R_overall = float(np.sqrt(_R_sq))

_get_proj = lambda key, scenario: results[key]["projected_ed_reduction_pct"][scenario]
_keys = ["intervention_1_extended_hours","intervention_2_triage_diversion",
         "intervention_3_mychart_sdoh","intervention_4_specialist_reallocation"]
results["overall"] = {
    "n_ed_encounters_raw"             : n_ed_raw,
    "n_ed_classified"                 : n_ed_classified,
    "n_ed_unknown_excluded"           : n_unknown,
    "unknown_exclusion_pct"           : round(n_unknown / n_ed_raw * 100, 2),
    "avoidable_pct_of_classified"     : round(avoidable_rate * 100, 2),
    "not_avoidable_pct_of_classified" : round(n_not_avoidable / n_ed_classified * 100, 2),
    "fisher_combined_chi2"            : round(_fisher_chi2, 4),
    "fisher_combined_p_value"         : _fisher_p,
    "multiple_R"                      : round(_R_overall, 4),
    "multiple_R_squared"              : round(_R_sq, 4),
    "projected_combined_reduction_pct": {
        s: round(min(sum(_get_proj(k, s) for k in _keys), 100), 2)
        for s in ["conservative","base","optimistic"]
    },
    "note": (
        "Combined reduction is additive upper-bound assuming non-overlapping patient "
        "pools. True effect will be lower due to partial overlap. "
        "Unknown-class encounters excluded from all models."
    ),
}
print(f"OVERALL — Fisher p={_fisher_p:.2e}, R={_R_overall:.4f}, R2={_R_sq:.4f}")

# ── write outputs ─────────────────────────────────────────────────────────────
json_path = OUT / "intervention_model_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"JSON → {json_path}")

csv_rows = []
for k in _keys:
    v = results[k]
    csv_rows.append({
        "intervention"                         : k,
        "name"                                 : v["name"],
        "p_value"                              : v["p_value"],
        "correlation_coeff_r"                  : v["correlation_coeff_r"],
        "effect_size"                          : (v.get("cramers_v") or v.get("auc_roc")
                                                   or v.get("cohens_d") or v.get("ols_slope")),
        "effect_size_metric"                   : ("cramers_v"  if "cramers_v"  in v else
                                                   "auc_roc"    if "auc_roc"    in v else
                                                   "cohens_d"   if "cohens_d"   in v else "ols_slope"),
        "sensitivity_recall"                   : v.get("sensitivity_recall"),
        "specificity"                          : v.get("specificity"),
        "projected_reduction_conservative_pct" : v["projected_ed_reduction_pct"]["conservative"],
        "projected_reduction_base_pct"         : v["projected_ed_reduction_pct"]["base"],
        "projected_reduction_optimistic_pct"   : v["projected_ed_reduction_pct"]["optimistic"],
        "n"                                    : (v.get("n_classified_ed") or v.get("n_model")
                                                   or v.get("n_activated") or v.get("n_patients")),
    })
ov = results["overall"]
csv_rows.append({
    "intervention"                         : "overall",
    "name"                                 : "Combined Four-Intervention Bundle",
    "p_value"                              : ov["fisher_combined_p_value"],
    "correlation_coeff_r"                  : ov["multiple_R"],
    "effect_size"                          : ov["multiple_R_squared"],
    "effect_size_metric"                   : "multiple_R_squared",
    "sensitivity_recall"                   : None,
    "specificity"                          : None,
    "projected_reduction_conservative_pct" : ov["projected_combined_reduction_pct"]["conservative"],
    "projected_reduction_base_pct"         : ov["projected_combined_reduction_pct"]["base"],
    "projected_reduction_optimistic_pct"   : ov["projected_combined_reduction_pct"]["optimistic"],
    "n"                                    : n_ed_classified,
})
csv_path = OUT / "intervention_model_results.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"CSV  → {csv_path}")
print("Done.")
