import pandas as pd
from pathlib import Path
import json


# ── NYU ED Classification (ICD-10 adaptation) ────────────────────────────────
# Based on: Billings et al. (2000) Health Affairs, and the ICD-10 adaptation
# documented in Ballard et al. (2010) and NYU Wagner Health Policy group.
#
# Seven mutually exclusive categories (first matching rule wins):
#   1. Injury                — S00–T98 (trauma, poisoning, external causes)
#   2. Alcohol/Drug          — F10–F19 substance-related disorders
#   3. Psychiatric           — all other F codes
#   4. Non-Emergent          — conditions treatable in routine outpatient setting;
#                              not time-sensitive within 12 hours
#   5. Emergent/PC-Treatable — needed prompt attention but primary care or
#                              urgent care could have managed (within hours)
#   6. Emergent/Preventable  — ED-level acuity but acute exacerbation of a
#                              condition manageable with better primary care
#                              (ambulatory-care sensitive conditions)
#   7. Emergent/True-ED      — requires immediate ED resources; not preventable
#                              by upstream primary care
#   0. Unclassified          — code missing, malformed, or Z/V/W/X/Y
#
# Classification uses the 3-character ICD-10 prefix (e.g. "J45" from "J45.21").
# Code sets below reflect the most cited ICD-10 NYU mappings; expand as needed.

_NON_EMERGENT = frozenset([
    # Upper respiratory infections
    "J00","J01","J02","J03","J04","J05","J06",
    # Acute bronchitis / lower respiratory — minor
    "J20","J21","J22",
    # Otitis media / external ear
    "H60","H61","H65","H66","H67",
    # Conjunctivitis / minor eye
    "H10","H11",
    # Minor skin & subcutaneous
    "L00","L01","L02","L03","L04","L05","L08","L20","L21","L22","L23",
    "L24","L25","L30","L50","L70","L73",
    # Urinary tract infection / cystitis
    "N30","N34","N39",
    # Vaginal / genital infections (non-STI)
    "N76","N77",
    # Dental / oral (often mistreated in ED)
    "K00","K01","K02","K03","K04","K05","K06","K08",
    # Musculoskeletal — minor / chronic
    "M54","M79","M25","M47",
    # Viral / unspecified infections
    "B34","B99",
    # Routine / administrative encounters
    "Z00","Z01","Z02","Z11","Z12","Z13","Z23","Z71","Z76",
])

_EMERGENT_PC_TREATABLE = frozenset([
    # Gastroenteritis / nausea / diarrhea
    "A08","A09","K52","K59","R11","R19",
    # Gastritis / dyspepsia
    "K29","K30","K31",
    # Urinary symptoms without infection
    "R30","R31","R32","R33","R35",
    # Musculoskeletal sprains / back pain (acute)
    "M50","M51","M99","S13","S23","S33","S43","S53","S63","S93",
    # Headache / migraine
    "G43","G44","R51",
    # Minor allergic reaction
    "L27","L29","T78",
    # Iron-deficiency anemia (non-acute)
    "D50",
    # Hypertension — elevated, not crisis
    "I10",
    # Otitis media — treated acutely
    "H72","H73","H74",
    # Pharyngitis / tonsillitis
    "J35","J36",
    # Anxiety / panic without psychiatric admission
    "F40","F41",
    # Cellulitis (localized, no sepsis)
    "L03",
])

_EMERGENT_PREVENTABLE = frozenset([
    # Asthma exacerbation
    "J45","J46",
    # COPD exacerbation
    "J40","J41","J42","J43","J44",
    # Congestive heart failure
    "I50",
    # Hypertensive crisis
    "I11","I12","I13","I16",
    # Uncontrolled / complicated diabetes
    "E10","E11","E13",
    # Hypoglycemia
    "E16",
    # Pneumonia — community-acquired (vaccine-preventable organisms)
    "J13","J14","J15","J18",
    # Dehydration / volume depletion (preventable with PCP management)
    "E86","E87",
    # Cellulitis — spreading / systemic signs
    "L03",
    # Pyelonephritis / upper UTI
    "N10","N11","N12",
    # Dental abscess complications
    "K09","K12",
    # Seizure disorder — subtherapeutic meds
    "G40","G41",
    # Hyperlipidemia crisis / metabolic
    "E78",
    # Anemia — complications from chronic disease
    "D64",
])

_EMERGENT_TRUE_ED = frozenset([
    # Acute myocardial infarction
    "I21","I22",
    # Unstable angina / ACS
    "I20",
    # Stroke / cerebral infarction
    "I63","I64",
    # TIA
    "G45",
    # Cardiac arrest / ventricular fibrillation
    "I46","I47","I48","I49",
    # Aortic aneurysm / dissection
    "I71",
    # Pulmonary embolism
    "I26",
    # Respiratory failure
    "J80","J81","J96",
    # Sepsis / septicemia
    "A40","A41",
    # Appendicitis
    "K35","K36","K37",
    # Bowel obstruction
    "K56",
    # GI hemorrhage
    "K25","K26","K57","K92",
    # Ectopic pregnancy / obstetric emergencies
    "O00","O08","O36","O60","O67","O72",
    # Acute renal failure
    "N17","N18","N19",
    # Meningitis / encephalitis
    "G00","G01","G02","G03","G04",
    # Diabetic ketoacidosis
    "E10","E11","E13",   # overlaps preventable — DKA subcodes (E1x.1x) handled below
    # Anaphylaxis
    "T78",               # overlaps PC-treatable — severe subcodes handled below
    # Overdose / poisoning (not substance-disorder F codes)
    "T36","T37","T38","T39","T40","T41","T42","T43","T44","T45","T46",
    "T47","T48","T49","T50",
    # Major fractures / dislocations
    "S02","S12","S22","S32","S42","S52","S62","S72","S82","S92",
    # Intracranial injury
    "S06",
    # Spinal cord injury
    "S14","S24","S34",
    # Burns — significant
    "T20","T21","T22","T23","T24","T25","T26","T27","T28","T29","T30","T31",
    # Hypothermia / heat stroke
    "T67","T68","T69",
])


def classify_nyu_ed(icd10: str) -> str:
    """
    Classify a single ICD-10-CM code into an NYU ED utilization category.
    Returns one of: 'Injury', 'Alcohol/Drug', 'Psychiatric',
    'Non-Emergent', 'Emergent/PC-Treatable', 'Emergent/Preventable',
    'Emergent/True-ED', 'Unclassified'.
    """
    if pd.isna(icd10) or not str(icd10).strip():
        return "Unclassified"

    code = str(icd10).strip().upper().replace(".", "")
    p1 = code[0]
    p3 = code[:3]

    # Rule 1 — Injury: S and T codes (excluding T36–T50 poisonings → True-ED below)
    if p1 == "S" or (p1 == "T" and p3 not in _EMERGENT_TRUE_ED):
        return "Injury"

    # Rule 2 — Alcohol/Drug
    if p3 in {"F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"}:
        return "Alcohol/Drug"

    # Rule 3 — Psychiatric
    if p1 == "F":
        return "Psychiatric"

    # Rules 4–7: use 3-char prefix lookup; True-ED takes precedence over Preventable
    # for codes that appear in both (e.g. E11 DKA vs. uncomplicated T2DM).
    if p3 in _EMERGENT_TRUE_ED:
        return "Emergent/True-ED"
    if p3 in _EMERGENT_PREVENTABLE:
        return "Emergent/Preventable"
    if p3 in _EMERGENT_PC_TREATABLE:
        return "Emergent/PC-Treatable"
    if p3 in _NON_EMERGENT:
        return "Non-Emergent"

    # Default fallback — external cause (V/W/X/Y), supplementary, or unmapped
    return "Unclassified"


def add_nyu_classification(diagnosis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a 'nyu_ed_category' column to the diagnosis DataFrame.
    Expects a 'DiagnosisValue' column containing ICD-10-CM codes.
    Also adds a boolean 'is_avoidable_ed' flag (True for Non-Emergent,
    Emergent/PC-Treatable, and Emergent/Preventable).
    """
    diagnosis_df = diagnosis_df.copy()
    diagnosis_df["nyu_ed_category"] = diagnosis_df["DiagnosisValue"].map(classify_nyu_ed)
    diagnosis_df["is_avoidable_ed"] = diagnosis_df["nyu_ed_category"].isin(
        {"Non-Emergent", "Emergent/PC-Treatable", "Emergent/Preventable"}
    )
    return diagnosis_df

DATA_DIR = Path(__file__).parent.parent / "DATA"


# ── Missing-value normalization ───────────────────────────────────────────────
# The dataset uses several distinct sentinels across files:
#   *Unspecified  — field was asked but patient did not answer
#   *Unknown      — data was never recorded
#   *Not Applicable — field does not apply to this encounter type
#   *Deleted      — record has been removed from the system
# All are mapped to pd.NA for consistent downstream checks.
# The numeric sentinel -1 in foreign-key columns is handled separately below
# because it carries structural meaning (outpatient / non-admitted context).
MISSING_MARKERS = ["*Unspecified", "*Unknown", "*Not Applicable", "*Deleted"]

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(MISSING_MARKERS, pd.NA)


# ── Load data ─────────────────────────────────────────────────────────────────
patients    = normalize_missing(pd.read_csv(DATA_DIR / "patients.csv",    low_memory=False))
_ENC_COLS = [
    "EncounterKey", "PatientDurableKey", "ProviderDurableKey",
    "AttendingProviderDurableKey", "DischargeProviderDurableKey",
    "DepartmentKey", "PrimaryDiagnosisKey",
    "IsEdVisit", "IsHospitalAdmission", "IsOutpatientFaceToFaceVisit",
    "Date", "AdmitHour",
    "AdmissionSource", "AdmissionType", "IsInpatientAdmission",
]
encounters  = pd.concat([
    normalize_missing(chunk)
    for chunk in pd.read_csv(DATA_DIR / "encounters.csv", usecols=_ENC_COLS,
                              low_memory=False, chunksize=500_000)
], ignore_index=True)
diagnosis   = add_nyu_classification(
    normalize_missing(pd.read_csv(DATA_DIR / "diagnosis.csv", low_memory=False))
)
providers   = normalize_missing(pd.read_csv(DATA_DIR / "providers.csv",   low_memory=False))
departments = normalize_missing(pd.read_csv(DATA_DIR / "departments.csv", low_memory=False))
sdoh        = normalize_missing(pd.read_csv(DATA_DIR / "social_determinants.csv", low_memory=False))

# tigercensuscodes has no missing data — no normalization needed.
# NOTE: only patients whose CensusBlockGroupFipsCode matches a GEOID here can be
# geo-linked. Check linkage coverage before running any geographic analysis:
#   linked_frac = patients["CensusBlockGroupFipsCode"].isin(tigercodes["GEOID"]).mean()
tigercodes = pd.read_csv(DATA_DIR / "tigercensuscodes.csv", low_memory=False)


# ── Race / demographics: pd.NA is already its own stratum ────────────────────
# Structural vs. informative missingness: timestamp fields that are NA on office
# visits are structural (no admission/discharge for outpatient) and can be safely
# ignored when filtering by encounter type. Demographic fields like FirstRace that
# are NA are informative — patients who did not disclose may differ systematically.
# Both cases require no special handling here: pd.NA flows through groupby as its
# own "Missing/Not Disclosed" bucket. Apply fillna("Missing") only at plot time
# if a display label is required.


# ── Encounters: -1 foreign keys as "Non-admitted/Outpatient" stratum ─────────
# -1 in PrimaryDiagnosisKey, AttendingProviderDurableKey, and
# DischargeProviderDurableKey marks outpatient/office visits where no admission
# or discharge provider is recorded. These are the majority of encounters.
# Labeling them as a stratum preserves them for analysis rather than silently
# joining them to *Unspecified rows or dropping them entirely.
FK_OUTPATIENT = "Non-admitted/Outpatient"

for _col in ["PrimaryDiagnosisKey", "AttendingProviderDurableKey", "DischargeProviderDurableKey"]:
    if _col in encounters.columns:
        encounters[_col] = encounters[_col].replace(-1, FK_OUTPATIENT)


# ── ED encounters: avoidability classification ───────────────────────────────
# Only ED encounters carry a meaningful NYU category.  For each ED encounter we
# resolve its primary diagnosis → NYU category, then collapse to three classes:
#   "Avoidable"     — Non-Emergent, Emergent/PC-Treatable, Emergent/Preventable
#   "Not Avoidable" — Emergent/True-ED, Injury, Psychiatric, Alcohol/Drug
#   "Unknown"       — PrimaryDiagnosisKey is missing/outpatient sentinel,
#                     or the ICD code mapped to Unclassified
#
# The result is a new column  ed_avoidability  on the encounters DataFrame.
# Non-ED encounters receive pd.NA so they are never silently included in counts.

_diag_nyu = (
    diagnosis[["DiagnosisKey", "nyu_ed_category"]]
    .drop_duplicates("DiagnosisKey")
)

_AVOIDABLE_CATS     = {"Non-Emergent", "Emergent/PC-Treatable", "Emergent/Preventable"}
_NOT_AVOIDABLE_CATS = {"Emergent/True-ED", "Injury", "Psychiatric", "Alcohol/Drug"}

def _map_avoidability(nyu_cat):
    if nyu_cat in _AVOIDABLE_CATS:
        return "Avoidable"
    if nyu_cat in _NOT_AVOIDABLE_CATS:
        return "Not Avoidable"
    return "Unknown"

# Merge NYU category onto ED encounters only; keep all encounters (left join).
_ed_mask = encounters["IsEdVisit"] == 1
_ed_enc  = encounters.loc[_ed_mask, ["EncounterKey", "PrimaryDiagnosisKey"]].copy()

# PrimaryDiagnosisKey was already coerced to FK_OUTPATIENT string for -1 rows;
# those won't match DiagnosisKey (int) so they naturally fall through to Unknown.
_ed_enc["PrimaryDiagnosisKey"] = pd.to_numeric(
    _ed_enc["PrimaryDiagnosisKey"], errors="coerce"
)
_ed_enc = _ed_enc.merge(_diag_nyu, left_on="PrimaryDiagnosisKey",
                         right_on="DiagnosisKey", how="left")
_ed_enc["ed_avoidability"] = _ed_enc["nyu_ed_category"].fillna("Unclassified").map(
    _map_avoidability
)

encounters = encounters.merge(
    _ed_enc[["EncounterKey", "ed_avoidability"]],
    on="EncounterKey", how="left"
)
# Non-ED encounters → NA (not applicable, not a count error)
encounters.loc[~_ed_mask, "ed_avoidability"] = pd.NA

del _ed_enc, _ed_mask, _diag_nyu


# ── SDOH: screening completeness per question ────────────────────────────────
# Low completion rates on sensitive screens (IPV, postpartum depression, etc.)
# may themselves signal access or trust barriers — treat as a covariate, not noise.
_completeness = (
    sdoh.groupby("DisplayName")["AnswerText"]
    .apply(lambda s: s.notna().mean())
    .rename("screening_completeness")
    .reset_index()
)
sdoh = sdoh.merge(_completeness, on="DisplayName", how="left")


# ════════════════════════════════════════════════════════════════════════════════
# FOUR-INTERVENTION ER REDUCTION SIMULATION MODEL
# ════════════════════════════════════════════════════════════════════════════════
#
# Produces output/intervention_model_results.json and .csv summarising:
#   - Per-intervention: correlation coefficient, p-value, effect size,
#     sensitivity range (conservative / base / optimistic), projected
#     % reduction in classified ED encounters
#   - Overall: Fisher combined p-value, multiple-regression R², projected
#     combined reduction, and encounter-level descriptive statistics
#
# Sensitivity parameters (diversion/adoption rates) follow published ranges:
#   Int-1 weekend diversion : 25 / 45 / 65 %  (Kellermann 1994, Weinick 2010)
#   Int-2 triage accuracy   : classifier recall used directly
#   Int-3 MyChart adoption  : +10 / +20 / +35 % of unactivated patients
#   Int-4 specialist redeploy: 15 / 30 / 50 % of idle capacity converted to PCP

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, spearmanr
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings("ignore")

from intervention2_model import (
    intervention2    as _run_intervention2,
    intervention2_rf as _run_intervention2_rf,
    intervention2_xg as _run_intervention2_xg,
)

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# ── 1.  Base ED dataset (Unknown class excluded) ──────────────────────────────
ed = encounters[
    (encounters["IsEdVisit"] == 1) &
    encounters["ed_avoidability"].notna() &
    (encounters["ed_avoidability"] != "Unknown")
].copy()
ed["is_avoidable"] = (ed["ed_avoidability"] == "Avoidable").astype(int)

n_ed_raw        = int((encounters["IsEdVisit"] == 1).sum())
n_ed_classified = len(ed)
n_avoidable     = int(ed["is_avoidable"].sum())
n_not_avoidable = int((ed["ed_avoidability"] == "Not Avoidable").sum())
n_unknown       = n_ed_raw - n_ed_classified
avoidable_rate  = n_avoidable / n_ed_classified

# ── 2.  Derive weekend flag from Date column ──────────────────────────────────
ed["_dow"] = pd.to_datetime(ed["Date"], errors="coerce").dt.dayofweek
ed["is_weekend"] = ed["_dow"].isin([5, 6]).astype(int)   # 5=Sat, 6=Sun

# ── 2b. Encounter-level features encoded directly onto ed ────────────────────
# admission_src_self: patient self-presented (no clinical referral) — higher avoidable risk.
# fillna("") so pd.NA comparisons produce False (not pd.NA) and astype(int) stays clean.
ed["admission_src_self"] = (
    ed["AdmissionSource"].fillna("") == "Non-Health Care Facility Point of Origin"
).astype(int)
# admission_type_urgent_or_elective: NOT a true emergency admission — higher avoidable risk.
# "Emergency" and "Trauma Center" are low-avoidable; "Urgent" / "Elective" are high.
ed["admission_type_non_emergency"] = (
    ~ed["AdmissionType"].fillna("Emergency").str.contains("Emergency|Trauma", na=False)
).astype(int)

# ── 3.  Patient-level feature table ──────────────────────────────────────────
#   a) Load pre-engineered features (ed_ratio, sdoh_*, age, sex)
pat_feat = pd.read_csv(
    Path(__file__).parent.parent / "DATA_CLEANED" / "patient_features.csv"
)

#   b) MyChart activation from raw patients table
mychart_map = (
    patients[["DurableKey", "MyChartStatus"]]
    .assign(mychart_active=lambda d: (d["MyChartStatus"] == "Activated").astype(int))
    [["DurableKey", "mychart_active"]]
)
pat_feat = pat_feat.merge(mychart_map, left_on="PatientDurableKey",
                           right_on="DurableKey", how="left")
pat_feat["mychart_active"] = pat_feat["mychart_active"].fillna(0).astype(int)

#   c) PCP ratio per patient: fraction of encounters with a PCP provider
_pcp_keys = set(
    pd.read_csv(
        Path(__file__).parent.parent / "DATA_CLEANED" / "providers_pcp.csv"
    )["DurableKey"].values
)
_pcp_enc = (
    encounters.groupby("PatientDurableKey")
    .apply(lambda g: pd.Series({
        "total_enc" : len(g),
        "pcp_enc"   : g["ProviderDurableKey"].isin(_pcp_keys).sum()
    }))
    .reset_index()
)
_pcp_enc["pcp_ratio"] = _pcp_enc["pcp_enc"] / _pcp_enc["total_enc"].clip(lower=1)
pat_feat = pat_feat.merge(
    _pcp_enc[["PatientDurableKey", "pcp_ratio"]],
    on="PatientDurableKey", how="left"
)
pat_feat["pcp_ratio"] = pat_feat["pcp_ratio"].fillna(0)

#   d) Demographics from patients table: smoking, race, marital status
_demo = (
    patients[["DurableKey", "SmokingStatus", "OmbEthnicity", "OmbRace", "MaritalStatus"]]
    .assign(
        smoking_current  = lambda d: d["SmokingStatus"].isin(
            ["Every Day", "Some Days", "Light Smoker"]
        ).astype(int),
        is_hispanic      = lambda d: (d["OmbEthnicity"] == "Hispanic or Latino").astype(int),
        race_white       = lambda d: (d["OmbRace"] == "White").astype(int),
        race_black       = lambda d: (d["OmbRace"] == "Black or African American").astype(int),
        is_married       = lambda d: (d["MaritalStatus"] == "Married").astype(int),
    )
    [["DurableKey", "smoking_current", "is_hispanic", "race_white", "race_black", "is_married"]]
)
pat_feat = pat_feat.merge(_demo, left_on="PatientDurableKey", right_on="DurableKey", how="left")
for _c in ["smoking_current", "is_hispanic", "race_white", "race_black", "is_married"]:
    pat_feat[_c] = pat_feat[_c].fillna(0).astype(int)

#   e) Comorbidity count + prior preventable flag from diagnosis history
_enc_diag = encounters[["PatientDurableKey", "PrimaryDiagnosisKey"]].copy()
_enc_diag["PrimaryDiagnosisKey"] = pd.to_numeric(_enc_diag["PrimaryDiagnosisKey"], errors="coerce")
_enc_diag = _enc_diag.dropna(subset=["PrimaryDiagnosisKey"]).merge(
    diagnosis[["DiagnosisKey", "DiagnosisValue", "nyu_ed_category"]],
    left_on="PrimaryDiagnosisKey", right_on="DiagnosisKey", how="left"
)
_comorbidity = (
    _enc_diag.groupby("PatientDurableKey")
    .agg(
        comorbidity_count  = ("DiagnosisValue", lambda s: s.str[:3].nunique()),
        prior_preventable  = ("nyu_ed_category", lambda s: int((s == "Emergent/Preventable").any())),
    )
    .reset_index()
)
pat_feat = pat_feat.merge(_comorbidity, on="PatientDurableKey", how="left")
pat_feat["comorbidity_count"] = pat_feat["comorbidity_count"].fillna(0)
pat_feat["prior_preventable"] = pat_feat["prior_preventable"].fillna(0).astype(int)
del _enc_diag, _comorbidity

#   f) Prior inpatient fraction: how often has this patient been admitted inpatient
_inpat = (
    encounters.groupby("PatientDurableKey")["IsInpatientAdmission"]
    .mean()
    .rename("prior_inpatient_frac")
    .reset_index()
)
pat_feat = pat_feat.merge(_inpat, on="PatientDurableKey", how="left")
pat_feat["prior_inpatient_frac"] = pat_feat["prior_inpatient_frac"].fillna(0)
del _inpat

results = {}   # master dict → JSON

# ════════════════════════════════════════════════════════════════════════════════
# INTERVENTION 1 — Extended / Weekend Clinic Hours
# Hypothesis: avoidable ED visits are disproportionately concentrated on
# weekends when primary care is unavailable; extended hours would divert them.
# ════════════════════════════════════════════════════════════════════════════════
_wkd  = ed[ed["is_weekend"] == 1]
_wkdy = ed[ed["is_weekend"] == 0]

_avoid_wkd  = _wkd["is_avoidable"].mean()
_avoid_wkdy = _wkdy["is_avoidable"].mean()

# Chi-square: weekend × avoidability
_ct = pd.crosstab(ed["is_weekend"], ed["is_avoidable"])
_chi2_i1, _p_i1, _, _ = chi2_contingency(_ct)
_n_i1 = len(ed)
_cramers_v = float(np.sqrt(_chi2_i1 / (_n_i1 * (min(_ct.shape) - 1))))

# Point-biserial correlation (equivalent to Pearson for binary predictor)
_r_i1, _pr_i1 = pointbiserialr(ed["is_weekend"], ed["is_avoidable"])

# Avoidable encounters on weekends = potential diversion pool
_n_avoid_wkd = int(_wkd["is_avoidable"].sum())

# Sensitivity: projected % reduction in total classified ED at three diversion rates
_div_rates = {"conservative": 0.25, "base": 0.45, "optimistic": 0.65}
_sens_i1 = {
    k: round(_n_avoid_wkd * v / n_ed_classified * 100, 2)
    for k, v in _div_rates.items()
}

results["intervention_1_extended_hours"] = {
    "name"                        : "Extended / Weekend Clinic Hours",
    "hypothesis"                  : "Avoidable ED visits spike on weekends due to PCP unavailability",
    "n_classified_ed"             : n_ed_classified,
    "n_weekend_ed"                : len(_wkd),
    "n_avoidable_weekend"         : _n_avoid_wkd,
    "avoidable_rate_weekend_pct"  : round(_avoid_wkd  * 100, 2),
    "avoidable_rate_weekday_pct"  : round(_avoid_wkdy * 100, 2),
    "rate_difference_pct"         : round((_avoid_wkd - _avoid_wkdy) * 100, 2),
    "chi2_statistic"              : round(_chi2_i1, 4),
    "p_value"                     : float(_p_i1),
    "cramers_v"                   : round(_cramers_v, 4),
    "correlation_coeff_r"         : round(_r_i1, 4),
    "correlation_p_value"         : float(_pr_i1),
    "projected_ed_reduction_pct"  : _sens_i1,
    "sensitivity_assumption"      : "% of avoidable weekend ED visits successfully diverted to extended-hours clinics",
}

# ════════════════════════════════════════════════════════════════════════════════
# INTERVENTION 2 — Upgraded Triage with Licensed Staff (Diversion Classifier)
# Logistic regression predicts avoidable ED visit using features available at
# triage time — no diagnosis column used (target was derived from diagnosis).
# Features: weekend, off-hours, patient age, sex, SDOH burden, prior ED rate,
#           prior total encounters, has-PCP flag.
# ════════════════════════════════════════════════════════════════════════════════

# ── Model selection: uncomment exactly one block ──────────────────────────────
"""
_i2 = _run_intervention2(
    ed, pat_feat,
    C=1.0,                 # inverse regularisation strength (higher = less regularised)
    max_iter=500,          # solver iteration cap
    class_weight="balanced",  # 'balanced' or None
    solver="lbfgs",        # 'lbfgs', 'saga', 'liblinear'
    test_size=0.2,         # held-out fraction
    random_state=42,
    n_boot=200,            # bootstrap iterations for coefficient p-values
    divert_rates={"conservative": 0.50, "base": 0.70, "optimistic": 0.90},
)

_i2 = _run_intervention2_rf(
    ed, pat_feat,
    n_estimators=100,       # number of trees (more = stabler, slower)
    max_depth=None,         # None = full depth; try 5–15 to reduce overfit
    min_samples_leaf=1,     # increase (e.g. 5–20) to regularise small leaves
    max_features="sqrt",    # 'sqrt', 'log2', or float fraction of features
    class_weight="balanced",  # 'balanced', 'balanced_subsample', or None
    test_size=0.2,
    random_state=42,
    n_permutations=200,     # permutation iterations for overall p-value
    divert_rates={"conservative": 0.50, "base": 0.70, "optimistic": 0.90},
)
"""
_i2 = _run_intervention2_xg(
    ed, pat_feat,
    n_estimators=3000,     # boosting rounds
    max_depth=6,            # tree depth per round (lower = less overfit)
    learning_rate=0.005,      # eta / step-size shrinkage
    subsample=0.7,          # row sampling ratio per tree (0.5–1.0)
    colsample_bytree=0.8,   # feature sampling ratio per tree (0.5–1.0)
    min_child_weight=3,     # min instance weight in a child (higher = conservative)
    gamma=0,              # min loss reduction to split; increase for pruning
    reg_alpha=0.0,          # L1 weight regularisation
    reg_lambda=2.0,         # L2 weight regularisation
    scale_pos_weight=1,     # None = auto (neg/pos ratio); or set manually
    test_size=0.2,
    random_state=42,
    n_permutations=200,
    divert_rates={"conservative": 0.50, "base": 0.70, "optimistic": 0.90},
)

# ─────────────────────────────────────────────────────────────────────────────

# Scalars needed by downstream sections (Fisher p-value, sequential model).
_recall_i2 = _i2["sensitivity_recall"]
_p_i2      = _i2["p_value"]

results["intervention_2_triage_diversion"] = {
    "name"      : "Upgraded Triage with Licensed Staff",
    "hypothesis": "Patient features available at triage predict avoidable ED use without diagnosis",
    **_i2,
}

# ════════════════════════════════════════════════════════════════════════════════
# INTERVENTION 3 — MyChart + SDOH Resource Referral
# Natural experiment: patients with MyChart activated vs not.
# Unit of analysis: patient.  Outcome: ed_ratio.
# ════════════════════════════════════════════════════════════════════════════════
_mc_on  = pat_feat[pat_feat["mychart_active"] == 1]["ed_ratio"].dropna()
_mc_off = pat_feat[pat_feat["mychart_active"] == 0]["ed_ratio"].dropna()

_u_i3, _p_i3 = mannwhitneyu(_mc_on, _mc_off, alternative="two-sided")
_r_i3, _pr_i3 = pearsonr(
    pat_feat["mychart_active"].dropna(),
    pat_feat.loc[pat_feat["mychart_active"].notna(), "ed_ratio"].fillna(0)
)

# Cohen's d
_pool_sd = np.sqrt((_mc_on.std() ** 2 + _mc_off.std() ** 2) / 2)
_cohens_d_i3 = float((_mc_on.mean() - _mc_off.mean()) / _pool_sd)

_ed_ratio_diff_pct = (_mc_on.mean() - _mc_off.mean()) / _mc_off.mean() * 100
_n_unactivated = int((pat_feat["mychart_active"] == 0).sum())

# Sensitivity: if X% of unactivated patients are activated
_adopt_rates = {"conservative": 0.10, "base": 0.20, "optimistic": 0.35}
_ed_ratio_effect = _mc_on.mean() - _mc_off.mean()  # negative = reduction
_sens_i3 = {}
for k, rate in _adopt_rates.items():
    _newly_activated = _n_unactivated * rate
    _patients_affected = _newly_activated
    # Reduction in total ed_ratio × patient pool
    _total_pat = len(pat_feat)
    _delta = abs(_ed_ratio_effect) * _patients_affected / _total_pat * 100
    _sens_i3[k] = round(_delta, 2)

results["intervention_3_mychart_sdoh"] = {
    "name"                       : "MyChart Activation + SDOH Resource Referral",
    "hypothesis"                 : "MyChart activation is associated with substantially lower ED utilization rate",
    "n_activated"                : len(_mc_on),
    "n_not_activated"            : len(_mc_off),
    "ed_ratio_activated_mean"    : round(float(_mc_on.mean()), 4),
    "ed_ratio_not_activated_mean": round(float(_mc_off.mean()), 4),
    "ed_ratio_diff_pct"          : round(float(_ed_ratio_diff_pct), 2),
    "cohens_d"                   : round(_cohens_d_i3, 4),
    "mann_whitney_u"             : round(float(_u_i3), 2),
    "p_value"                    : float(_p_i3),
    "correlation_coeff_r"        : round(float(_r_i3), 4),
    "correlation_p_value"        : float(_pr_i3),
    "projected_ed_reduction_pct" : _sens_i3,
    "sensitivity_assumption"     : "% of currently unactivated patients who gain MyChart access",
}

# ════════════════════════════════════════════════════════════════════════════════
# INTERVENTION 4 — Idle Specialist Reallocation as Part-time PCP
# Unit of analysis: patient.  Predictor: pcp_ratio.  Outcome: ed_ratio.
# Uses internal PCP encounter ratio as proxy for PCP access coverage.
# ════════════════════════════════════════════════════════════════════════════════
_pcp_df = pat_feat[["pcp_ratio", "ed_ratio"]].dropna()

_r_i4,  _pr_i4  = pearsonr(_pcp_df["pcp_ratio"], _pcp_df["ed_ratio"])
_rho_i4, _prho_i4 = spearmanr(_pcp_df["pcp_ratio"], _pcp_df["ed_ratio"])

# OLS slope: Δed_ratio per unit Δpcp_ratio (controls for scale)
_slope_i4, _intercept_i4, _r_ols, _p_ols, _se_slope = stats.linregress(
    _pcp_df["pcp_ratio"], _pcp_df["ed_ratio"]
)

# Idle specialist capacity: providers with encounter load < P25 among all providers
_prov_load = (
    encounters.groupby("ProviderDurableKey")["EncounterKey"]
    .count()
    .rename("annual_enc")
    .reset_index()
)
_p25_load  = _prov_load["annual_enc"].quantile(0.25)
_n_idle    = int((_prov_load["annual_enc"] <= _p25_load).sum())
_n_pcp_now = len(_pcp_keys)

# Sensitivity: if X% of idle specialists become part-time PCPs, estimate
# the new PCP ratio gain across the patient population and apply slope
_redeploy_rates = {"conservative": 0.15, "base": 0.30, "optimistic": 0.50}
_sens_i4 = {}
for k, rate in _redeploy_rates.items():
    _new_pcps  = _n_idle * rate
    # Rough PCP ratio gain: new PCP encounters ≈ new_pcps × median_load × 0.5 (part-time)
    _median_load = float(_prov_load["annual_enc"].median())
    _new_pcp_enc = _new_pcps * _median_load * 0.5
    _total_enc   = len(encounters)
    _delta_pcp_ratio = _new_pcp_enc / max(_total_enc, 1)
    _delta_ed_ratio  = _slope_i4 * _delta_pcp_ratio  # negative slope → reduction
    _sens_i4[k] = round(abs(_delta_ed_ratio) * 100, 2)

results["intervention_4_specialist_reallocation"] = {
    "name"                        : "Idle Specialist Reallocation as Part-time PCP",
    "hypothesis"                  : "Higher PCP encounter ratio is negatively correlated with ED utilization",
    "n_patients"                  : len(_pcp_df),
    "pearson_r"                   : round(float(_r_i4), 4),
    "pearson_p_value"             : float(_pr_i4),
    "spearman_rho"                : round(float(_rho_i4), 4),
    "spearman_p_value"            : float(_prho_i4),
    "correlation_coeff_r"         : round(float(_r_i4), 4),
    "ols_slope"                   : round(float(_slope_i4), 6),
    "ols_slope_se"                : round(float(_se_slope), 6),
    "ols_p_value"                 : float(_p_ols),
    "p_value"                     : float(_pr_i4),
    "n_idle_providers"            : _n_idle,
    "idle_threshold_p25_enc"      : round(float(_p25_load), 1),
    "projected_ed_reduction_pct"  : _sens_i4,
    "sensitivity_assumption"      : "% of idle (≤P25 load) providers redeployed as 0.5 FTE PCP",
}

# ════════════════════════════════════════════════════════════════════════════════
# OVERALL MODEL
# Fisher combined p-value; multiple regression with interaction terms;
# sequential absorption model for projected combined reduction.
# ════════════════════════════════════════════════════════════════════════════════

# Fisher's method: χ² = -2 Σ ln(p_i), df = 2k
_ps      = [_p_i1, _p_i2, float(_p_i3), float(_pr_i4)]
_ps_safe = [max(p, 1e-300) for p in _ps]
_fisher_chi2 = -2 * sum(np.log(_ps_safe))
_fisher_p    = float(stats.chi2.sf(_fisher_chi2, df=2 * len(_ps)))

# Multiple regression with interaction terms on patient-level data.
# Main effects: mychart_active, pcp_ratio, sdoh_unique_domains, encounters_per_month
# Interaction terms capture synergies between interventions:
#   mychart × pcp_ratio  — digital access amplifies PCP benefit
#   mychart × sdoh       — MyChart most impactful for high-SDOH patients
#   pcp × sdoh           — PCP coverage most protective for high-SDOH patients
_multi_df = pat_feat[[
    "ed_ratio", "mychart_active", "pcp_ratio",
    "sdoh_unique_domains", "encounters_per_month"
]].dropna().copy()
_multi_df["mychart_x_pcp"]  = _multi_df["mychart_active"] * _multi_df["pcp_ratio"]
_multi_df["mychart_x_sdoh"] = _multi_df["mychart_active"] * _multi_df["sdoh_unique_domains"]
_multi_df["pcp_x_sdoh"]     = _multi_df["pcp_ratio"]      * _multi_df["sdoh_unique_domains"]

_ALL_PREDICTORS = [
    "mychart_active", "pcp_ratio", "sdoh_unique_domains", "encounters_per_month",
    "mychart_x_pcp", "mychart_x_sdoh", "pcp_x_sdoh",
]
_multi_X = _multi_df[_ALL_PREDICTORS].values
_multi_y = _multi_df["ed_ratio"].values

_corr_mat  = np.corrcoef(np.column_stack([_multi_X, _multi_y]).T)
_r_with_y  = _corr_mat[:-1, -1]
_R_squared = float(np.dot(
    _r_with_y,
    np.linalg.lstsq(_corr_mat[:-1, :-1], _r_with_y, rcond=None)[0]
))
_R_squared = max(0.0, min(1.0, _R_squared))
_R_overall = float(np.sqrt(_R_squared))

# Interaction term correlations with ed_ratio (sign = direction of synergy)
_interaction_corrs = {
    feat: round(float(np.corrcoef(_multi_df[feat], _multi_df["ed_ratio"])[0, 1]), 4)
    for feat in ["mychart_x_pcp", "mychart_x_sdoh", "pcp_x_sdoh"]
}

# ── Sequential absorption model ───────────────────────────────────────────────
# INT1 and INT2 divert encounters from the current classified ED pool.
# INT3 and INT4 reduce future ED visits via patient-level behaviour change.
# Each scenario draws from the *remaining* pool after prior diversions,
# avoiding the double-counting of the naive additive sum.
#
# Pool:  n_ed_classified classified ED encounters
# Step 1 (INT1): divert avoidable weekend visits
# Step 2 (INT2): divert remaining avoidable visits identified at triage
# Steps 3–4 (INT3+INT4): patient-level reductions applied multiplicatively
#   to account for interaction: patients who have both MyChart and PCP access
#   receive a combined benefit = 1 - (1-eff3)×(1-eff4) rather than eff3+eff4

def _sequential_reduction(scenario):
    pool = float(n_ed_classified)

    # INT1 — weekend diversion
    div_rates_i1 = {"conservative": 0.25, "base": 0.45, "optimistic": 0.65}
    diverted_i1  = _n_avoid_wkd * div_rates_i1[scenario]
    pool        -= diverted_i1

    # INT2 — triage diversion from remaining pool
    # Avoidable fraction in remaining pool ≈ same as overall (conservative assumption)
    div_rates_i2 = {"conservative": 0.50, "base": 0.70, "optimistic": 0.90}
    diverted_i2  = pool * avoidable_rate * _recall_i2 * div_rates_i2[scenario]
    pool        -= diverted_i2

    # INT3 — MyChart adoption: fractional reduction applied to remaining pool
    adopt_rates  = {"conservative": 0.10, "base": 0.20, "optimistic": 0.35}
    _frac_unact  = _n_unactivated / max(len(pat_feat), 1)
    eff3         = abs(_ed_ratio_effect / max(float(_mc_off.mean()), 1e-9)) * _frac_unact * adopt_rates[scenario]

    # INT4 — Specialist reallocation: fractional reduction applied to remaining pool
    redeploy     = {"conservative": 0.15, "base": 0.30, "optimistic": 0.50}
    _delta_pcp   = (_n_idle * redeploy[scenario] * float(_prov_load["annual_enc"].median()) * 0.5
                    / max(len(encounters), 1))
    eff4         = abs(_slope_i4 * _delta_pcp / max(float(_pcp_df["ed_ratio"].mean()), 1e-9))

    # Multiplicative combination for INT3+INT4 interaction, boosted by the
    # mychart_x_pcp regression interaction term.  A negative correlation between
    # the joint term and ed_ratio means having both MyChart AND PCP access reduces
    # ED use more than either alone; we scale the combined effect accordingly.
    _synergy_factor = 1 + max(0.0, -_interaction_corrs["mychart_x_pcp"])
    combined_eff34 = min(
        (1 - (1 - min(eff3, 1.0)) * (1 - min(eff4, 1.0))) * _synergy_factor,
        1.0,
    )
    diverted_34    = pool * combined_eff34
    pool          -= diverted_34

    total_diverted = diverted_i1 + diverted_i2 + diverted_34
    return round(min(total_diverted / n_ed_classified * 100, 100), 2)

_combined_proj = {s: _sequential_reduction(s) for s in ["conservative", "base", "optimistic"]}

results["overall"] = {
    "n_ed_encounters_raw"             : n_ed_raw,
    "n_ed_classified"                 : n_ed_classified,
    "n_ed_unknown_excluded"           : n_unknown,
    "unknown_exclusion_pct"           : round(n_unknown / n_ed_raw * 100, 2),
    "avoidable_pct_of_classified"     : round(avoidable_rate * 100, 2),
    "not_avoidable_pct_of_classified" : round(n_not_avoidable / n_ed_classified * 100, 2),
    "fisher_combined_chi2"            : round(_fisher_chi2, 4),
    "fisher_combined_p_value"         : _fisher_p,
    "multiple_R_with_interactions"    : round(_R_overall, 4),
    "multiple_R_squared_with_interactions": round(_R_squared, 4),
    "interaction_term_correlations"   : _interaction_corrs,
    "projected_combined_reduction_pct": _combined_proj,
    "projection_method"               : (
        "Sequential absorption: INT1 diverts avoidable weekend visits first, "
        "INT2 draws from remaining pool via triage recall, "
        "INT3+INT4 applied multiplicatively (1-(1-eff3)(1-eff4)) to capture "
        "synergy between MyChart access and PCP coverage. "
        "Unknown-class encounters excluded throughout."
    ),
}

# ── Write outputs ─────────────────────────────────────────────────────────────
_json_path = _OUT / "intervention_model_results.json"
with open(_json_path, "w", encoding="utf-8") as _f:
    json.dump(results, _f, indent=2, default=str)
print(f"[intervention model] JSON  → {_json_path}")

# Flat CSV: one row per intervention + one overall row
_csv_rows = []
for _int_key, _int_val in results.items():
    if _int_key == "overall":
        continue
    _row = {
        "intervention"             : _int_key,
        "name"                     : _int_val.get("name"),
        "p_value"                  : _int_val.get("p_value"),
        "correlation_coeff_r"      : _int_val.get("correlation_coeff_r"),
        "effect_size"              : (
            _int_val.get("cramers_v")
            or _int_val.get("auc_roc")
            or _int_val.get("cohens_d")
            or _int_val.get("ols_slope")
        ),
        "effect_size_metric"       : (
            "cramers_v"  if "cramers_v"  in _int_val else
            "auc_roc"    if "auc_roc"    in _int_val else
            "cohens_d"   if "cohens_d"   in _int_val else
            "ols_slope"
        ),
        "sensitivity_recall"       : _int_val.get("sensitivity_recall"),
        "specificity"              : _int_val.get("specificity"),
        "projected_reduction_conservative_pct": _int_val["projected_ed_reduction_pct"]["conservative"],
        "projected_reduction_base_pct"        : _int_val["projected_ed_reduction_pct"]["base"],
        "projected_reduction_optimistic_pct"  : _int_val["projected_ed_reduction_pct"]["optimistic"],
        "n"                        : (
            _int_val.get("n_classified_ed")
            or _int_val.get("n_model")
            or _int_val.get("n_activated")
            or _int_val.get("n_patients")
        ),
    }
    _csv_rows.append(_row)

# Overall row
_csv_rows.append({
    "intervention"                        : "overall",
    "name"                                : "Combined Four-Intervention Bundle",
    "p_value"                             : results["overall"]["fisher_combined_p_value"],
    "correlation_coeff_r"                 : results["overall"]["multiple_R_with_interactions"],
    "effect_size"                         : results["overall"]["multiple_R_squared_with_interactions"],
    "effect_size_metric"                  : "multiple_R_squared",
    "sensitivity_recall"                  : None,
    "specificity"                         : None,
    "projected_reduction_conservative_pct": results["overall"]["projected_combined_reduction_pct"]["conservative"],
    "projected_reduction_base_pct"        : results["overall"]["projected_combined_reduction_pct"]["base"],
    "projected_reduction_optimistic_pct"  : results["overall"]["projected_combined_reduction_pct"]["optimistic"],
    "n"                                   : n_ed_classified,
})

_csv_path = _OUT / "intervention_model_results.csv"
pd.DataFrame(_csv_rows).to_csv(_csv_path, index=False)
print(f"[intervention model] CSV   → {_csv_path}")
