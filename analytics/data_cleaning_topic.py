"""
DataFest 2026 - Data Cleaning Pipeline
Topic: Clock Gap (Day/Night workload imbalance + overloaded PCPs)

Produces cleaned datasets in DATA_CLEANED/ for downstream analysis.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'DATA')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'DATA_CLEANED')
SDOH_MAP_FILE = os.path.join(os.path.dirname(__file__), 'Social Determinant Questions and Domains.csv')

os.makedirs(OUT_DIR, exist_ok=True)

# Sentinel values treatment
SENTINEL_TO_NAN = ['*Not Applicable', '*Unknown', '*Deleted']
SENTINEL_KEEP = ['*Unspecified']  # keep as "Unspecified" category

# Time period definitions (aligned with clinical shifts)
def assign_time_period(hour):
    if 7 <= hour <= 17:
        return 'Day'
    elif 18 <= hour <= 21:
        return 'Evening'
    else:
        return 'Night'

# PCP specialties
PCP_SPECIALTIES = [
    'Family Medicine',
    'Internal Medicine',
    'Pediatrics',
    'General Practice',
    'Internal Medicine/Pediatrics'
]

# Diagnosis category mapping (broad groups for Clock Gap analysis)
DIAG_CATEGORIES = {
    'Mental Health': [
        'Depressive', 'Bipolar', 'Anxiety', 'Schizophreni', 'Psycho',
        'PTSD', 'Substance', 'Alcohol', 'Opioid', 'Cannabis',
        'Attention Deficit', 'ADHD', 'Obsessive', 'Eating Disorder',
        'Autism', 'Schizoaffective'
    ],
    'Chronic': [
        'Diabetes', 'Hypertension', 'Heart Failure', 'COPD',
        'Asthma', 'Chronic Kidney', 'Obesity', 'Hyperlipidemia',
        'Coronary', 'Atrial Fibrillation', 'Thyroid'
    ],
    'Acute': [
        'Pneumonia', 'Fracture', 'Appendicitis', 'Sepsis',
        'Acute Kidney', 'Myocardial Infarction', 'Stroke', 'Cellulitis'
    ]
}

def categorize_diagnosis(name):
    if pd.isna(name):
        return 'Unknown'
    name_upper = str(name).upper()
    for cat, keywords in DIAG_CATEGORIES.items():
        for kw in keywords:
            if kw.upper() in name_upper:
                return cat
    return 'Other'


# ============================================================
# REPORT TRACKING
# ============================================================
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

def log_table_stats(name, df_before_rows, df_after_rows, notes=""):
    log(f"  Rows before: {df_before_rows:,}")
    log(f"  Rows after:  {df_after_rows:,}")
    log(f"  Removed:     {df_before_rows - df_after_rows:,} ({(df_before_rows - df_after_rows) / max(df_before_rows,1) * 100:.2f}%)")
    if notes:
        log(f"  Notes: {notes}")


# ============================================================
# STEP 1: DEPARTMENTS
# ============================================================
log("=" * 60)
log("STEP 1: Cleaning departments.csv")
log("=" * 60)

dept = pd.read_csv(os.path.join(DATA_DIR, 'departments.csv'))
dept_raw_rows = len(dept)
log(f"  Columns: {list(dept.columns)}")

# Check sentinel rows
sentinel_rows = dept[dept['DepartmentKey'].isin([-1, -2, -3])]
log(f"  Sentinel rows (Key=-1/-2/-3): {len(sentinel_rows)}")

# Remove sentinel rows
dept_clean = dept[dept['DepartmentKey'] > 0].copy()

# Clean text fields
for col in ['Address', 'City', 'County', 'DepartmentName', 'DepartmentSpecialty', 'DepartmentType', 'PostalCode', 'CensusTract']:
    if col in dept_clean.columns:
        dept_clean[col] = dept_clean[col].replace(SENTINEL_TO_NAN, np.nan)
        dept_clean[col] = dept_clean[col].replace(SENTINEL_KEEP, 'Unspecified')

# Missing rate
log("  Missing rates after cleaning:")
for col in dept_clean.columns:
    miss = dept_clean[col].isna().sum()
    if miss > 0:
        log(f"    {col}: {miss}/{len(dept_clean)} ({miss/len(dept_clean)*100:.1f}%)")

log_table_stats('departments', dept_raw_rows, len(dept_clean), "Removed 3 sentinel rows")
dept_clean.to_csv(os.path.join(OUT_DIR, 'departments_cleaned.csv'), index=False)
log(f"  Output: departments_cleaned.csv")
log("")


# ============================================================
# STEP 2: PROVIDERS
# ============================================================
log("=" * 60)
log("STEP 2: Cleaning providers.csv")
log("=" * 60)

prov = pd.read_csv(os.path.join(DATA_DIR, 'providers.csv'))
prov_raw_rows = len(prov)
log(f"  Columns: {list(prov.columns)}")
log(f"  Total providers: {prov_raw_rows:,}")

# Remove MODEL USER
model_users = prov[prov['ClinicianTitle'].str.contains('MODEL USER', case=False, na=False)]
log(f"  MODEL USER entries: {len(model_users)}")
prov_clean = prov[~prov['ClinicianTitle'].str.contains('MODEL USER', case=False, na=False)].copy()

# Clean sentinel values
for col in ['ClinicianTitle', 'OfficeAddress', 'OfficeCity', 'OfficePostalCode', 'PrimaryDepartment', 'PrimarySpecialty', 'Type']:
    prov_clean[col] = prov_clean[col].replace(SENTINEL_TO_NAN, np.nan)
    prov_clean[col] = prov_clean[col].replace(SENTINEL_KEEP, 'Unspecified')

# Mark PCP
prov_clean['is_pcp'] = (
    (prov_clean['Type'] == 'Physician') &
    (prov_clean['PrimarySpecialty'].isin(PCP_SPECIALTIES))
)
log(f"  Total PCPs identified: {prov_clean['is_pcp'].sum()}")
log(f"  PCP specialties: {prov_clean[prov_clean['is_pcp']]['PrimarySpecialty'].value_counts().to_dict()}")

# Missing rate
log("  Missing rates:")
for col in prov_clean.columns:
    miss = prov_clean[col].isna().sum()
    if miss > 0:
        log(f"    {col}: {miss}/{len(prov_clean)} ({miss/len(prov_clean)*100:.1f}%)")

log_table_stats('providers', prov_raw_rows, len(prov_clean), "Removed MODEL USER entries")

# Save full providers + PCP subset
prov_clean.to_csv(os.path.join(OUT_DIR, 'providers_cleaned.csv'), index=False)
prov_pcp = prov_clean[prov_clean['is_pcp']].copy()
prov_pcp.to_csv(os.path.join(OUT_DIR, 'providers_pcp.csv'), index=False)
log(f"  Output: providers_cleaned.csv ({len(prov_clean):,} rows)")
log(f"  Output: providers_pcp.csv ({len(prov_pcp):,} rows)")
log("")


# ============================================================
# STEP 3: PATIENTS
# ============================================================
log("=" * 60)
log("STEP 3: Cleaning patients.csv")
log("=" * 60)

pat = pd.read_csv(os.path.join(DATA_DIR, 'patients.csv'))
pat_raw_rows = len(pat)
log(f"  Columns: {list(pat.columns)}")
log(f"  Total patients: {pat_raw_rows:,}")

# Check near-useless columns
for col in ['SexAssignedAtBirth', 'SexualOrientation']:
    unspec = pat[col].isin(['*Unspecified', '*Unknown', '*Not Applicable']).sum()
    log(f"  {col}: {unspec}/{pat_raw_rows} unspecified/unknown ({unspec/pat_raw_rows*100:.1f}%)")

# Drop near-useless columns
drop_cols = ['SexAssignedAtBirth', 'SexualOrientation']
pat_clean = pat.drop(columns=drop_cols).copy()
log(f"  Dropped columns (>90% uninformative): {drop_cols}")

# Clean sentinel values in remaining columns
for col in pat_clean.columns:
    if pat_clean[col].dtype == object:
        pat_clean[col] = pat_clean[col].replace(SENTINEL_TO_NAN, np.nan)
        pat_clean[col] = pat_clean[col].replace(SENTINEL_KEEP, 'Unspecified')

# Add geo validity flag
pat_clean['has_geo'] = pat_clean['CensusBlockGroupFipsCode'].notna() & (pat_clean['CensusBlockGroupFipsCode'] != 'Unspecified')
log(f"  Patients with valid geo: {pat_clean['has_geo'].sum():,} ({pat_clean['has_geo'].mean()*100:.1f}%)")

# Missing rate
log("  Missing rates:")
for col in pat_clean.columns:
    if col == 'has_geo':
        continue
    miss = pat_clean[col].isna().sum()
    if miss > 0:
        log(f"    {col}: {miss}/{len(pat_clean)} ({miss/len(pat_clean)*100:.1f}%)")

log_table_stats('patients', pat_raw_rows, len(pat_clean), "No rows removed, dropped 2 uninformative columns")
pat_clean.to_csv(os.path.join(OUT_DIR, 'patients_cleaned.csv'), index=False)
log(f"  Output: patients_cleaned.csv ({len(pat_clean):,} rows)")
log("")


# ============================================================
# STEP 4: DIAGNOSIS
# ============================================================
log("=" * 60)
log("STEP 4: Cleaning diagnosis.csv")
log("=" * 60)

diag = pd.read_csv(os.path.join(DATA_DIR, 'diagnosis.csv'))
diag_raw_rows = len(diag)
log(f"  Columns: {list(diag.columns)}")
log(f"  Total diagnoses: {diag_raw_rows:,}")

# Check sentinel keys
sentinel_diag = diag[diag['DiagnosisKey'] < 0]
log(f"  Sentinel keys (Key<0): {len(sentinel_diag)}")
if len(sentinel_diag) > 0:
    log(f"    Values: {sentinel_diag['DiagnosisKey'].value_counts().to_dict()}")

# Remove sentinel rows
diag_clean = diag[diag['DiagnosisKey'] > 0].copy()

# Clean text sentinel values
for col in ['GroupName', 'GroupCode', 'DiagnosisName', 'DiagnosisValue']:
    diag_clean[col] = diag_clean[col].replace(SENTINEL_TO_NAN, np.nan)
    diag_clean[col] = diag_clean[col].replace(SENTINEL_KEEP, 'Unspecified')

# Add broad category
diag_clean['DiagCategory'] = diag_clean['DiagnosisName'].apply(categorize_diagnosis)
cat_dist = diag_clean['DiagCategory'].value_counts()
log(f"  Diagnosis category distribution:")
for cat, cnt in cat_dist.items():
    log(f"    {cat}: {cnt:,} ({cnt/len(diag_clean)*100:.1f}%)")

log_table_stats('diagnosis', diag_raw_rows, len(diag_clean), "Removed sentinel key rows")
diag_clean.to_csv(os.path.join(OUT_DIR, 'diagnosis_cleaned.csv'), index=False)
log(f"  Output: diagnosis_cleaned.csv ({len(diag_clean):,} rows)")
log("")


# ============================================================
# STEP 5: SOCIAL DETERMINANTS
# ============================================================
log("=" * 60)
log("STEP 5: Cleaning social_determinants.csv")
log("=" * 60)

sdoh = pd.read_csv(os.path.join(DATA_DIR, 'social_determinants.csv'))
sdoh_raw_rows = len(sdoh)
log(f"  Columns: {list(sdoh.columns)}")
log(f"  Total SDOH rows: {sdoh_raw_rows:,}")

# Load domain mapping
domain_map = pd.read_csv(SDOH_MAP_FILE)
display_to_domain = dict(zip(domain_map['DisplayName'], domain_map['Domain']))
log(f"  Domain mapping loaded: {len(display_to_domain)} questions -> {domain_map['Domain'].nunique()} domains")

# Check missing Domain
missing_domain = sdoh['Domain'].isna().sum()
log(f"  Missing Domain: {missing_domain:,} ({missing_domain/sdoh_raw_rows*100:.1f}%)")

# Fill Domain using mapping
sdoh['Domain'] = sdoh.apply(
    lambda row: display_to_domain.get(row['DisplayName'], row['Domain'])
    if pd.isna(row['Domain']) else row['Domain'],
    axis=1
)
still_missing = sdoh['Domain'].isna().sum()
log(f"  Missing Domain after mapping: {still_missing:,}")

# Remove rows with no Domain and no DisplayName
invalid_sdoh = sdoh['Domain'].isna() & sdoh['DisplayName'].isna()
sdoh_clean = sdoh[~invalid_sdoh].copy()

# Remove rows where DisplayName is sentinel
sdoh_clean = sdoh_clean[~sdoh_clean['DisplayName'].isin(SENTINEL_TO_NAN + SENTINEL_KEEP)].copy()

# Clean AnswerText sentinels
sdoh_clean['AnswerText'] = sdoh_clean['AnswerText'].replace(SENTINEL_TO_NAN, np.nan)
sdoh_clean['AnswerText'] = sdoh_clean['AnswerText'].replace(SENTINEL_KEEP, 'Unspecified')

# Remove rows with missing EncounterKey or PatientDurableKey
sdoh_clean = sdoh_clean.dropna(subset=['EncounterKey', 'PatientDurableKey'])

# Mark positive SDOH (domain-specific logic)
# "Positive" means the patient shows risk/need in that domain
def is_positive_sdoh(row):
    answer = row['AnswerText']
    domain = row['Domain']
    if pd.isna(answer):
        return False
    answer_lower = str(answer).lower().strip()
    
    # Domain-specific rules
    if domain == 'Alcohol Use':
        # Risk: frequent drinking (2-3/week, 4+/week, daily, 6+ drinks)
        risk_terms = ['2-3 times a week', '4 or more times a week', 'daily or almost daily',
                      '3 or 4', '5 or 6', '7, 8, or 9', '10 or more']
        return any(t.lower() in answer_lower for t in risk_terms)
    
    elif domain == 'physical activity':
        # Risk: 0 days or 0 min (sedentary)
        if answer_lower in ['0 days', '0 min', '0']:
            return True
        return False
    
    elif domain == 'stress':
        # Risk: high stress
        return answer_lower in ['rather much', 'very much', 'quite a bit']
    
    elif domain == 'social connections':
        # Risk: low social connection indicators
        risk_terms = ['never', 'none', 'no', 'separated', 'widowed']
        return any(t in answer_lower for t in risk_terms)
    
    elif domain == 'Depression':
        # Numeric scores > 0, or affirmative
        try:
            return float(answer_lower) > 0
        except ValueError:
            return answer_lower in ['nearly every day', 'more than half the days', 
                                     'several days', 'yes', 'often']
    
    else:
        # General: yes, often, sometimes, always, hard, etc.
        general_positive = ['yes', 'often', 'sometimes', 'always', 'fairly often',
                           'very hard', 'somewhat hard', 'nearly every day', 
                           'more than half', 'several days', 'quite a bit']
        # Numeric > 0
        try:
            val = float(answer_lower)
            return val > 0
        except ValueError:
            pass
        return any(kw in answer_lower for kw in general_positive)

sdoh_clean['is_positive'] = sdoh_clean.apply(is_positive_sdoh, axis=1)

# Domain distribution
log("  Domain distribution (after cleaning):")
domain_counts = sdoh_clean['Domain'].value_counts()
for dom, cnt in domain_counts.head(15).items():
    pos = sdoh_clean[sdoh_clean['Domain'] == dom]['is_positive'].sum()
    log(f"    {dom}: {cnt:,} rows, {pos:,} positive ({pos/cnt*100:.1f}%)")

log_table_stats('social_determinants', sdoh_raw_rows, len(sdoh_clean), "Removed invalid Domain/DisplayName/keys")
sdoh_clean.to_csv(os.path.join(OUT_DIR, 'sdoh_cleaned.csv'), index=False)
log(f"  Output: sdoh_cleaned.csv ({len(sdoh_clean):,} rows)")
log("")


# ============================================================
# STEP 6: GEO LOOKUP (tigercensuscodes)
# ============================================================
log("=" * 60)
log("STEP 6: Cleaning tigercensuscodes.csv (geo lookup)")
log("=" * 60)

geo = pd.read_csv(os.path.join(DATA_DIR, 'tigercensuscodes.csv'))
geo_raw_rows = len(geo)
log(f"  Columns: {list(geo.columns)}")
log(f"  Total geo rows: {geo_raw_rows:,}")

# Check for valid coordinates
geo_clean = geo.dropna(subset=['CENTLAT', 'CENTLON']).copy()
geo_clean = geo_clean[geo_clean['PopulationValue'] >= 0]

# Ensure GEOID is string for matching
geo_clean['GEOID'] = geo_clean['GEOID'].astype(str)

log_table_stats('tigercensuscodes', geo_raw_rows, len(geo_clean), "Removed rows with missing coordinates")
geo_clean.to_csv(os.path.join(OUT_DIR, 'geo_lookup.csv'), index=False)
log(f"  Output: geo_lookup.csv ({len(geo_clean):,} rows)")
log("")


# ============================================================
# STEP 7: ENCOUNTERS (chunked processing)
# ============================================================
log("=" * 60)
log("STEP 7: Cleaning encounters.csv (chunked, 1.37GB)")
log("=" * 60)

# Only load columns needed for Clock Gap analysis
ENC_COLS = [
    'EncounterKey', 'PatientDurableKey', 'ProviderDurableKey',
    'AttendingProviderDurableKey', 'DepartmentKey', 'PrimaryDiagnosisKey',
    'AdmitYear', 'AdmitMonth', 'AdmitDay', 'AdmitHour', 'AdmitMinute',
    'Type', 'VisitType',
    'IsEdVisit', 'IsHospitalAdmission', 'IsOutpatientFaceToFaceVisit',
    'Date'
]

CHUNK_SIZE = 500_000
valid_pcp_keys = set(prov_pcp['DurableKey'].values)
valid_dept_keys = set(dept_clean['DepartmentKey'].values)
valid_diag_keys = set(diag_clean['DiagnosisKey'].values)

total_rows = 0
kept_rows = 0
removed_no_date = 0
removed_dup = 0
filled_from_date = 0
chunks_processed = 0

enc_chunks = []

log(f"  Loading columns: {len(ENC_COLS)} of 31")
log(f"  Chunk size: {CHUNK_SIZE:,}")
log(f"  Valid PCP keys: {len(valid_pcp_keys):,}")
log(f"  Strategy: Keep ALL rows with valid Date; parse year/month/day from Date if AdmitYear is NaN")
log(f"  Rows without AdmitHour will be kept but flagged has_time=False")
log("")

seen_keys = set()

for chunk in pd.read_csv(os.path.join(DATA_DIR, 'encounters.csv'), 
                         usecols=ENC_COLS, chunksize=CHUNK_SIZE):
    chunks_processed += 1
    chunk_rows = len(chunk)
    total_rows += chunk_rows
    
    # 1. Parse Date column to fill missing AdmitYear/Month/Day
    date_parsed = pd.to_datetime(chunk['Date'], format='mixed', dayfirst=False, errors='coerce')
    
    # Fill AdmitYear from Date where missing
    mask_no_year = chunk['AdmitYear'].isna() & date_parsed.notna()
    filled_from_date += mask_no_year.sum()
    chunk.loc[mask_no_year, 'AdmitYear'] = date_parsed[mask_no_year].dt.year
    chunk.loc[mask_no_year, 'AdmitMonth'] = date_parsed[mask_no_year].dt.month
    chunk.loc[mask_no_year, 'AdmitDay'] = date_parsed[mask_no_year].dt.day
    
    # 2. Remove rows with no date info at all
    no_date = chunk['AdmitYear'].isna()
    removed_no_date += no_date.sum()
    chunk = chunk[~no_date]
    
    # 3. Filter to 2022-2025
    valid_year = chunk['AdmitYear'].between(2022, 2025)
    chunk = chunk[valid_year]
    
    # 4. Deduplicate by EncounterKey
    dup_mask = chunk['EncounterKey'].isin(seen_keys)
    removed_dup += dup_mask.sum()
    chunk = chunk[~dup_mask]
    seen_keys.update(chunk['EncounterKey'].values)
    
    # 5. Mark has_time (whether AdmitHour is available)
    chunk['has_time'] = chunk['AdmitHour'].notna()
    
    # 6. Add time period (only for rows with hour)
    chunk['TimePeriod'] = np.where(
        chunk['has_time'],
        chunk['AdmitHour'].apply(lambda h: assign_time_period(int(h)) if pd.notna(h) else 'Unknown'),
        'Unknown'
    )
    
    # 7. Add weekday from parsed date
    chunk['Weekday'] = date_parsed.reindex(chunk.index).dt.dayofweek
    chunk['is_weekend'] = chunk['Weekday'].isin([5, 6])
    
    # 8. Mark PCP encounter
    chunk['is_pcp_encounter'] = chunk['ProviderDurableKey'].isin(valid_pcp_keys)
    
    # 9. Mark FK validity (don't remove, just flag)
    chunk['dept_valid'] = chunk['DepartmentKey'].isin(valid_dept_keys)
    chunk['diag_valid'] = chunk['PrimaryDiagnosisKey'].isin(valid_diag_keys)
    chunk['provider_valid'] = chunk['ProviderDurableKey'].notna() & (chunk['ProviderDurableKey'] > 0)
    
    # 10. Clean FK sentinel values (-1/-2/-3 -> NaN)
    for fk_col in ['DepartmentKey', 'PrimaryDiagnosisKey', 'ProviderDurableKey', 'AttendingProviderDurableKey']:
        chunk.loc[chunk[fk_col] < 0, fk_col] = np.nan
    
    enc_chunks.append(chunk)
    kept_rows += len(chunk)
    
    if chunks_processed % 3 == 0:
        log(f"  ... processed {chunks_processed} chunks ({total_rows:,} rows read)")

log(f"  Total chunks processed: {chunks_processed}")
log(f"  Total rows read: {total_rows:,}")
log(f"  Filled year/month/day from Date column: {filled_from_date:,}")
log(f"  Removed (no date at all): {removed_no_date:,}")
log(f"  Removed (duplicates): {removed_dup:,}")
log(f"  Kept rows: {kept_rows:,}")

# Concatenate
enc_clean = pd.concat(enc_chunks, ignore_index=True)
del enc_chunks  # free memory

# Drop Date column (we have year/month/day/hour)
enc_clean = enc_clean.drop(columns=['Date'], errors='ignore')

# Summary stats
has_time_count = enc_clean['has_time'].sum()
log(f"  Rows with time info (has_time=True): {has_time_count:,} ({has_time_count/len(enc_clean)*100:.1f}%)")
log(f"  Rows without time (date only): {len(enc_clean) - has_time_count:,} ({(len(enc_clean)-has_time_count)/len(enc_clean)*100:.1f}%)")
log(f"  PCP encounters: {enc_clean['is_pcp_encounter'].sum():,} ({enc_clean['is_pcp_encounter'].mean()*100:.1f}%)")
log(f"  ED visits: {enc_clean['IsEdVisit'].sum():,} ({enc_clean['IsEdVisit'].mean()*100:.1f}%)")
log(f"  Time period distribution (has_time rows only):")
tp_subset = enc_clean[enc_clean['has_time']]
tp_dist = tp_subset['TimePeriod'].value_counts()
for tp, cnt in tp_dist.items():
    log(f"    {tp}: {cnt:,} ({cnt/has_time_count*100:.1f}%)")

log(f"  Weekend encounters: {enc_clean['is_weekend'].sum():,} ({enc_clean['is_weekend'].mean()*100:.1f}%)")
log(f"  Year distribution:")
year_dist = enc_clean['AdmitYear'].value_counts().sort_index()
for yr, cnt in year_dist.items():
    log(f"    {int(yr)}: {cnt:,}")

# FK validity
log(f"  FK validity:")
log(f"    dept_valid: {enc_clean['dept_valid'].sum():,} ({enc_clean['dept_valid'].mean()*100:.1f}%)")
log(f"    diag_valid: {enc_clean['diag_valid'].sum():,} ({enc_clean['diag_valid'].mean()*100:.1f}%)")
log(f"    provider_valid: {enc_clean['provider_valid'].sum():,} ({enc_clean['provider_valid'].mean()*100:.1f}%)")

log_table_stats('encounters', total_rows, len(enc_clean), 
                "Parsed Date for NaN-year rows; flagged has_time; added TimePeriod/weekend/PCP/FK flags")

# Save
log("  Saving encounters_cleaned.csv (this may take a minute)...")
enc_clean.to_csv(os.path.join(OUT_DIR, 'encounters_cleaned.csv'), index=False)
log(f"  Output: encounters_cleaned.csv ({len(enc_clean):,} rows)")
log("")


# ============================================================
# STEP 8: GENERATE CLEANING REPORT
# ============================================================
log("=" * 60)
log("CLEANING COMPLETE - SUMMARY")
log("=" * 60)

summary = f"""
Output files in DATA_CLEANED/:
  1. departments_cleaned.csv   ({len(dept_clean):,} rows)
  2. providers_cleaned.csv     ({len(prov_clean):,} rows)
  3. providers_pcp.csv         ({len(prov_pcp):,} rows, PCP subset)
  4. patients_cleaned.csv      ({len(pat_clean):,} rows)
  5. diagnosis_cleaned.csv     ({len(diag_clean):,} rows)
  6. sdoh_cleaned.csv          ({len(sdoh_clean):,} rows)
  7. geo_lookup.csv            ({len(geo_clean):,} rows)
  8. encounters_cleaned.csv    ({len(enc_clean):,} rows)

Key derived columns in encounters_cleaned.csv:
  - has_time: True if AdmitHour was available (for time-period analysis)
  - TimePeriod: Day (7-17) / Evening (18-21) / Night (22-6) / Unknown (no hour)
  - is_weekend: True/False
  - is_pcp_encounter: True if ProviderDurableKey matches a PCP
  - dept_valid / diag_valid / provider_valid: FK integrity flags

Key derived columns in other tables:
  - providers: is_pcp flag
  - patients: has_geo flag
  - diagnosis: DiagCategory (Mental Health / Chronic / Acute / Other / Unknown)
  - sdoh: is_positive flag, Domain filled via mapping table
"""
log(summary)

# Write cleaning_report.md
report_path = os.path.join(OUT_DIR, 'cleaning_report.md')
with open(report_path, 'w') as f:
    f.write("# DataFest 2026 - Data Cleaning Report\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Topic: Clock Gap (Day/Night workload imbalance)\n\n")
    f.write("## Sentinel Value Treatment\n\n")
    f.write("| Sentinel | Meaning | Action |\n")
    f.write("|----------|---------|--------|\n")
    f.write("| *Not Applicable | N/A | Converted to NaN |\n")
    f.write("| *Unknown | Not recorded | Converted to NaN |\n")
    f.write("| *Deleted | Removed record | Converted to NaN |\n")
    f.write("| *Unspecified | Asked but not answered | Kept as 'Unspecified' category |\n")
    f.write("| -1/-2/-3 (FK) | Missing foreign key | Set to NaN, flagged with _valid column |\n\n")
    f.write("## Processing Log\n\n")
    f.write("```\n")
    f.write("\n".join(report_lines))
    f.write("\n```\n")

log(f"\nCleaning report saved to: {report_path}")
log("DONE.")
