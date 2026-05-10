"""
DataFest 2026 - 数据清理与挖掘一体化脚本
==========================================
模块:
1. 数据质量 Profiling（缺失率、类型分布、唯一值）
2. 统一缺失标记处理（6种标记）
3. 异常值检测
4. 外键完整性检查
5. 患者特征矩阵构建
6. 聚类分析（KMeans + PCA可视化）
7. 30天再入院风险因子分析（随机森林）
8. 时序 + 地理分析
9. 生成交互式 HTML 报告

输出文件:
- DATA_CLEANED/ 目录（parquet 中间数据）
- cleaning_mining_report.html（交互式报告）
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import json
from datetime import timedelta

warnings.filterwarnings('ignore')

# ============== 配置 ==============
DATA_DIR = "/Users/aaronwen/Documents/Datafest/DATA"
OUTPUT_DIR = "/Users/aaronwen/Documents/Datafest/DATA_CLEANED"
REPORT_PATH = "/Users/aaronwen/Documents/Datafest/cleaning_mining_report.html"
SDOH_MAPPING_PATH = "/Users/aaronwen/Documents/Datafest/Social Determinant Questions and Domains.csv"

CHUNK_SIZE = 500000  # encounters 分块大小

# 缺失值标记映射
MISSING_MARKERS = ['*Unspecified', '*Not Applicable', '*Unknown', '*Deleted']
SPECIAL_FK_VALUES = [-1, -2, -3]  # 外键特殊值

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DataFest 2026 - 数据清理与挖掘")
print("=" * 70)

# =====================================================================
# STEP 1: 数据质量 Profiling
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 1] 数据质量 Profiling")
print("=" * 70)

# 小文件直接读取
files_info = {}

# --- patients.csv ---
print("\n  [1/7] patients.csv...")
patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
files_info['patients'] = {
    'rows': len(patients), 'cols': len(patients.columns),
    'columns': list(patients.columns)
}
print(f"    行数: {len(patients):,}, 列数: {len(patients.columns)}")

# --- providers.csv ---
print("  [2/7] providers.csv...")
providers = pd.read_csv(os.path.join(DATA_DIR, "providers.csv"))
files_info['providers'] = {
    'rows': len(providers), 'cols': len(providers.columns),
    'columns': list(providers.columns)
}
print(f"    行数: {len(providers):,}, 列数: {len(providers.columns)}")

# --- departments.csv ---
print("  [3/7] departments.csv...")
departments = pd.read_csv(os.path.join(DATA_DIR, "departments.csv"))
files_info['departments'] = {
    'rows': len(departments), 'cols': len(departments.columns),
    'columns': list(departments.columns)
}
print(f"    行数: {len(departments):,}, 列数: {len(departments.columns)}")

# --- diagnosis.csv ---
print("  [4/7] diagnosis.csv...")
diagnosis = pd.read_csv(os.path.join(DATA_DIR, "diagnosis.csv"))
files_info['diagnosis'] = {
    'rows': len(diagnosis), 'cols': len(diagnosis.columns),
    'columns': list(diagnosis.columns)
}
print(f"    行数: {len(diagnosis):,}, 列数: {len(diagnosis.columns)}")

# --- tigercensuscodes.csv ---
print("  [5/7] tigercensuscodes.csv...")
tiger = pd.read_csv(os.path.join(DATA_DIR, "tigercensuscodes.csv"))
files_info['tigercensuscodes'] = {
    'rows': len(tiger), 'cols': len(tiger.columns),
    'columns': list(tiger.columns)
}
print(f"    行数: {len(tiger):,}, 列数: {len(tiger.columns)}")

# --- social_determinants.csv ---
print("  [6/7] social_determinants.csv...")
sdoh = pd.read_csv(os.path.join(DATA_DIR, "social_determinants.csv"))
files_info['social_determinants'] = {
    'rows': len(sdoh), 'cols': len(sdoh.columns),
    'columns': list(sdoh.columns)
}
print(f"    行数: {len(sdoh):,}, 列数: {len(sdoh.columns)}")

# --- encounters.csv (chunked profiling) ---
print("  [7/7] encounters.csv (chunked profiling)...")
enc_profile = {
    'total_rows': 0,
    'columns': None,
    'missing_counts': None,
    'marker_counts': {}
}

first_chunk = True
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"), chunksize=CHUNK_SIZE):
    enc_profile['total_rows'] += len(chunk)
    
    if first_chunk:
        enc_profile['columns'] = list(chunk.columns)
        enc_profile['missing_counts'] = chunk.isna().sum()
        # 统计特殊标记
        for marker in MISSING_MARKERS:
            enc_profile['marker_counts'][marker] = (chunk == marker).sum()
        first_chunk = False
    else:
        enc_profile['missing_counts'] += chunk.isna().sum()
        for marker in MISSING_MARKERS:
            enc_profile['marker_counts'][marker] += (chunk == marker).sum()
    
    print(f"    已处理 {enc_profile['total_rows']:,} 行...", end='\r')

files_info['encounters'] = {
    'rows': enc_profile['total_rows'],
    'cols': len(enc_profile['columns']),
    'columns': enc_profile['columns']
}
print(f"\n    行数: {enc_profile['total_rows']:,}, 列数: {len(enc_profile['columns'])}")

# 全面缺失率统计
print("\n  --- 各文件缺失率概览 ---")
profiling_results = {}

for name, df in [('patients', patients), ('providers', providers), 
                  ('departments', departments), ('diagnosis', diagnosis),
                  ('social_determinants', sdoh), ('tigercensuscodes', tiger)]:
    profile = {}
    for col in df.columns:
        na_count = df[col].isna().sum()
        marker_count = df[col].isin(MISSING_MARKERS).sum()
        if name in ['providers'] and col in ['DurableKey']:
            fk_count = df[col].isin(SPECIAL_FK_VALUES).sum()
        else:
            fk_count = 0
        total_missing = na_count + marker_count + fk_count
        profile[col] = {
            'na_count': int(na_count),
            'marker_count': int(marker_count),
            'fk_missing': int(fk_count),
            'total_missing': int(total_missing),
            'missing_pct': round(total_missing / len(df) * 100, 2),
            'dtype': str(df[col].dtype),
            'nunique': int(df[col].nunique())
        }
    profiling_results[name] = profile
    
    # 打印高缺失率列
    high_missing = {k: v for k, v in profile.items() if v['missing_pct'] > 5}
    if high_missing:
        print(f"\n    {name} - 缺失率>5%的列:")
        for col, info in sorted(high_missing.items(), key=lambda x: x[1]['missing_pct'], reverse=True):
            print(f"      {col}: {info['missing_pct']:.1f}% (NA={info['na_count']:,}, 标记={info['marker_count']:,})")

# encounters profiling
enc_profiling = {}
for col in enc_profile['columns']:
    na_count = int(enc_profile['missing_counts'].get(col, 0))
    marker_count = sum(int(enc_profile['marker_counts'][m].get(col, 0)) 
                       for m in MISSING_MARKERS if col in enc_profile['marker_counts'][m].index)
    total_missing = na_count + marker_count
    enc_profiling[col] = {
        'na_count': na_count,
        'marker_count': marker_count,
        'total_missing': total_missing,
        'missing_pct': round(total_missing / enc_profile['total_rows'] * 100, 2)
    }
profiling_results['encounters'] = enc_profiling

print(f"\n    encounters - 缺失率>5%的列:")
for col, info in sorted(enc_profiling.items(), key=lambda x: x[1]['missing_pct'], reverse=True):
    if info['missing_pct'] > 5:
        print(f"      {col}: {info['missing_pct']:.1f}%")


# =====================================================================
# STEP 2: 统一缺失标记处理 + 数据清理
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 2] 统一缺失标记处理")
print("=" * 70)

def clean_missing_markers(df, name=""):
    """统一处理缺失标记"""
    df = df.copy()
    cleaned_stats = {}
    
    for col in df.columns:
        original_na = df[col].isna().sum()
        
        # *Unknown → NaN（真缺失）
        unknown_mask = df[col].isin(['*Unknown'])
        unknown_count = unknown_mask.sum()
        df.loc[unknown_mask, col] = np.nan
        
        # *Deleted → 标记为删除（后续可选择排除）
        deleted_mask = df[col].isin(['*Deleted'])
        deleted_count = deleted_mask.sum()
        df.loc[deleted_mask, col] = np.nan
        
        # *Unspecified → 保留为特殊类别（有信息价值）
        # 不转换，保持原值作为一个有意义的类别
        unspec_count = (df[col] == '*Unspecified').sum()
        
        # *Not Applicable → NaN（结构性缺失）
        na_mask = df[col].isin(['*Not Applicable'])
        na_count = na_mask.sum()
        df.loc[na_mask, col] = np.nan
        
        if unknown_count + deleted_count + na_count + unspec_count > 0:
            cleaned_stats[col] = {
                'Unknown→NaN': int(unknown_count),
                'Deleted→NaN': int(deleted_count),
                'NotApplicable→NaN': int(na_count),
                'Unspecified_kept': int(unspec_count)
            }
    
    if cleaned_stats:
        print(f"    {name}: 处理了 {len(cleaned_stats)} 列的特殊标记")
    return df, cleaned_stats

# 清理各小表
print("\n  清理 patients...")
patients_clean, patients_stats = clean_missing_markers(patients, "patients")

# 处理外键 -1
fk_cols_patients = []  # patients 没有外键列需要处理
# 但 CensusBlockGroupFipsCode 中的 *Unspecified 保留

print("  清理 providers...")
providers_clean, providers_stats = clean_missing_markers(providers, "providers")
# 排除 Key = -1/-2/-3 的特殊行 (providers 表列名为 DurableKey)
providers_clean = providers_clean[~providers_clean['DurableKey'].isin(SPECIAL_FK_VALUES)]
print(f"    排除特殊键行后: {len(providers_clean):,} 行")

print("  清理 departments...")
departments_clean, departments_stats = clean_missing_markers(departments, "departments")
# 排除特殊键
if 'DepartmentKey' in departments_clean.columns:
    departments_clean = departments_clean[~departments_clean['DepartmentKey'].isin(SPECIAL_FK_VALUES)]
    print(f"    排除特殊键行后: {len(departments_clean):,} 行")

print("  清理 diagnosis...")
diagnosis_clean, diagnosis_stats = clean_missing_markers(diagnosis, "diagnosis")

print("  清理 social_determinants...")
sdoh_clean, sdoh_stats = clean_missing_markers(sdoh, "social_determinants")

# 用映射表回填 Domain
print("  回填 social_determinants.Domain（通过 DisplayName 映射）...")
sdoh_mapping = pd.read_csv(SDOH_MAPPING_PATH)
display_to_domain = dict(zip(sdoh_mapping['DisplayName'], sdoh_mapping['Domain']))

# 对 Domain 为空的行，用 DisplayName 查找
domain_null_mask = sdoh_clean['Domain'].isna()
domain_null_before = domain_null_mask.sum()
sdoh_clean.loc[domain_null_mask, 'Domain'] = sdoh_clean.loc[domain_null_mask, 'DisplayName'].map(display_to_domain)
domain_null_after = sdoh_clean['Domain'].isna().sum()
print(f"    Domain 回填: {domain_null_before:,} → {domain_null_after:,} (填补了 {domain_null_before - domain_null_after:,} 行)")

# =====================================================================
# STEP 3: 异常值检测
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 3] 异常值检测")
print("=" * 70)

anomaly_results = {}

# patients: 检查 PatientBirthYearBin 合理性
if 'PatientBirthYearBin' in patients_clean.columns:
    birth_years = patients_clean['PatientBirthYearBin'].dropna()
    outlier_birth = birth_years[(birth_years < 1900) | (birth_years > 2025)]
    anomaly_results['patients_BirthYear'] = {
        'total': len(birth_years),
        'outliers': len(outlier_birth),
        'range': f"[{birth_years.min()}, {birth_years.max()}]"
    }
    print(f"  patients.PatientBirthYearBin: 范围 {birth_years.min()}-{birth_years.max()}, 异常 {len(outlier_birth)} 行")

# encounters: 用 chunked 方式检查日期逻辑
print("  encounters: 检查日期逻辑异常（chunked）...")
date_anomalies = {'future_dates': 0, 'discharge_before_admit': 0, 'los_over_365': 0}
encounter_date_stats = {'min_date': None, 'max_date': None}

for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"), 
                         usecols=['Date', 'DischargeInstant', 'AdmissionInstant'],
                         chunksize=CHUNK_SIZE):
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk['DischargeInstant'] = pd.to_datetime(chunk['DischargeInstant'], errors='coerce')
    chunk['AdmissionInstant'] = pd.to_datetime(chunk['AdmissionInstant'], errors='coerce')
    
    # 未来日期
    future = (chunk['Date'] > pd.Timestamp('2025-12-31')).sum()
    date_anomalies['future_dates'] += future
    
    # 出院早于入院
    valid_both = chunk[chunk['DischargeInstant'].notna() & chunk['AdmissionInstant'].notna()]
    discharge_before = (valid_both['DischargeInstant'] < valid_both['AdmissionInstant']).sum()
    date_anomalies['discharge_before_admit'] += discharge_before
    
    # 住院>365天
    los = (valid_both['DischargeInstant'] - valid_both['AdmissionInstant']).dt.days
    long_stay = (los > 365).sum()
    date_anomalies['los_over_365'] += long_stay
    
    # 日期范围
    chunk_min = chunk['Date'].min()
    chunk_max = chunk['Date'].max()
    if encounter_date_stats['min_date'] is None or chunk_min < encounter_date_stats['min_date']:
        encounter_date_stats['min_date'] = chunk_min
    if encounter_date_stats['max_date'] is None or chunk_max > encounter_date_stats['max_date']:
        encounter_date_stats['max_date'] = chunk_max

anomaly_results['encounters_dates'] = date_anomalies
print(f"    日期范围: {encounter_date_stats['min_date']} ~ {encounter_date_stats['max_date']}")
print(f"    未来日期: {date_anomalies['future_dates']:,}")
print(f"    出院<入院: {date_anomalies['discharge_before_admit']:,}")
print(f"    住院>365天: {date_anomalies['los_over_365']:,}")

# =====================================================================
# STEP 4: 外键完整性检查
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 4] 外键完整性检查")
print("=" * 70)

fk_integrity = {}

# encounters.PrimaryDiagnosisKey → diagnosis.DiagnosisKey
valid_diag_keys = set(diagnosis_clean['DiagnosisKey'].unique())
print("  encounters.PrimaryDiagnosisKey → diagnosis.DiagnosisKey...")

orphan_diag_count = 0
total_with_diag = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=['PrimaryDiagnosisKey'], chunksize=CHUNK_SIZE):
    valid_mask = chunk['PrimaryDiagnosisKey'].notna()
    total_with_diag += valid_mask.sum()
    orphan_mask = valid_mask & ~chunk['PrimaryDiagnosisKey'].isin(valid_diag_keys)
    orphan_diag_count += orphan_mask.sum()

fk_integrity['PrimaryDiagnosisKey'] = {
    'total_non_null': total_with_diag,
    'orphans': orphan_diag_count,
    'orphan_pct': round(orphan_diag_count / max(total_with_diag, 1) * 100, 4)
}
print(f"    有效: {total_with_diag:,}, 孤儿键: {orphan_diag_count:,} ({fk_integrity['PrimaryDiagnosisKey']['orphan_pct']}%)")

# encounters.DepartmentKey → departments.DepartmentKey
valid_dept_keys = set(departments_clean['DepartmentKey'].unique()) if 'DepartmentKey' in departments_clean.columns else set()
print("  encounters.DepartmentKey → departments.DepartmentKey...")

orphan_dept_count = 0
total_with_dept = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=['DepartmentKey'], chunksize=CHUNK_SIZE):
    valid_mask = chunk['DepartmentKey'].notna()
    total_with_dept += valid_mask.sum()
    orphan_mask = valid_mask & ~chunk['DepartmentKey'].isin(valid_dept_keys)
    orphan_dept_count += orphan_mask.sum()

fk_integrity['DepartmentKey'] = {
    'total_non_null': total_with_dept,
    'orphans': orphan_dept_count,
    'orphan_pct': round(orphan_dept_count / max(total_with_dept, 1) * 100, 4)
}
print(f"    有效: {total_with_dept:,}, 孤儿键: {orphan_dept_count:,} ({fk_integrity['DepartmentKey']['orphan_pct']}%)")

# encounters.ProviderDurableKey → providers.DurableKey
valid_provider_keys = set(providers_clean['DurableKey'].unique())
print("  encounters.ProviderDurableKey → providers.ProviderDurableKey...")

orphan_prov_count = 0
total_with_prov = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=['ProviderDurableKey'], chunksize=CHUNK_SIZE):
    # 排除 -1 外键
    valid_mask = chunk['ProviderDurableKey'].notna() & ~chunk['ProviderDurableKey'].isin(SPECIAL_FK_VALUES)
    total_with_prov += valid_mask.sum()
    orphan_mask = valid_mask & ~chunk['ProviderDurableKey'].isin(valid_provider_keys)
    orphan_prov_count += orphan_mask.sum()

fk_integrity['ProviderDurableKey'] = {
    'total_non_null': total_with_prov,
    'orphans': orphan_prov_count,
    'orphan_pct': round(orphan_prov_count / max(total_with_prov, 1) * 100, 4)
}
print(f"    有效: {total_with_prov:,}, 孤儿键: {orphan_prov_count:,} ({fk_integrity['ProviderDurableKey']['orphan_pct']}%)")

# =====================================================================
# STEP 5: 构建患者特征矩阵
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 5] 构建患者特征矩阵")
print("=" * 70)

# 5a: 从 encounters 提取患者级聚合特征
print("  从 encounters 提取患者级特征（chunked）...")

patient_agg = {}  # PatientDurableKey -> features dict

enc_cols_needed = ['PatientDurableKey', 'Date', 'IsEdVisit', 'IsHospitalAdmission',
                   'IsInpatientAdmission', 'PrimaryDiagnosisKey', 'DepartmentKey',
                   'DischargeInstant', 'AdmissionInstant']

for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=enc_cols_needed, chunksize=CHUNK_SIZE):
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    
    for pid, group in chunk.groupby('PatientDurableKey'):
        if pid not in patient_agg:
            patient_agg[pid] = {
                'total_encounters': 0,
                'ed_visits': 0,
                'hospital_admissions': 0,
                'inpatient_admissions': 0,
                'unique_diagnoses': set(),
                'unique_departments': set(),
                'first_encounter': None,
                'last_encounter': None,
            }
        
        pa = patient_agg[pid]
        pa['total_encounters'] += len(group)
        pa['ed_visits'] += (group['IsEdVisit'] == True).sum() + (group['IsEdVisit'].astype(str).str.lower() == 'true').sum()
        pa['hospital_admissions'] += (group['IsHospitalAdmission'] == True).sum()
        pa['inpatient_admissions'] += (group['IsInpatientAdmission'] == True).sum()
        
        diags = group['PrimaryDiagnosisKey'].dropna().unique()
        pa['unique_diagnoses'].update(diags)
        
        depts = group['DepartmentKey'].dropna().unique()
        pa['unique_departments'].update(depts)
        
        dates = group['Date'].dropna()
        if len(dates) > 0:
            min_d = dates.min()
            max_d = dates.max()
            if pa['first_encounter'] is None or min_d < pa['first_encounter']:
                pa['first_encounter'] = min_d
            if pa['last_encounter'] is None or max_d > pa['last_encounter']:
                pa['last_encounter'] = max_d
    
    print(f"    已处理患者数: {len(patient_agg):,}", end='\r')

print(f"\n    总患者数: {len(patient_agg):,}")

# 构建特征 DataFrame
print("  构建特征矩阵...")
patient_features_list = []
for pid, feat in patient_agg.items():
    # 计算就诊跨度（天）
    if feat['first_encounter'] and feat['last_encounter']:
        span_days = (feat['last_encounter'] - feat['first_encounter']).days
    else:
        span_days = 0
    
    patient_features_list.append({
        'PatientDurableKey': pid,
        'total_encounters': feat['total_encounters'],
        'ed_visits': feat['ed_visits'],
        'hospital_admissions': feat['hospital_admissions'],
        'inpatient_admissions': feat['inpatient_admissions'],
        'unique_diagnoses': len(feat['unique_diagnoses']),
        'unique_departments': len(feat['unique_departments']),
        'span_days': span_days,
        'encounters_per_month': feat['total_encounters'] / max(span_days / 30.44, 1),
        'ed_ratio': feat['ed_visits'] / max(feat['total_encounters'], 1),
        'inpatient_ratio': feat['inpatient_admissions'] / max(feat['total_encounters'], 1),
    })

patient_features = pd.DataFrame(patient_features_list)

# 5b: 合并患者人口统计学
print("  合并人口统计学...")
# patients 表用 DurableKey 而非 PatientDurableKey
patients_demo = patients_clean[['DurableKey', 'PatientBirthYearBin', 'SexAssignedAtBirth', 
                                 'FirstRace', 'OmbEthnicity']].copy()
patients_demo = patients_demo.rename(columns={'DurableKey': 'PatientDurableKey'})
# 编码 Sex
patients_demo['is_female'] = (patients_demo['SexAssignedAtBirth'] == 'Female').astype(int)
# 计算近似年龄
patients_demo['approx_age'] = 2025 - patients_demo['PatientBirthYearBin']

patient_features = patient_features.merge(
    patients_demo[['PatientDurableKey', 'is_female', 'approx_age']], 
    on='PatientDurableKey', how='left'
)

# 5c: SDOH 特征
print("  计算 SDOH 暴露特征...")
sdoh_by_patient = sdoh_clean.groupby('PatientDurableKey').agg(
    sdoh_total_screenings=('EncounterKey', 'count'),
    sdoh_unique_domains=('Domain', 'nunique'),
    sdoh_domains_list=('Domain', lambda x: list(x.dropna().unique()))
).reset_index()

# 为每个 domain 创建 0/1 标记
all_domains = sdoh_clean['Domain'].dropna().unique()
for domain in all_domains:
    domain_patients = set(sdoh_clean[sdoh_clean['Domain'] == domain]['PatientDurableKey'].unique())
    col_name = f"sdoh_{domain.replace(' ', '_').lower()}"
    sdoh_by_patient[col_name] = sdoh_by_patient['PatientDurableKey'].isin(domain_patients).astype(int)

patient_features = patient_features.merge(
    sdoh_by_patient.drop(columns=['sdoh_domains_list']),
    on='PatientDurableKey', how='left'
)

# 填充 SDOH 缺失为 0（未被筛查 = 无 SDOH 记录）
sdoh_cols = [c for c in patient_features.columns if c.startswith('sdoh_')]
patient_features[sdoh_cols] = patient_features[sdoh_cols].fillna(0)

# 5d: 30天再入院标签
print("  计算 30 天再入院标签...")
readmit_patients = set()

inpatient_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=['PatientDurableKey', 'Date', 'DischargeInstant', 
                                  'IsInpatientAdmission'],
                         chunksize=CHUNK_SIZE):
    mask = chunk['IsInpatientAdmission'] == True
    if mask.sum() == 0:
        mask = chunk['IsInpatientAdmission'].astype(str).str.strip().str.lower().isin(['true', '1'])
    filtered = chunk[mask].copy()
    if len(filtered) > 0:
        inpatient_chunks.append(filtered[['PatientDurableKey', 'Date', 'DischargeInstant']])

if inpatient_chunks:
    inpatient_enc = pd.concat(inpatient_chunks, ignore_index=True)
    inpatient_enc['Date'] = pd.to_datetime(inpatient_enc['Date'], errors='coerce')
    inpatient_enc['DischargeInstant'] = pd.to_datetime(inpatient_enc['DischargeInstant'], errors='coerce')
    inpatient_enc = inpatient_enc.sort_values(['PatientDurableKey', 'Date'])
    
    inpatient_enc['NextAdmitDate'] = inpatient_enc.groupby('PatientDurableKey')['Date'].shift(-1)
    inpatient_enc['DaysToNext'] = (inpatient_enc['NextAdmitDate'] - inpatient_enc['DischargeInstant']).dt.days
    
    valid_discharges = inpatient_enc[inpatient_enc['DischargeInstant'].notna()]
    readmit_30 = valid_discharges[valid_discharges['DaysToNext'].between(0, 30)]
    readmit_patients = set(readmit_30['PatientDurableKey'].unique())
    
    overall_readmit_rate = len(readmit_30) / max(len(valid_discharges), 1)
    print(f"    Inpatient admissions: {len(inpatient_enc):,}")
    print(f"    30天再入院次数: {len(readmit_30):,}")
    print(f"    30天再入院率: {overall_readmit_rate*100:.2f}%")
    print(f"    有再入院的患者数: {len(readmit_patients):,}")
else:
    overall_readmit_rate = 0
    print("    ⚠️ 未找到 inpatient admissions")

patient_features['readmit_30d'] = patient_features['PatientDurableKey'].isin(readmit_patients).astype(int)
print(f"  特征矩阵形状: {patient_features.shape}")

# 保存
patient_features.to_csv(os.path.join(OUTPUT_DIR, "patient_features.csv"), index=False)
print(f"  ✅ 特征矩阵已保存: {OUTPUT_DIR}/patient_features.csv")

# =====================================================================
# STEP 6: 聚类分析
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 6] 聚类分析 (KMeans + PCA)")
print("=" * 70)

# 选择数值特征用于聚类
cluster_features = ['total_encounters', 'ed_visits', 'inpatient_admissions',
                    'unique_diagnoses', 'unique_departments', 'span_days',
                    'encounters_per_month', 'ed_ratio', 'inpatient_ratio',
                    'is_female', 'approx_age', 'sdoh_total_screenings', 'sdoh_unique_domains']

# 加入可用的 SDOH domain 列
cluster_features += [c for c in sdoh_cols if c not in cluster_features]

# 过滤有效行
cluster_df = patient_features[cluster_features].copy()
cluster_df = cluster_df.fillna(0)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# 确定最优 k（silhouette score）
print("  寻找最优聚类数 k...")
silhouette_scores = {}
# 抽样加速（如果患者太多）
if len(X_scaled) > 50000:
    sample_idx = np.random.choice(len(X_scaled), 50000, replace=False)
    X_sample = X_scaled[sample_idx]
else:
    X_sample = X_scaled
    sample_idx = np.arange(len(X_scaled))

for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    score = silhouette_score(X_sample, labels, sample_size=min(10000, len(X_sample)))
    silhouette_scores[k] = score
    print(f"    k={k}: silhouette={score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"  最优 k = {best_k} (silhouette = {silhouette_scores[best_k]:.4f})")

# 最终聚类
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
patient_features['cluster'] = km_final.fit_predict(X_scaled)

# PCA 降维可视化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
patient_features['pca_1'] = X_pca[:, 0]
patient_features['pca_2'] = X_pca[:, 1]

# 各簇特征均值
cluster_summary = patient_features.groupby('cluster')[cluster_features].mean()
print(f"\n  各簇均值摘要:")
print(cluster_summary[['total_encounters', 'ed_visits', 'inpatient_admissions', 
                        'unique_diagnoses', 'approx_age', 'sdoh_total_screenings']].round(1).to_string())

# 各簇再入院率
cluster_readmit = patient_features.groupby('cluster')['readmit_30d'].mean() * 100
print(f"\n  各簇30天再入院率:")
for c, rate in cluster_readmit.items():
    print(f"    Cluster {c}: {rate:.2f}%")

# =====================================================================
# STEP 7: 风险因子分析 (随机森林)
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 7] 30天再入院风险因子分析 (Random Forest)")
print("=" * 70)

# 只对有住院记录的患者建模
inpatient_patients = set()
if inpatient_chunks:
    inpatient_patients = set(inpatient_enc['PatientDurableKey'].unique())

model_df = patient_features[patient_features['PatientDurableKey'].isin(inpatient_patients)].copy()
print(f"  建模样本数: {len(model_df):,} (有住院记录的患者)")
print(f"  正例(再入院): {model_df['readmit_30d'].sum():,} ({model_df['readmit_30d'].mean()*100:.2f}%)")

# 特征列
feature_cols = [c for c in cluster_features if c in model_df.columns]
X = model_df[feature_cols].fillna(0)
y = model_df['readmit_30d']

# 训练随机森林
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# 特征重要性
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 风险因子:")
for i, row in importances.head(15).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# =====================================================================
# STEP 8: 时序 + 地理分析
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 8] 时序与地理分析")
print("=" * 70)

# 8a: 月度就诊量趋势
print("  计算月度就诊量趋势...")
monthly_encounters = {}

for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=['Date', 'IsEdVisit', 'IsInpatientAdmission'],
                         chunksize=CHUNK_SIZE):
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk['YearMonth'] = chunk['Date'].dt.to_period('M').astype(str)
    
    monthly = chunk.groupby('YearMonth').agg(
        total=('Date', 'count'),
        ed=('IsEdVisit', lambda x: (x == True).sum()),
        inpatient=('IsInpatientAdmission', lambda x: (x == True).sum())
    )
    
    for ym, row in monthly.iterrows():
        if ym not in monthly_encounters:
            monthly_encounters[ym] = {'total': 0, 'ed': 0, 'inpatient': 0}
        monthly_encounters[ym]['total'] += row['total']
        monthly_encounters[ym]['ed'] += row['ed']
        monthly_encounters[ym]['inpatient'] += row['inpatient']

monthly_df = pd.DataFrame(monthly_encounters).T.reset_index()
monthly_df.columns = ['YearMonth', 'Total', 'ED', 'Inpatient']
monthly_df = monthly_df.sort_values('YearMonth')
print(f"    月份范围: {monthly_df['YearMonth'].iloc[0]} ~ {monthly_df['YearMonth'].iloc[-1]}")

# 8b: 地理分析 - Census Block Group 患者分布
print("  地理分析: Census Block Group 患者分布...")
# patients 表用 DurableKey
geo_patients = patients_clean[['DurableKey', 'CensusBlockGroupFipsCode']].copy()
geo_patients = geo_patients.rename(columns={'DurableKey': 'PatientDurableKey'})
geo_patients = geo_patients[
    geo_patients['CensusBlockGroupFipsCode'].notna() & 
    (geo_patients['CensusBlockGroupFipsCode'] != '*Unspecified')
]

# 每个 Census Block 的患者数和再入院率
geo_patients['readmit_30d'] = geo_patients['PatientDurableKey'].isin(readmit_patients).astype(int)
geo_summary = geo_patients.groupby('CensusBlockGroupFipsCode').agg(
    patient_count=('PatientDurableKey', 'count'),
    readmit_count=('readmit_30d', 'sum'),
    readmit_rate=('readmit_30d', 'mean')
).reset_index()

# 关联 tiger census codes
geo_summary = geo_summary.merge(
    tiger, left_on='CensusBlockGroupFipsCode', right_on='GEOID', how='left'
)

print(f"    有地理编码的患者: {len(geo_patients):,}")
print(f"    唯一 Census Block Groups: {geo_summary['CensusBlockGroupFipsCode'].nunique():,}")
print(f"    再入院率最高区域 top5:")
top_geo = geo_summary[geo_summary['patient_count'] >= 50].nlargest(5, 'readmit_rate')
for _, row in top_geo.iterrows():
    print(f"      {row['CensusBlockGroupFipsCode']}: 率={row['readmit_rate']*100:.1f}%, 患者={row['patient_count']}")

# =====================================================================
# STEP 9: 生成交互式 HTML 报告
# =====================================================================
print("\n" + "=" * 70)
print("[STEP 9] 生成交互式 HTML 报告")
print("=" * 70)

# --- 图1: 缺失率热力图 ---
missing_data = []
for fname, profile in profiling_results.items():
    for col, info in profile.items():
        pct = info.get('missing_pct', info.get('missing_pct', 0))
        if pct > 0:
            missing_data.append({'file': fname, 'column': col, 'missing_pct': pct})

missing_df = pd.DataFrame(missing_data)
if len(missing_df) > 0:
    # 取 top 30 缺失率最高的
    top_missing = missing_df.nlargest(30, 'missing_pct')
    fig_missing = px.bar(
        top_missing, x='missing_pct', y='column', color='file',
        orientation='h', title='数据缺失率 Top 30 (含特殊标记)',
        labels={'missing_pct': '缺失率 (%)', 'column': '字段', 'file': '文件'},
        template='plotly_white'
    )
    fig_missing.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
else:
    fig_missing = go.Figure()

# --- 图2: PCA 聚类散点图 ---
# 抽样以避免图表过大
sample_size = min(20000, len(patient_features))
plot_sample = patient_features.sample(sample_size, random_state=42)

fig_cluster = px.scatter(
    plot_sample, x='pca_1', y='pca_2', color='cluster',
    color_continuous_scale='Viridis',
    title=f'患者聚类 (KMeans k={best_k}, PCA 2D投影)',
    labels={'pca_1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'pca_2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
    template='plotly_white',
    opacity=0.6
)
fig_cluster.update_layout(height=600)

# --- 图3: 风险因子重要性 ---
fig_importance = px.bar(
    importances.head(20), x='importance', y='feature',
    orientation='h', title='30天再入院风险因子 Top 20 (Random Forest)',
    labels={'importance': 'Feature Importance', 'feature': '特征'},
    template='plotly_white',
    color='importance', color_continuous_scale='Reds'
)
fig_importance.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})

# --- 图4: 月度就诊趋势 ---
fig_timeline = make_subplots(rows=1, cols=1)
fig_timeline.add_trace(go.Scatter(
    x=monthly_df['YearMonth'], y=monthly_df['Total'],
    mode='lines+markers', name='总就诊', line=dict(width=2, color='steelblue')
))
fig_timeline.add_trace(go.Scatter(
    x=monthly_df['YearMonth'], y=monthly_df['ED'],
    mode='lines+markers', name='ED就诊', line=dict(width=2, color='coral')
))
fig_timeline.add_trace(go.Scatter(
    x=monthly_df['YearMonth'], y=monthly_df['Inpatient'],
    mode='lines+markers', name='住院', line=dict(width=2, color='forestgreen')
))
fig_timeline.update_layout(
    title='月度就诊量趋势', template='plotly_white',
    xaxis_title='月份', yaxis_title='就诊次数', height=500
)

# --- 图5: 各簇特征雷达图 ---
radar_features = ['total_encounters', 'ed_ratio', 'inpatient_ratio', 
                  'unique_diagnoses', 'sdoh_total_screenings', 'approx_age']
cluster_means = patient_features.groupby('cluster')[radar_features].mean()

# 标准化到 0-1
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-10)

fig_radar = go.Figure()
for cluster_id in range(best_k):
    values = cluster_means_norm.loc[cluster_id].values.tolist()
    values.append(values[0])  # 闭合
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name=f'Cluster {cluster_id}'
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title='各患者簇特征画像 (雷达图, 归一化)',
    template='plotly_white', height=550
)

# --- 图6: SDOH Domain 与再入院率 ---
sdoh_readmit_data = []
for domain in all_domains:
    col = f"sdoh_{domain.replace(' ', '_').lower()}"
    if col in patient_features.columns:
        exposed = patient_features[patient_features[col] == 1]
        not_exposed = patient_features[patient_features[col] == 0]
        if len(exposed) > 0:
            sdoh_readmit_data.append({
                'Domain': domain,
                'Exposed_Rate': exposed['readmit_30d'].mean() * 100,
                'Not_Exposed_Rate': not_exposed['readmit_30d'].mean() * 100,
                'Exposed_N': len(exposed)
            })

sdoh_readmit_df = pd.DataFrame(sdoh_readmit_data)
if len(sdoh_readmit_df) > 0:
    fig_sdoh = go.Figure()
    fig_sdoh.add_trace(go.Bar(
        x=sdoh_readmit_df['Domain'], y=sdoh_readmit_df['Exposed_Rate'],
        name='有SDOH风险', marker_color='#e74c3c'
    ))
    fig_sdoh.add_trace(go.Bar(
        x=sdoh_readmit_df['Domain'], y=sdoh_readmit_df['Not_Exposed_Rate'],
        name='无SDOH风险', marker_color='#3498db'
    ))
    fig_sdoh.update_layout(
        title='SDOH 风险暴露与30天再入院率',
        barmode='group', template='plotly_white',
        xaxis_tickangle=-30, height=500,
        yaxis_title='30天再入院率(%)'
    )
else:
    fig_sdoh = go.Figure()

# --- 图7: 地理分布 ---
if len(geo_summary) > 0 and 'TractCensus' in geo_summary.columns:
    fig_geo = px.scatter(
        geo_summary[geo_summary['patient_count'] >= 10],
        x='patient_count', y='readmit_rate',
        size='patient_count', color='readmit_rate',
        color_continuous_scale='RdYlGn_r',
        title='Census Block Group: 患者数 vs 再入院率',
        labels={'patient_count': '患者数', 'readmit_rate': '30天再入院率'},
        template='plotly_white', height=500
    )
else:
    fig_geo = px.scatter(
        geo_summary[geo_summary['patient_count'] >= 10],
        x='patient_count', y='readmit_rate',
        size='patient_count', color='readmit_rate',
        color_continuous_scale='RdYlGn_r',
        title='Census Block Group: 患者数 vs 再入院率',
        labels={'patient_count': '患者数', 'readmit_rate': '30天再入院率'},
        template='plotly_white', height=500
    )

# --- 图8: Silhouette scores ---
fig_silhouette = go.Figure(go.Bar(
    x=list(silhouette_scores.keys()),
    y=list(silhouette_scores.values()),
    marker_color=['#e74c3c' if k == best_k else '#3498db' for k in silhouette_scores.keys()],
    text=[f"{v:.3f}" for v in silhouette_scores.values()],
    textposition='auto'
))
fig_silhouette.update_layout(
    title=f'聚类评估: Silhouette Score (最优 k={best_k})',
    xaxis_title='聚类数 k', yaxis_title='Silhouette Score',
    template='plotly_white', height=400
)

# === 组装 HTML ===
print("  组装 HTML 报告...")

# 各簇统计表格
cluster_table_rows = ""
for c in range(best_k):
    c_data = patient_features[patient_features['cluster'] == c]
    cluster_table_rows += f"""
    <tr>
        <td><strong>Cluster {c}</strong></td>
        <td>{len(c_data):,}</td>
        <td>{c_data['total_encounters'].mean():.1f}</td>
        <td>{c_data['ed_ratio'].mean()*100:.1f}%</td>
        <td>{c_data['inpatient_ratio'].mean()*100:.1f}%</td>
        <td>{c_data['unique_diagnoses'].mean():.1f}</td>
        <td>{c_data['sdoh_total_screenings'].mean():.1f}</td>
        <td>{c_data['readmit_30d'].mean()*100:.2f}%</td>
    </tr>"""

html_report = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>DataFest 2026 - 数据清理与挖掘报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif;
            max-width: 1300px;
            margin: 0 auto;
            padding: 30px;
            background: #f0f2f5;
            color: #2c3e50;
            line-height: 1.6;
        }}
        h1 {{
            color: #1a1a2e;
            font-size: 2em;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 50px;
            padding: 10px 0;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{ color: #34495e; margin-top: 30px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-3px); }}
        .stat-value {{
            font-size: 2.2em;
            font-weight: 700;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .chart-box {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 25px 0;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #edf2f7;
        }}
        tr:hover {{ background: #f7fafc; }}
        .insight-box {{
            background: #eaf4fe;
            border-left: 5px solid #3498db;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .warning-box {{
            background: #fff8e1;
            border-left: 5px solid #f39c12;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .section-intro {{
            color: #5d6d7e;
            font-size: 0.95em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>📊 DataFest 2026 - 数据清理与挖掘报告</h1>
    <p style="color:#7f8c8d;">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 数据覆盖: {encounter_date_stats['min_date'].strftime('%Y-%m') if encounter_date_stats['min_date'] else 'N/A'} ~ {encounter_date_stats['max_date'].strftime('%Y-%m') if encounter_date_stats['max_date'] else 'N/A'}</p>

    <!-- ===== SECTION 1: 数据概览 ===== -->
    <h2>1. 数据集概览</h2>
    <div class="summary-grid">
        <div class="stat-card">
            <div class="stat-value">7</div>
            <div class="stat-label">CSV 文件</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{enc_profile['total_rows']:,}</div>
            <div class="stat-label">总就诊记录</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(patient_features):,}</div>
            <div class="stat-label">唯一患者数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(diagnosis):,}</div>
            <div class="stat-label">诊断记录</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(sdoh):,}</div>
            <div class="stat-label">SDOH 筛查记录</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{overall_readmit_rate*100:.1f}%</div>
            <div class="stat-label">30天再入院率</div>
        </div>
    </div>

    <!-- ===== SECTION 2: 数据质量 ===== -->
    <h2>2. 数据质量 Profiling</h2>
    <p class="section-intro">数据集使用 6 种缺失标记: NA, *Unspecified (问了未答), *Not Applicable (不适用), *Unknown (未记录), *Deleted (已删除), -1 (外键缺失)。以下展示缺失率最高的字段。</p>
    
    <div class="chart-box" id="chart_missing"></div>

    <h3>异常值检测结果</h3>
    <div class="insight-box">
        <strong>日期逻辑检查:</strong><br>
        • 日期范围: {encounter_date_stats['min_date'].strftime('%Y-%m-%d') if encounter_date_stats['min_date'] else 'N/A'} ~ {encounter_date_stats['max_date'].strftime('%Y-%m-%d') if encounter_date_stats['max_date'] else 'N/A'}<br>
        • 未来日期异常: {date_anomalies['future_dates']:,} 行<br>
        • 出院早于入院: {date_anomalies['discharge_before_admit']:,} 行<br>
        • 住院>365天: {date_anomalies['los_over_365']:,} 行
    </div>

    <h3>外键完整性</h3>
    <table>
        <tr><th>外键关系</th><th>有效记录</th><th>孤儿键</th><th>孤儿率</th></tr>
        <tr><td>encounters.PrimaryDiagnosisKey → diagnosis</td><td>{fk_integrity['PrimaryDiagnosisKey']['total_non_null']:,}</td><td>{fk_integrity['PrimaryDiagnosisKey']['orphans']:,}</td><td>{fk_integrity['PrimaryDiagnosisKey']['orphan_pct']}%</td></tr>
        <tr><td>encounters.DepartmentKey → departments</td><td>{fk_integrity['DepartmentKey']['total_non_null']:,}</td><td>{fk_integrity['DepartmentKey']['orphans']:,}</td><td>{fk_integrity['DepartmentKey']['orphan_pct']}%</td></tr>
        <tr><td>encounters.ProviderDurableKey → providers</td><td>{fk_integrity['ProviderDurableKey']['total_non_null']:,}</td><td>{fk_integrity['ProviderDurableKey']['orphans']:,}</td><td>{fk_integrity['ProviderDurableKey']['orphan_pct']}%</td></tr>
    </table>

    <!-- ===== SECTION 3: 时序分析 ===== -->
    <h2>3. 时序分析</h2>
    <p class="section-intro">月度就诊量趋势，包含总就诊、急诊(ED)和住院三条线。</p>
    <div class="chart-box" id="chart_timeline"></div>

    <!-- ===== SECTION 4: 聚类分析 ===== -->
    <h2>4. 患者聚类分析</h2>
    <p class="section-intro">基于就诊频率、急诊使用率、住院率、诊断多样性、SDOH 暴露等特征，使用 KMeans 对患者进行分群。</p>
    
    <div class="chart-box" id="chart_silhouette"></div>
    <div class="chart-box" id="chart_cluster"></div>
    <div class="chart-box" id="chart_radar"></div>

    <h3>各簇统计摘要</h3>
    <table>
        <tr><th>簇</th><th>患者数</th><th>平均就诊次数</th><th>ED比例</th><th>住院比例</th><th>诊断多样性</th><th>SDOH筛查</th><th>30天再入院率</th></tr>
        {cluster_table_rows}
    </table>

    <!-- ===== SECTION 5: 风险因子 ===== -->
    <h2>5. 30天再入院风险因子</h2>
    <p class="section-intro">使用随机森林(n=200, balanced)对有住院记录的{len(model_df):,}名患者建模，提取特征重要性排序。</p>
    <div class="chart-box" id="chart_importance"></div>

    <div class="insight-box">
        <strong>关键发现:</strong> Top 5 风险因子为: {', '.join(importances.head(5)['feature'].tolist())}
    </div>

    <!-- ===== SECTION 6: SDOH 分析 ===== -->
    <h2>6. SDOH 风险暴露分析</h2>
    <p class="section-intro">比较有/无各 SDOH domain 风险暴露的患者，其 30天再入院率的差异。</p>
    <div class="chart-box" id="chart_sdoh"></div>

    <!-- ===== SECTION 7: 地理分析 ===== -->
    <h2>7. 地理分布分析</h2>
    <p class="section-intro">通过 Census Block Group FIPS Code 关联地理信息，识别高再入院率区域。</p>
    <div class="chart-box" id="chart_geo"></div>

    <!-- ===== SECTION 8: 清理建议 ===== -->
    <h2>8. 数据清理策略建议</h2>
    <table>
        <tr><th>缺失标记</th><th>含义</th><th>处理策略</th><th>理由</th></tr>
        <tr><td>NA / NaN</td><td>标准缺失</td><td>保留为 NaN，按需插补</td><td>真实缺失，可用中位数/众数填充</td></tr>
        <tr><td>*Unspecified</td><td>问了但未答</td><td>保留为独立类别</td><td>有信息价值（可能反映患者不愿披露）</td></tr>
        <tr><td>*Not Applicable</td><td>不适用</td><td>转为 NaN（结构性缺失）</td><td>如非住院就诊无入院信息，正常</td></tr>
        <tr><td>*Unknown</td><td>未记录</td><td>转为 NaN</td><td>真缺失</td></tr>
        <tr><td>*Deleted</td><td>已删除</td><td>排除该行或标记</td><td>数据质量问题，不应参与分析</td></tr>
        <tr><td>-1 (FK)</td><td>外键缺失</td><td>转为 NaN</td><td>关联缺失，表示无对应记录</td></tr>
    </table>

    <div class="warning-box">
        <strong>⚠️ 注意事项:</strong><br>
        1. encounters.csv 1.37GB 需 chunked 处理，切勿一次性加载全部列<br>
        2. social_determinants.Domain 约 80%+ 缺失，但可通过 DisplayName 映射回填<br>
        3. 两位年份日期(MM/DD/YY)需确认解析为 2022-2025 而非 1922-1925<br>
        4. providers 的 -1/-2/-3 特殊键代表系统保留值，分析时应排除
    </div>

    <!-- ===== Plotly 渲染 ===== -->
    <script>
        Plotly.newPlot('chart_missing', {fig_missing.to_json()}.data, {fig_missing.to_json()}.layout);
        Plotly.newPlot('chart_timeline', {fig_timeline.to_json()}.data, {fig_timeline.to_json()}.layout);
        Plotly.newPlot('chart_silhouette', {fig_silhouette.to_json()}.data, {fig_silhouette.to_json()}.layout);
        Plotly.newPlot('chart_cluster', {fig_cluster.to_json()}.data, {fig_cluster.to_json()}.layout);
        Plotly.newPlot('chart_radar', {fig_radar.to_json()}.data, {fig_radar.to_json()}.layout);
        Plotly.newPlot('chart_importance', {fig_importance.to_json()}.data, {fig_importance.to_json()}.layout);
        Plotly.newPlot('chart_sdoh', {fig_sdoh.to_json()}.data, {fig_sdoh.to_json()}.layout);
        Plotly.newPlot('chart_geo', {fig_geo.to_json()}.data, {fig_geo.to_json()}.layout);
    </script>
</body>
</html>"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html_report)

print(f"\n✅ 报告已保存: {REPORT_PATH}")

# 保存更新后的特征矩阵
patient_features.to_csv(os.path.join(OUTPUT_DIR, "patient_features.csv"), index=False)
patients_clean.to_csv(os.path.join(OUTPUT_DIR, "patients_cleaned.csv"), index=False)
sdoh_clean.to_csv(os.path.join(OUTPUT_DIR, "sdoh_cleaned.csv"), index=False)

print(f"✅ 清理后数据已保存至: {OUTPUT_DIR}/")

# === 最终摘要 ===
print("\n" + "=" * 70)
print("执行完成 - 摘要")
print("=" * 70)
print(f"  总患者: {len(patient_features):,}")
print(f"  总就诊: {enc_profile['total_rows']:,}")
print(f"  聚类数: {best_k} (silhouette={silhouette_scores[best_k]:.3f})")
print(f"  30天再入院率: {overall_readmit_rate*100:.2f}%")
print(f"  Top 风险因子: {importances.iloc[0]['feature']} ({importances.iloc[0]['importance']:.4f})")
print(f"\n  产出物:")
print(f"    - {REPORT_PATH}")
print(f"    - {OUTPUT_DIR}/patient_features.csv")
print(f"    - {OUTPUT_DIR}/patients_cleaned.csv")
print(f"    - {OUTPUT_DIR}/sdoh_cleaned.csv")
print("=" * 70)
