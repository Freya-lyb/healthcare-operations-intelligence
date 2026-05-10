"""
DataFest 2026 - Step 1: 数据清理
=================================
将原始 DATA/ 中的 7 个 CSV 进行系统性清理，输出到 output/ 文件夹。

清理策略:
  NA / NaN         → 保持 NaN
  *Unspecified     → 保留（有信息价值: 问了但未答）
  *Not Applicable  → NaN（结构性缺失）
  *Unknown         → NaN（真缺失）
  *Deleted         → 排除整行
  -1 (外键)        → NaN

输出: output/ 目录下的清理后 CSV 文件
"""

import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ============== 配置 ==============
DATA_DIR = "/Users/aaronwen/Documents/Datafest/DATA"
OUTPUT_DIR = "/Users/aaronwen/Documents/Datafest/output"
SDOH_MAPPING_PATH = "/Users/aaronwen/Documents/Datafest/Social Determinant Questions and Domains.csv"
CHUNK_SIZE = 500000

SPECIAL_FK_VALUES = [-1, -2, -3]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DataFest 2026 - Step 1: 数据清理")
print(f"输入: {DATA_DIR}/")
print(f"输出: {OUTPUT_DIR}/")
print("=" * 70)

# =====================================================================
# 通用清理函数
# =====================================================================
def clean_markers(df, name="", exclude_deleted_rows=True):
    """
    统一处理缺失标记:
    - *Unknown → NaN
    - *Deleted → NaN (或排除整行)
    - *Not Applicable → NaN
    - *Unspecified → 保留（有信息价值）
    """
    df = df.copy()
    stats = {'rows_before': len(df)}
    
    # 识别含 *Deleted 的行（任意列含 *Deleted 即排除）
    if exclude_deleted_rows:
        deleted_mask = df.apply(lambda row: row.astype(str).str.contains(r'^\*Deleted$', regex=True).any(), axis=1)
        deleted_count = deleted_mask.sum()
        if deleted_count > 0:
            df = df[~deleted_mask].copy()
            print(f"    排除 *Deleted 行: {deleted_count:,}")
    
    # 逐列处理
    for col in df.columns:
        # *Unknown → NaN
        mask = df[col].astype(str) == '*Unknown'
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
        
        # *Not Applicable → NaN
        mask = df[col].astype(str) == '*Not Applicable'
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
        
        # *Unspecified → 保留（不转换）
    
    stats['rows_after'] = len(df)
    stats['rows_removed'] = stats['rows_before'] - stats['rows_after']
    return df, stats


# =====================================================================
# 1. 清理 patients.csv
# =====================================================================
print("\n[1/7] 清理 patients.csv...")
patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
print(f"  原始: {len(patients):,} 行, {len(patients.columns)} 列")
print(f"  列: {list(patients.columns)}")

patients_clean, p_stats = clean_markers(patients, "patients")

# 处理 PatientBirthYearBin 异常值: <1900 视为异常
if 'PatientBirthYearBin' in patients_clean.columns:
    patients_clean['PatientBirthYearBin'] = pd.to_numeric(patients_clean['PatientBirthYearBin'], errors='coerce')
    outlier_mask = patients_clean['PatientBirthYearBin'] < 1900
    patients_clean.loc[outlier_mask, 'PatientBirthYearBin'] = np.nan
    print(f"  BirthYear 异常值置空: {outlier_mask.sum():,}")

# 重命名 DurableKey → PatientDurableKey (统一外键名)
patients_clean = patients_clean.rename(columns={'DurableKey': 'PatientDurableKey'})

patients_clean.to_csv(os.path.join(OUTPUT_DIR, "patients_cleaned.csv"), index=False)
print(f"  输出: {len(patients_clean):,} 行 → output/patients_cleaned.csv")

# =====================================================================
# 2. 清理 providers.csv
# =====================================================================
print("\n[2/7] 清理 providers.csv...")
providers = pd.read_csv(os.path.join(DATA_DIR, "providers.csv"))
print(f"  原始: {len(providers):,} 行, {len(providers.columns)} 列")

providers_clean, prov_stats = clean_markers(providers, "providers")

# 排除特殊键行 (-1, -2, -3)
providers_clean = providers_clean[~providers_clean['DurableKey'].isin(SPECIAL_FK_VALUES)]
print(f"  排除特殊键(-1/-2/-3)后: {len(providers_clean):,} 行")

# 重命名 DurableKey → ProviderDurableKey
providers_clean = providers_clean.rename(columns={'DurableKey': 'ProviderDurableKey'})

providers_clean.to_csv(os.path.join(OUTPUT_DIR, "providers_cleaned.csv"), index=False)
print(f"  输出: {len(providers_clean):,} 行 → output/providers_cleaned.csv")

# =====================================================================
# 3. 清理 departments.csv
# =====================================================================
print("\n[3/7] 清理 departments.csv...")
departments = pd.read_csv(os.path.join(DATA_DIR, "departments.csv"))
print(f"  原始: {len(departments):,} 行, {len(departments.columns)} 列")

departments_clean, dept_stats = clean_markers(departments, "departments")

# 排除特殊键
departments_clean = departments_clean[~departments_clean['DepartmentKey'].isin(SPECIAL_FK_VALUES)]
print(f"  排除特殊键后: {len(departments_clean):,} 行")

departments_clean.to_csv(os.path.join(OUTPUT_DIR, "departments_cleaned.csv"), index=False)
print(f"  输出: {len(departments_clean):,} 行 → output/departments_cleaned.csv")

# =====================================================================
# 4. 清理 diagnosis.csv
# =====================================================================
print("\n[4/7] 清理 diagnosis.csv...")
diagnosis = pd.read_csv(os.path.join(DATA_DIR, "diagnosis.csv"))
print(f"  原始: {len(diagnosis):,} 行, {len(diagnosis.columns)} 列")

diagnosis_clean, diag_stats = clean_markers(diagnosis, "diagnosis")

diagnosis_clean.to_csv(os.path.join(OUTPUT_DIR, "diagnosis_cleaned.csv"), index=False)
print(f"  输出: {len(diagnosis_clean):,} 行 → output/diagnosis_cleaned.csv")

# =====================================================================
# 5. 清理 social_determinants.csv
# =====================================================================
print("\n[5/7] 清理 social_determinants.csv...")
sdoh = pd.read_csv(os.path.join(DATA_DIR, "social_determinants.csv"))
print(f"  原始: {len(sdoh):,} 行, {len(sdoh.columns)} 列")

sdoh_clean, sdoh_stats = clean_markers(sdoh, "social_determinants", exclude_deleted_rows=False)

# 回填 Domain（通过 DisplayName → Domain 映射表）
sdoh_mapping = pd.read_csv(SDOH_MAPPING_PATH)
display_to_domain = dict(zip(sdoh_mapping['DisplayName'], sdoh_mapping['Domain']))

domain_null_before = sdoh_clean['Domain'].isna().sum()
# DisplayName 为 *Unspecified 时无法映射，这部分 Domain 仍为空
sdoh_clean['Domain'] = sdoh_clean['Domain'].fillna(
    sdoh_clean['DisplayName'].map(display_to_domain)
)
domain_null_after = sdoh_clean['Domain'].isna().sum()
print(f"  Domain 回填: {domain_null_before:,} NaN → {domain_null_after:,} NaN (填补 {domain_null_before - domain_null_after:,})")

# 移除 DisplayName 为 *Unspecified 且 Domain 仍为空的行（无信息价值）
no_info_mask = (sdoh_clean['DisplayName'] == '*Unspecified') & sdoh_clean['Domain'].isna()
sdoh_clean = sdoh_clean[~no_info_mask]
print(f"  移除无信息行(DisplayName=*Unspecified & 无Domain): {no_info_mask.sum():,}")

sdoh_clean.to_csv(os.path.join(OUTPUT_DIR, "social_determinants_cleaned.csv"), index=False)
print(f"  输出: {len(sdoh_clean):,} 行 → output/social_determinants_cleaned.csv")

# =====================================================================
# 6. 清理 encounters.csv (chunked)
# =====================================================================
print("\n[6/7] 清理 encounters.csv (chunked, 1.37GB)...")

enc_output_path = os.path.join(OUTPUT_DIR, "encounters_cleaned.csv")
first_chunk_written = False
total_in = 0
total_out = 0
date_issues = {'future': 0, 'discharge_before_admit': 0}

for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"), chunksize=CHUNK_SIZE):
    total_in += len(chunk)
    
    # 处理缺失标记
    for col in chunk.columns:
        mask_unknown = chunk[col].astype(str) == '*Unknown'
        chunk.loc[mask_unknown, col] = np.nan
        
        mask_na = chunk[col].astype(str) == '*Not Applicable'
        chunk.loc[mask_na, col] = np.nan
        
        mask_del = chunk[col].astype(str) == '*Deleted'
        chunk.loc[mask_del, col] = np.nan
    
    # 处理外键 -1
    for fk_col in ['ProviderDurableKey', 'AttendingProviderDurableKey', 'DischargeProviderDurableKey']:
        if fk_col in chunk.columns:
            fk_mask = chunk[fk_col].isin(SPECIAL_FK_VALUES)
            chunk.loc[fk_mask, fk_col] = np.nan
    
    # 日期解析和验证
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk['AdmissionInstant'] = pd.to_datetime(chunk['AdmissionInstant'], errors='coerce')
    chunk['DischargeInstant'] = pd.to_datetime(chunk['DischargeInstant'], errors='coerce')
    
    # 标记异常日期（不删除，只记录）
    future_mask = chunk['Date'] > pd.Timestamp('2025-12-31')
    date_issues['future'] += future_mask.sum()
    
    valid_both = chunk['DischargeInstant'].notna() & chunk['AdmissionInstant'].notna()
    dis_before_adm = valid_both & (chunk['DischargeInstant'] < chunk['AdmissionInstant'])
    date_issues['discharge_before_admit'] += dis_before_adm.sum()
    
    total_out += len(chunk)
    
    # 写出
    chunk.to_csv(enc_output_path, mode='a' if first_chunk_written else 'w', 
                 header=not first_chunk_written, index=False)
    first_chunk_written = True
    print(f"    处理: {total_in:,} 行...", end='\r')

print(f"\n  输入: {total_in:,} 行, 输出: {total_out:,} 行")
print(f"  日期异常: 未来日期={date_issues['future']}, 出院<入院={date_issues['discharge_before_admit']}")
print(f"  输出 → output/encounters_cleaned.csv")

# =====================================================================
# 7. 复制 tigercensuscodes.csv (无需清理)
# =====================================================================
print("\n[7/7] tigercensuscodes.csv (无需清理，直接复制)...")
tiger = pd.read_csv(os.path.join(DATA_DIR, "tigercensuscodes.csv"))
tiger.to_csv(os.path.join(OUTPUT_DIR, "tigercensuscodes.csv"), index=False)
print(f"  输出: {len(tiger):,} 行 → output/tigercensuscodes.csv")

# =====================================================================
# 生成清理报告摘要
# =====================================================================
print("\n" + "=" * 70)
print("数据清理完成 - 摘要")
print("=" * 70)

cleaning_summary = {
    'patients': {'input': len(patients), 'output': len(patients_clean)},
    'providers': {'input': len(providers), 'output': len(providers_clean)},
    'departments': {'input': len(departments), 'output': len(departments_clean)},
    'diagnosis': {'input': len(diagnosis), 'output': len(diagnosis_clean)},
    'social_determinants': {'input': len(sdoh), 'output': len(sdoh_clean)},
    'encounters': {'input': total_in, 'output': total_out},
    'tigercensuscodes': {'input': len(tiger), 'output': len(tiger)},
}

print(f"\n{'文件':<30} {'输入行数':>12} {'输出行数':>12} {'变化':>10}")
print("-" * 70)
for name, info in cleaning_summary.items():
    delta = info['output'] - info['input']
    print(f"  {name:<28} {info['input']:>12,} {info['output']:>12,} {delta:>+10,}")

# 保存清理元数据
metadata = {
    'cleaning_strategy': {
        '*Unknown': 'converted to NaN (true missing)',
        '*Not Applicable': 'converted to NaN (structural missing)',
        '*Deleted': 'rows excluded (data quality issue)',
        '*Unspecified': 'kept as-is (informative: asked but not answered)',
        '-1/-2/-3 FK': 'converted to NaN (missing foreign key)',
    },
    'files_produced': [
        'patients_cleaned.csv',
        'providers_cleaned.csv', 
        'departments_cleaned.csv',
        'diagnosis_cleaned.csv',
        'social_determinants_cleaned.csv',
        'encounters_cleaned.csv',
        'tigercensuscodes.csv',
    ],
    'summary': cleaning_summary,
    'date_anomalies': date_issues,
    'key_changes': [
        'patients: DurableKey renamed to PatientDurableKey',
        'providers: DurableKey renamed to ProviderDurableKey, special keys removed',
        'departments: special keys removed',
        'social_determinants: Domain backfilled via DisplayName mapping, no-info rows removed',
        'encounters: FK -1 values set to NaN, dates parsed as datetime',
    ]
}

with open(os.path.join(OUTPUT_DIR, "cleaning_metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✅ 所有文件已输出到: {OUTPUT_DIR}/")
print(f"✅ 元数据: {OUTPUT_DIR}/cleaning_metadata.json")
print("=" * 70)
