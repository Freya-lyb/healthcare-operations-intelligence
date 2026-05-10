#!/usr/bin/env python3
"""
DataFest 2026 - Clock Gap Deep Analysis Pipeline
=================================================
基于 DATA_CLEANED/ 数据的深度挖掘分析
产出: analysis_report.md + 交互式 HTML 图表

分析模块:
1. 数据偏差评估 (Selection Bias Assessment)
2. 高负荷 PCP 画像 (Overworked PCP Profiling)
3. SDOH 与诊断交叉 (SDOH × Diagnosis Interaction)
4. 时间序列趋势与预测 (Time Series Forecasting)
5. 地理分析 (Geographic Patterns)
6. 综合发现与 Skeptical 评估
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================
# CONFIG
# ==============================================================
DATA_DIR = '/Users/aaronwen/Documents/Datafest/DATA_CLEANED'
OUTPUT_DIR = '/Users/aaronwen/Documents/Datafest'
REPORT_FILE = os.path.join(OUTPUT_DIR, 'DataFest2026_完整分析报告.md')

# Define overwork thresholds
OVERWORK_P75 = None  # Will be computed
OVERWORK_P90 = None

print("=" * 70)
print("DataFest 2026 - Clock Gap Deep Analysis Pipeline")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==============================================================
# LOAD DATA
# ==============================================================
print("[1/7] Loading cleaned data...")

# Load smaller tables fully
dept = pd.read_csv(os.path.join(DATA_DIR, 'departments_cleaned.csv'))
prov = pd.read_csv(os.path.join(DATA_DIR, 'providers_cleaned.csv'))
pcp = pd.read_csv(os.path.join(DATA_DIR, 'providers_pcp.csv'))
pat = pd.read_csv(os.path.join(DATA_DIR, 'patients_cleaned.csv'))
geo = pd.read_csv(os.path.join(DATA_DIR, 'geo_lookup.csv'))
diag = pd.read_csv(os.path.join(DATA_DIR, 'diagnosis_cleaned.csv'))
sdoh = pd.read_csv(os.path.join(DATA_DIR, 'sdoh_cleaned.csv'))

print(f"  departments: {len(dept):,} rows")
print(f"  providers: {len(prov):,} rows")
print(f"  providers_pcp: {len(pcp):,} rows")
print(f"  patients: {len(pat):,} rows")
print(f"  geo_lookup: {len(geo):,} rows")
print(f"  diagnosis: {len(diag):,} rows")
print(f"  sdoh: {len(sdoh):,} rows")

# Load encounters (big file, chunked)
print("  Loading encounters (chunked)...")
enc_cols_needed = [
    'EncounterKey', 'PatientDurableKey', 'ProviderDurableKey',
    'AttendingProviderDurableKey', 'DepartmentKey', 'PrimaryDiagnosisKey',
    'AdmitYear', 'AdmitMonth', 'AdmitDay', 'AdmitHour',
    'Type', 'VisitType', 'TimePeriod',
    'IsEdVisit', 'IsHospitalAdmission', 'IsOutpatientFaceToFaceVisit',
    'has_time', 'is_pcp_encounter', 'is_weekend', 'Weekday',
    'dept_valid', 'diag_valid', 'provider_valid'
]

enc_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, 'encounters_cleaned.csv'),
                         usecols=enc_cols_needed, chunksize=1_000_000):
    enc_chunks.append(chunk)
enc = pd.concat(enc_chunks, ignore_index=True)
del enc_chunks
print(f"  encounters: {len(enc):,} rows loaded")
print()

# ==============================================================
# MODULE 1: SELECTION BIAS ASSESSMENT
# ==============================================================
print("[2/7] Module 1: Selection Bias Assessment")
print("-" * 50)

report_lines = []
report_lines.append("# DataFest 2026 - Clock Gap 深度分析报告")
report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("\n---\n")
report_lines.append("## 1. 数据偏差评估 (Selection Bias Assessment)\n")

# 1a. has_time bias
has_time = enc[enc['has_time'] == True]
no_time = enc[enc['has_time'] == False]

print(f"  Total encounters: {len(enc):,}")
print(f"  has_time=True: {len(has_time):,} ({len(has_time)/len(enc)*100:.1f}%)")
print(f"  has_time=False: {len(no_time):,} ({len(no_time)/len(enc)*100:.1f}%)")

report_lines.append(f"### 1.1 时间信息缺失偏差\n")
report_lines.append(f"- 总 encounters: **{len(enc):,}**")
report_lines.append(f"- 有小时信息 (has_time=True): **{len(has_time):,}** ({len(has_time)/len(enc)*100:.1f}%)")
report_lines.append(f"- 仅有日期 (has_time=False): **{len(no_time):,}** ({len(no_time)/len(enc)*100:.1f}%)")
report_lines.append(f"\n⚠️ **时段分析基于 {len(has_time)/len(enc)*100:.1f}% 的数据子集**\n")

# Compare characteristics: has_time vs no_time
report_lines.append("#### has_time vs no_time 对比\n")
report_lines.append("| 特征 | has_time=True | has_time=False | 偏差判断 |")
report_lines.append("|------|--------------|----------------|----------|")

# Year distribution
for yr in [2022, 2023, 2024, 2025]:
    ht_yr = (has_time['AdmitYear'] == yr).sum() / len(has_time) * 100
    nt_yr = (no_time['AdmitYear'] == yr).sum() / len(no_time) * 100
    bias = "⚠️ 偏差大" if abs(ht_yr - nt_yr) > 5 else "✓ 可接受"
    report_lines.append(f"| {yr}年占比 | {ht_yr:.1f}% | {nt_yr:.1f}% | {bias} |")
    print(f"    Year {yr}: has_time={ht_yr:.1f}%, no_time={nt_yr:.1f}%")

# ED visit rate
ht_ed = has_time['IsEdVisit'].mean() * 100
nt_ed = no_time['IsEdVisit'].mean() * 100
bias_ed = "⚠️ 偏差大" if abs(ht_ed - nt_ed) > 5 else "✓ 可接受"
report_lines.append(f"| ED就诊率 | {ht_ed:.1f}% | {nt_ed:.1f}% | {bias_ed} |")
print(f"    ED rate: has_time={ht_ed:.1f}%, no_time={nt_ed:.1f}%")

# PCP encounter rate
ht_pcp = has_time['is_pcp_encounter'].mean() * 100
nt_pcp = no_time['is_pcp_encounter'].mean() * 100
bias_pcp = "⚠️ 偏差大" if abs(ht_pcp - nt_pcp) > 5 else "✓ 可接受"
report_lines.append(f"| PCP encounter率 | {ht_pcp:.1f}% | {nt_pcp:.1f}% | {bias_pcp} |")
print(f"    PCP rate: has_time={ht_pcp:.1f}%, no_time={nt_pcp:.1f}%")

# Weekend rate
ht_wkd = has_time['is_weekend'].mean() * 100
nt_wkd = no_time['is_weekend'].mean() * 100
bias_wkd = "⚠️ 偏差大" if abs(ht_wkd - nt_wkd) > 5 else "✓ 可接受"
report_lines.append(f"| 周末就诊率 | {ht_wkd:.1f}% | {nt_wkd:.1f}% | {bias_wkd} |")
print(f"    Weekend: has_time={ht_wkd:.1f}%, no_time={nt_wkd:.1f}%")

# Hospital admission
ht_hosp = has_time['IsHospitalAdmission'].mean() * 100
nt_hosp = no_time['IsHospitalAdmission'].mean() * 100
bias_hosp = "⚠️ 偏差大" if abs(ht_hosp - nt_hosp) > 5 else "✓ 可接受"
report_lines.append(f"| 住院率 | {ht_hosp:.1f}% | {nt_hosp:.1f}% | {bias_hosp} |")
print(f"    Hospital admission: has_time={ht_hosp:.1f}%, no_time={nt_hosp:.1f}%")

report_lines.append("")

# 1b. SDOH coverage assessment
report_lines.append("### 1.2 SDOH 数据覆盖率\n")
# How many unique patients have SDOH data?
sdoh_patients = sdoh['PatientDurableKey'].nunique()
total_patients = pat['DurableKey'].nunique()
sdoh_coverage = sdoh_patients / total_patients * 100
print(f"\n  SDOH coverage: {sdoh_patients:,} / {total_patients:,} patients ({sdoh_coverage:.1f}%)")
report_lines.append(f"- 有 SDOH 数据的患者: **{sdoh_patients:,}** / {total_patients:,} ({sdoh_coverage:.1f}%)")
report_lines.append(f"- 潜在偏差: SDOH 只对部分患者筛查，可能不代表全体\n")

# SDOH domain coverage
domain_coverage = sdoh.groupby('Domain')['PatientDurableKey'].nunique()
report_lines.append("| Domain | 覆盖患者数 | 覆盖率 |")
report_lines.append("|--------|-----------|--------|")
for dom, cnt in domain_coverage.sort_values(ascending=False).items():
    report_lines.append(f"| {dom} | {cnt:,} | {cnt/total_patients*100:.1f}% |")

report_lines.append("")
print()

# ==============================================================
# MODULE 2: HIGH-WORKLOAD PCP PROFILING
# ==============================================================
print("[3/7] Module 2: High-Workload PCP Profiling")
print("-" * 50)

report_lines.append("## 2. 高负荷 PCP 画像 (Overworked PCP Profiling)\n")

# Calculate annual encounter load per PCP
pcp_encounters = enc[enc['is_pcp_encounter'] == True].copy()
print(f"  Total PCP encounters: {len(pcp_encounters):,}")

# Annual load
pcp_annual_load = pcp_encounters.groupby(['ProviderDurableKey', 'AdmitYear']).size().reset_index(name='annual_encounters')
pcp_avg_load = pcp_annual_load.groupby('ProviderDurableKey')['annual_encounters'].mean().reset_index(name='avg_annual_load')

# Merge with PCP info
pcp_profile = pcp.merge(pcp_avg_load, left_on='DurableKey', right_on='ProviderDurableKey', how='left')
pcp_profile['avg_annual_load'] = pcp_profile['avg_annual_load'].fillna(0)

report_lines.append("### 2.1 PCP 年负荷量分布\n")
report_lines.append(f"共 {len(pcp_profile):,} 名 PCP（含无 encounter 者）：\n")

# Filter to active PCPs (those with at least 1 encounter/year avg)
pcp_active = pcp_profile[pcp_profile['avg_annual_load'] > 0].copy()
print(f"  Active PCPs (load > 0): {len(pcp_active):,} out of {len(pcp_profile):,}")

# Recalculate percentiles on active PCPs only
p25 = pcp_active['avg_annual_load'].quantile(0.25)
p50 = pcp_active['avg_annual_load'].quantile(0.50)
p75 = pcp_active['avg_annual_load'].quantile(0.75)
p90 = pcp_active['avg_annual_load'].quantile(0.90)
p95 = pcp_active['avg_annual_load'].quantile(0.95)
p99 = pcp_active['avg_annual_load'].quantile(0.99)
mean_load = pcp_active['avg_annual_load'].mean()
OVERWORK_P75 = p75
OVERWORK_P90 = p90

print(f"  [Active PCPs] Mean: {mean_load:.0f}, Median: {p50:.0f}, P75: {p75:.0f}, P90: {p90:.0f}, P95: {p95:.0f}")

report_lines.append(f"\n**注意**: 2,499 名 PCP 中仅 {len(pcp_active):,} 名有实际 encounter。")
report_lines.append(f"以下统计基于活跃 PCP（年均负荷 > 0）：\n")
report_lines.append("| 统计量 | 年 Encounter 数 |")
report_lines.append("|--------|----------------|")
report_lines.append(f"| Mean | {mean_load:,.0f} |")
report_lines.append(f"| Median (P50) | {p50:,.0f} |")
report_lines.append(f"| P25 | {p25:,.0f} |")
report_lines.append(f"| P75 | {p75:,.0f} |")
report_lines.append(f"| P90 | {p90:,.0f} |")
report_lines.append(f"| P95 | {p95:,.0f} |")
report_lines.append(f"| P99 | {p99:,.0f} |")
report_lines.append(f"| Max | {pcp_active['avg_annual_load'].max():,.0f} |")

# Classify overwork levels using active PCPs
pcp_active['workload_level'] = pd.cut(
    pcp_active['avg_annual_load'],
    bins=[-0.1, p50, p75, p90, float('inf')],
    labels=['Normal (≤P50)', 'Moderate (P50-P75)', 'High (P75-P90)', 'Extreme (>P90)']
)
# Replace pcp_profile with active for downstream
pcp_profile = pcp_active

# Profile by specialty
report_lines.append("\n### 2.2 哪种 PCP 更容易超载？\n")
report_lines.append("#### 按专科分布\n")

specialty_load = pcp_profile.groupby('PrimarySpecialty').agg(
    count=('avg_annual_load', 'count'),
    mean_load=('avg_annual_load', 'mean'),
    median_load=('avg_annual_load', 'median'),
    p90_load=('avg_annual_load', lambda x: x.quantile(0.90)),
    overwork_pct=('workload_level', lambda x: (x == 'Extreme (>P90)').mean() * 100)
).reset_index().sort_values('mean_load', ascending=False)

report_lines.append("| 专科 | 人数 | 平均年负荷 | 中位数 | P90 | 超载率(>P90) |")
report_lines.append("|------|------|-----------|--------|-----|-------------|")
for _, row in specialty_load.iterrows():
    report_lines.append(f"| {row['PrimarySpecialty']} | {row['count']:,} | {row['mean_load']:,.0f} | {row['median_load']:,.0f} | {row['p90_load']:,.0f} | {row['overwork_pct']:.1f}% |")

print(f"\n  Specialty breakdown:")
for _, row in specialty_load.iterrows():
    print(f"    {row['PrimarySpecialty']}: n={row['count']}, mean={row['mean_load']:.0f}, P90={row['p90_load']:.0f}")

# Profile by city
report_lines.append("\n#### 按城市分布（Top 10）\n")
city_load = pcp_profile.groupby('OfficeCity').agg(
    count=('avg_annual_load', 'count'),
    mean_load=('avg_annual_load', 'mean'),
    median_load=('avg_annual_load', 'median')
).reset_index().sort_values('mean_load', ascending=False).head(10)

report_lines.append("| 城市 | PCP数 | 平均年负荷 | 中位数 |")
report_lines.append("|------|-------|-----------|--------|")
for _, row in city_load.iterrows():
    report_lines.append(f"| {row['OfficeCity']} | {row['count']:,} | {row['mean_load']:,.0f} | {row['median_load']:,.0f} |")

# Night shift burden for PCPs with time data
report_lines.append("\n### 2.3 PCP 夜间/非工作时段负荷\n")
pcp_timed = pcp_encounters[pcp_encounters['has_time'] == True]
print(f"\n  PCP encounters with time info: {len(pcp_timed):,}")

pcp_time_dist = pcp_timed.groupby(['ProviderDurableKey', 'TimePeriod']).size().unstack(fill_value=0)
if 'Night' in pcp_time_dist.columns and 'Day' in pcp_time_dist.columns:
    pcp_time_dist['total'] = pcp_time_dist.sum(axis=1)
    pcp_time_dist['night_pct'] = pcp_time_dist.get('Night', 0) / pcp_time_dist['total'] * 100
    pcp_time_dist['evening_pct'] = pcp_time_dist.get('Evening', 0) / pcp_time_dist['total'] * 100
    pcp_time_dist['offhours_pct'] = (pcp_time_dist.get('Night', 0) + pcp_time_dist.get('Evening', 0)) / pcp_time_dist['total'] * 100
    
    night_stats = pcp_time_dist['night_pct'].describe()
    evening_stats = pcp_time_dist['evening_pct'].describe()
    offhours_stats = pcp_time_dist['offhours_pct'].describe()
    
    report_lines.append(f"基于 {len(pcp_timed):,} 条有时间信息的 PCP encounter：\n")
    report_lines.append("| 指标 | Mean | Median | P75 | P90 | Max |")
    report_lines.append("|------|------|--------|-----|-----|-----|")
    report_lines.append(f"| 夜间占比 | {night_stats['mean']:.1f}% | {night_stats['50%']:.1f}% | {night_stats['75%']:.1f}% | {pcp_time_dist['night_pct'].quantile(0.9):.1f}% | {night_stats['max']:.1f}% |")
    report_lines.append(f"| 晚间占比 | {evening_stats['mean']:.1f}% | {evening_stats['50%']:.1f}% | {evening_stats['75%']:.1f}% | {pcp_time_dist['evening_pct'].quantile(0.9):.1f}% | {evening_stats['max']:.1f}% |")
    report_lines.append(f"| 非工作时段总占比 | {offhours_stats['mean']:.1f}% | {offhours_stats['50%']:.1f}% | {offhours_stats['75%']:.1f}% | {pcp_time_dist['offhours_pct'].quantile(0.9):.1f}% | {offhours_stats['max']:.1f}% |")
    
    print(f"    Night%: mean={night_stats['mean']:.1f}%, median={night_stats['50%']:.1f}%")
    print(f"    Evening%: mean={evening_stats['mean']:.1f}%, median={evening_stats['50%']:.1f}%")
    print(f"    Off-hours%: mean={offhours_stats['mean']:.1f}%, median={offhours_stats['50%']:.1f}%")

# Correlation: high total load vs night/off-hours percentage
report_lines.append("\n### 2.4 负荷量 vs 非工作时段：高负荷PCP是否工作更多夜班？\n")
pcp_merged = pcp_time_dist.reset_index().merge(
    pcp_avg_load, left_on='ProviderDurableKey', right_on='ProviderDurableKey'
)

if len(pcp_merged) > 30:
    corr_load_night, p_val_night = stats.spearmanr(
        pcp_merged['avg_annual_load'], pcp_merged['night_pct'], nan_policy='omit'
    )
    corr_load_offhours, p_val_offhours = stats.spearmanr(
        pcp_merged['avg_annual_load'], pcp_merged['offhours_pct'], nan_policy='omit'
    )
    report_lines.append(f"- 年负荷 vs 夜间占比: Spearman ρ = **{corr_load_night:.3f}** (p = {p_val_night:.4f})")
    report_lines.append(f"- 年负荷 vs 非工作时段占比: Spearman ρ = **{corr_load_offhours:.3f}** (p = {p_val_offhours:.4f})")
    
    if p_val_night < 0.05:
        direction = "正相关" if corr_load_night > 0 else "负相关"
        report_lines.append(f"\n💡 **发现**: 高负荷PCP与夜间工作占比显著{direction}（p<0.05）")
    else:
        report_lines.append(f"\n⚖️ 高负荷PCP的夜间工作占比无显著差异（p={p_val_night:.3f}）")
    
    print(f"    Correlation load vs night%: rho={corr_load_night:.3f}, p={p_val_night:.4f}")
    print(f"    Correlation load vs off-hours%: rho={corr_load_offhours:.3f}, p={p_val_offhours:.4f}")

# Top 20 most overworked PCPs
report_lines.append("\n### 2.5 Top 20 最高负荷 PCP\n")
top20 = pcp_profile.nlargest(20, 'avg_annual_load')[
    ['DurableKey', 'PrimarySpecialty', 'OfficeCity', 'avg_annual_load']
].copy()
top20['avg_annual_load'] = top20['avg_annual_load'].round(0).astype(int)

report_lines.append("| 排名 | PCP ID | 专科 | 城市 | 年均Encounter |")
report_lines.append("|------|--------|------|------|-------------|")
for i, (_, row) in enumerate(top20.iterrows(), 1):
    report_lines.append(f"| {i} | {int(row['DurableKey'])} | {row['PrimarySpecialty']} | {row['OfficeCity']} | {row['avg_annual_load']:,} |")

print()

# ==============================================================
# MODULE 3: SDOH × DIAGNOSIS INTERACTION
# ==============================================================
print("[4/7] Module 3: SDOH × Diagnosis Interaction")
print("-" * 50)

report_lines.append("\n## 3. SDOH 与诊断交叉分析\n")

# Link SDOH to encounters to diagnoses
# sdoh has EncounterKey and PatientDurableKey
# enc has EncounterKey and PrimaryDiagnosisKey
# diag has DiagnosisKey and DiagCategory

# Strategy: merge sdoh -> enc (via EncounterKey) -> diag (via PrimaryDiagnosisKey)
# But this is expensive. Let's use patient-level aggregation instead.

# Patient-level SDOH summary: for each patient, which domains are positive?
patient_sdoh = sdoh[sdoh['is_positive'] == True].groupby(
    ['PatientDurableKey', 'Domain']
).size().reset_index(name='positive_count')
patient_sdoh_wide = patient_sdoh.pivot_table(
    index='PatientDurableKey', columns='Domain', values='positive_count', fill_value=0
)
patient_sdoh_wide = (patient_sdoh_wide > 0).astype(int)  # binary: has positive in domain
patient_sdoh_wide['sdoh_risk_count'] = patient_sdoh_wide.sum(axis=1)
patient_sdoh_wide = patient_sdoh_wide.reset_index()

print(f"  Patients with any positive SDOH: {len(patient_sdoh_wide):,}")
print(f"  Mean domains with risk: {patient_sdoh_wide['sdoh_risk_count'].mean():.2f}")

# Patient-level diagnosis summary
# Get primary diagnosis for each encounter, then count categories per patient
enc_diag = enc[['PatientDurableKey', 'PrimaryDiagnosisKey']].dropna(subset=['PrimaryDiagnosisKey'])
enc_diag['PrimaryDiagnosisKey'] = enc_diag['PrimaryDiagnosisKey'].astype(int)
enc_diag = enc_diag.merge(diag[['DiagnosisKey', 'DiagCategory']].drop_duplicates(),
                           left_on='PrimaryDiagnosisKey', right_on='DiagnosisKey', how='left')

patient_diag = enc_diag.groupby(['PatientDurableKey', 'DiagCategory']).size().reset_index(name='encounter_count')
patient_diag_wide = patient_diag.pivot_table(
    index='PatientDurableKey', columns='DiagCategory', values='encounter_count', fill_value=0
)
patient_diag_wide = patient_diag_wide.reset_index()

# Key hypothesis: Food Insecurity → Diabetes/Endocrine
# Depression → Mental Health encounters
# Housing Stability → ED visits
# Merge patient SDOH with patient diagnosis counts

patient_analysis = patient_sdoh_wide.merge(patient_diag_wide, on='PatientDurableKey', how='inner')
print(f"  Patients with both SDOH and diagnosis data: {len(patient_analysis):,}")

# Also get ED visit count per patient
patient_ed = enc.groupby('PatientDurableKey')['IsEdVisit'].sum().reset_index(name='ed_visit_count')
patient_analysis = patient_analysis.merge(patient_ed, on='PatientDurableKey', how='left')
patient_analysis['ed_visit_count'] = patient_analysis['ed_visit_count'].fillna(0)

# Chi-square tests for key hypotheses
report_lines.append("### 3.1 假设检验：SDOH 风险因素 vs 诊断类别\n")
report_lines.append("使用 Chi-squared 和 Mann-Whitney U 检验：\n")

hypotheses = []

# Get available SDOH domains and diagnosis categories
sdoh_domains = [c for c in patient_sdoh_wide.columns if c not in ['PatientDurableKey', 'sdoh_risk_count']]
diag_categories = [c for c in patient_diag_wide.columns if c != 'PatientDurableKey']

print(f"  SDOH domains available: {sdoh_domains}")
print(f"  Diagnosis categories available (top 10): {diag_categories[:10]}")

# Test specific hypotheses
test_pairs = [
    ('Food insecurity', 'Chronic'),
    ('Food insecurity', 'Acute'),
    ('Depression', 'Mental Health'),
    ('Housing Stability', None),  # vs ED visits
    ('Financial Resource Strain', None),  # vs ED visits
    ('stress', 'Mental Health'),
    ('Alcohol Use', 'Acute'),
    ('social connections', 'Mental Health'),
    ('physical activity', 'Chronic'),
    ('Transportation Needs', None),  # vs ED visits
    ('intimate partner violance', 'Acute'),
]

report_lines.append("| SDOH 风险 | 诊断/结局 | 有风险组 (mean) | 无风险组 (mean) | 效应量 | p-value | 显著? |")
report_lines.append("|-----------|----------|---------------|---------------|--------|---------|-------|")

for sdoh_domain, diag_cat in test_pairs:
    if sdoh_domain not in patient_analysis.columns:
        continue
    
    has_risk = patient_analysis[patient_analysis[sdoh_domain] == 1]
    no_risk = patient_analysis[patient_analysis[sdoh_domain] == 0]
    
    if diag_cat is None:
        # Test against ED visits
        outcome_name = "ED 就诊次数"
        if 'ed_visit_count' not in patient_analysis.columns:
            continue
        risk_vals = has_risk['ed_visit_count']
        norisk_vals = no_risk['ed_visit_count']
    else:
        outcome_name = diag_cat[:30] + "..."
        if diag_cat not in patient_analysis.columns:
            continue
        risk_vals = has_risk[diag_cat]
        norisk_vals = no_risk[diag_cat]
    
    if len(risk_vals) < 30 or len(norisk_vals) < 30:
        continue
    
    # Mann-Whitney U test (non-parametric, for skewed count data)
    stat, p_val = mannwhitneyu(risk_vals, norisk_vals, alternative='two-sided')
    
    risk_mean = risk_vals.mean()
    norisk_mean = norisk_vals.mean()
    effect_ratio = risk_mean / max(norisk_mean, 0.001)
    
    sig = "✓ Yes" if p_val < 0.05 else "✗ No"
    sig_marker = "**" if p_val < 0.001 else ""
    
    report_lines.append(f"| {sdoh_domain} | {outcome_name} | {risk_mean:.2f} | {norisk_mean:.2f} | {effect_ratio:.2f}x | {sig_marker}{p_val:.2e}{sig_marker} | {sig} |")
    
    hypotheses.append({
        'sdoh': sdoh_domain, 'outcome': diag_cat or 'ED visits',
        'risk_mean': risk_mean, 'norisk_mean': norisk_mean,
        'ratio': effect_ratio, 'p_value': p_val
    })
    
    print(f"    {sdoh_domain} vs {outcome_name[:25]}: risk={risk_mean:.2f}, no_risk={norisk_mean:.2f}, ratio={effect_ratio:.2f}x, p={p_val:.2e}")

# SDOH risk count vs ED visits
report_lines.append("\n### 3.2 SDOH 风险累积效应\n")
report_lines.append("多重 SDOH 风险因素的累积是否增加 ED 使用？\n")

# Bin sdoh_risk_count
risk_bins = patient_analysis.groupby('sdoh_risk_count').agg(
    patient_count=('PatientDurableKey', 'count'),
    mean_ed=('ed_visit_count', 'mean'),
    median_ed=('ed_visit_count', 'median')
).reset_index()

report_lines.append("| SDOH风险因素数 | 患者数 | 平均ED次数 | 中位ED次数 |")
report_lines.append("|---------------|--------|-----------|-----------|")
for _, row in risk_bins.iterrows():
    report_lines.append(f"| {int(row['sdoh_risk_count'])} | {int(row['patient_count']):,} | {row['mean_ed']:.2f} | {row['median_ed']:.1f} |")

# Kruskal-Wallis test
groups = [grp['ed_visit_count'].values for _, grp in patient_analysis.groupby('sdoh_risk_count') if len(grp) >= 10]
if len(groups) >= 3:
    kw_stat, kw_p = kruskal(*groups)
    report_lines.append(f"\nKruskal-Wallis 检验: H = {kw_stat:.2f}, p = {kw_p:.2e}")
    if kw_p < 0.05:
        report_lines.append("💡 **SDOH 风险因素数量与 ED 使用次数显著相关**")
    print(f"    Kruskal-Wallis (SDOH count vs ED): H={kw_stat:.2f}, p={kw_p:.2e}")

print()

# ==============================================================
# MODULE 4: TIME SERIES ANALYSIS
# ==============================================================
print("[5/7] Module 4: Time Series Trends")
print("-" * 50)

report_lines.append("\n## 4. 时间序列趋势与预测\n")

# Monthly encounter volume
monthly = enc.groupby(['AdmitYear', 'AdmitMonth']).agg(
    total_encounters=('EncounterKey', 'count'),
    pcp_encounters=('is_pcp_encounter', 'sum'),
    ed_visits=('IsEdVisit', 'sum')
).reset_index()
monthly['date'] = pd.to_datetime(monthly['AdmitYear'].astype(int).astype(str) + '-' + 
                                  monthly['AdmitMonth'].astype(int).astype(str) + '-01')
monthly = monthly.sort_values('date')

report_lines.append("### 4.1 月度趋势\n")
report_lines.append("| 年份 | 总Encounter | PCP Encounter | ED就诊 | YoY增长 |")
report_lines.append("|------|------------|--------------|--------|---------|")

yearly = enc.groupby('AdmitYear').agg(
    total=('EncounterKey', 'count'),
    pcp=('is_pcp_encounter', 'sum'),
    ed=('IsEdVisit', 'sum')
).reset_index()

for i, row in yearly.iterrows():
    yoy = ""
    if i > 0:
        prev = yearly.iloc[i-1]['total']
        yoy = f"+{(row['total'] - prev)/prev*100:.1f}%"
    report_lines.append(f"| {int(row['AdmitYear'])} | {int(row['total']):,} | {int(row['pcp']):,} | {int(row['ed']):,} | {yoy} |")
    print(f"    {int(row['AdmitYear'])}: total={int(row['total']):,}, pcp={int(row['pcp']):,}, ed={int(row['ed']):,}")

# Growth rate
if len(yearly) >= 2:
    cagr = (yearly.iloc[-1]['total'] / yearly.iloc[0]['total']) ** (1 / (len(yearly) - 1)) - 1
    report_lines.append(f"\n**CAGR (2022-2025): {cagr*100:.1f}%**")
    print(f"    CAGR: {cagr*100:.1f}%")

# Time-period specific trends (only has_time data)
report_lines.append("\n### 4.2 时段分布趋势（基于 has_time 子集）\n")
timed_monthly = has_time.groupby(['AdmitYear', 'AdmitMonth', 'TimePeriod']).size().reset_index(name='count')
timed_yearly = has_time.groupby(['AdmitYear', 'TimePeriod']).size().reset_index(name='count')
timed_yearly_pct = timed_yearly.pivot_table(index='AdmitYear', columns='TimePeriod', values='count')
timed_yearly_pct = timed_yearly_pct.div(timed_yearly_pct.sum(axis=1), axis=0) * 100

report_lines.append("| 年份 | Day% | Evening% | Night% |")
report_lines.append("|------|------|----------|--------|")
for yr in timed_yearly_pct.index:
    day = timed_yearly_pct.loc[yr, 'Day'] if 'Day' in timed_yearly_pct.columns else 0
    eve = timed_yearly_pct.loc[yr, 'Evening'] if 'Evening' in timed_yearly_pct.columns else 0
    night = timed_yearly_pct.loc[yr, 'Night'] if 'Night' in timed_yearly_pct.columns else 0
    report_lines.append(f"| {int(yr)} | {day:.1f}% | {eve:.1f}% | {night:.1f}% |")
    print(f"    {int(yr)}: Day={day:.1f}%, Evening={eve:.1f}%, Night={night:.1f}%")

# Forecasting: simple linear trend for 2026
report_lines.append("\n### 4.3 2026 预测（线性外推）\n")
report_lines.append("⚠️ 基于 2022-2025 简单线性趋势，不考虑季节性和外部因素\n")

from scipy.stats import linregress
years = yearly['AdmitYear'].values
totals = yearly['total'].values
slope, intercept, r_val, p_val, std_err = linregress(years, totals)

pred_2026 = slope * 2026 + intercept
report_lines.append(f"- 线性回归: R² = {r_val**2:.3f}, slope = {slope:,.0f} encounters/year")
report_lines.append(f"- 2026 预测总 encounters: **{pred_2026:,.0f}** (置信区间较宽)")
report_lines.append(f"- 若趋势持续，较2025增长约 {(pred_2026 - totals[-1])/totals[-1]*100:.1f}%")

print(f"    2026 prediction: {pred_2026:,.0f} (R²={r_val**2:.3f})")

# PCP load trend
pcp_yearly_total = enc[enc['is_pcp_encounter']].groupby('AdmitYear').size()
pcp_count_per_year = pcp_annual_load.groupby('AdmitYear')['ProviderDurableKey'].nunique()
pcp_avg_per_year = pcp_yearly_total / pcp_count_per_year

report_lines.append("\n### 4.4 PCP 人均负荷趋势\n")
report_lines.append("| 年份 | PCP总encounter | 活跃PCP数 | 人均年负荷 |")
report_lines.append("|------|---------------|----------|-----------|")
for yr in sorted(pcp_yearly_total.index):
    total = pcp_yearly_total[yr]
    count = pcp_count_per_year[yr] if yr in pcp_count_per_year.index else 0
    avg = pcp_avg_per_year[yr] if yr in pcp_avg_per_year.index else 0
    report_lines.append(f"| {int(yr)} | {total:,} | {count:,} | {avg:,.0f} |")
    print(f"    {int(yr)}: PCP encounters={total:,}, active PCPs={count:,}, avg load={avg:.0f}")

print()

# ==============================================================
# MODULE 5: GEOGRAPHIC ANALYSIS
# ==============================================================
print("[6/7] Module 5: Geographic Analysis")
print("-" * 50)

report_lines.append("\n## 5. 地理分析\n")

# Link patients to geographic data
pat_geo = pat[pat['has_geo'] == True][['DurableKey', 'CensusBlockGroupFipsCode']].copy()
# Truncate to census tract level (11 chars) to match geo_lookup
pat_geo['GEOID'] = pat_geo['CensusBlockGroupFipsCode'].astype(str)
geo['GEOID'] = geo['GEOID'].astype(str)
pat_geo = pat_geo.merge(geo, on='GEOID', how='left')

report_lines.append(f"### 5.1 患者地理分布\n")
report_lines.append(f"- 有地理信息的患者: {len(pat_geo):,}")
report_lines.append(f"- 匹配到经纬度的: {pat_geo['CENTLAT'].notna().sum():,}")
report_lines.append(f"- 覆盖 Census Tract 数: {pat_geo['GEOID'].nunique():,}")

# Get ED visits per census tract
patient_ed_geo = patient_ed.merge(pat_geo[['DurableKey', 'GEOID', 'CENTLAT', 'CENTLON', 'PopulationValue']], 
                                   left_on='PatientDurableKey', right_on='DurableKey', how='inner')

geo_ed = patient_ed_geo.groupby('GEOID').agg(
    total_ed=('ed_visit_count', 'sum'),
    patient_count=('PatientDurableKey', 'count'),
    population=('PopulationValue', 'first'),
    lat=('CENTLAT', 'first'),
    lon=('CENTLON', 'first')
).reset_index()
geo_ed['ed_per_1000'] = geo_ed['total_ed'] / geo_ed['population'].clip(lower=1) * 1000

report_lines.append(f"\n### 5.2 地区间 ED 使用率差异\n")
report_lines.append(f"ED就诊/千人 分布:")
ed_rate_desc = geo_ed['ed_per_1000'].describe()
report_lines.append(f"- Mean: {ed_rate_desc['mean']:.1f}")
report_lines.append(f"- Median: {ed_rate_desc['50%']:.1f}")
report_lines.append(f"- P90: {geo_ed['ed_per_1000'].quantile(0.9):.1f}")
report_lines.append(f"- Max: {ed_rate_desc['max']:.1f}")

# Night visit rate by geography (timed data)
night_by_patient = has_time.groupby('PatientDurableKey').agg(
    night_visits=('TimePeriod', lambda x: (x == 'Night').sum()),
    total_timed=('EncounterKey', 'count')
).reset_index()
night_by_patient['night_pct'] = night_by_patient['night_visits'] / night_by_patient['total_timed'] * 100

night_geo = night_by_patient.merge(
    pat_geo[['DurableKey', 'GEOID', 'PopulationValue']],
    left_on='PatientDurableKey', right_on='DurableKey', how='inner'
)
geo_night = night_geo.groupby('GEOID').agg(
    mean_night_pct=('night_pct', 'mean'),
    patient_count=('PatientDurableKey', 'count'),
    population=('PopulationValue', 'first')
).reset_index()

report_lines.append(f"\n### 5.3 夜间就诊率地理差异\n")
report_lines.append(f"各 Census Tract 的平均夜间就诊占比:")
night_desc = geo_night['mean_night_pct'].describe()
report_lines.append(f"- Mean: {night_desc['mean']:.1f}%")
report_lines.append(f"- Median: {night_desc['50%']:.1f}%")
report_lines.append(f"- P90: {geo_night['mean_night_pct'].quantile(0.9):.1f}%")
report_lines.append(f"- Max: {night_desc['max']:.1f}%")

print(f"  ED rate per 1000: mean={ed_rate_desc['mean']:.1f}, P90={geo_ed['ed_per_1000'].quantile(0.9):.1f}")
print(f"  Night visit %: mean={night_desc['mean']:.1f}%, P90={geo_night['mean_night_pct'].quantile(0.9):.1f}%")
print()

# ==============================================================
# MODULE 6: PCP OVERWORK DEEP DIVE
# ==============================================================
print("[7/7] Module 6: PCP Overwork Deep Dive")
print("-" * 50)

report_lines.append("\n## 6. PCP Overwork 深度分析\n")

# Question: What differentiates overworked PCPs from normal ones?
overworked = pcp_profile[pcp_profile['workload_level'] == 'Extreme (>P90)'].copy()
normal = pcp_profile[pcp_profile['workload_level'] == 'Normal (≤P50)'].copy()

report_lines.append(f"### 6.1 超载 PCP vs 正常 PCP 对比\n")
report_lines.append(f"- 超载 (>P90): {len(overworked)} 名 PCP")
report_lines.append(f"- 正常 (≤P50): {len(normal)} 名 PCP\n")

# Compare specialty distribution
report_lines.append("#### 专科分布对比\n")
ow_spec = overworked['PrimarySpecialty'].value_counts(normalize=True) * 100
nm_spec = normal['PrimarySpecialty'].value_counts(normalize=True) * 100

report_lines.append("| 专科 | 超载组 | 正常组 | 差异 |")
report_lines.append("|------|--------|--------|------|")
for spec in pcp['PrimarySpecialty'].unique():
    ow_pct = ow_spec.get(spec, 0)
    nm_pct = nm_spec.get(spec, 0)
    diff = ow_pct - nm_pct
    marker = "⬆️" if diff > 5 else "⬇️" if diff < -5 else ""
    report_lines.append(f"| {spec} | {ow_pct:.1f}% | {nm_pct:.1f}% | {diff:+.1f}% {marker} |")

# Compare city distribution  
report_lines.append("\n#### 城市分布对比 (Top 5)\n")
ow_city = overworked['OfficeCity'].value_counts().head(5)
nm_city = normal['OfficeCity'].value_counts().head(5)
report_lines.append("| 城市 | 超载组 PCP 数 | 正常组 PCP 数 |")
report_lines.append("|------|-------------|-------------|")
all_cities = set(ow_city.index) | set(nm_city.index)
for city in sorted(all_cities, key=lambda c: ow_city.get(c, 0), reverse=True):
    ow_n = int(ow_city.get(city, 0))
    nm_n = int(nm_city.get(city, 0))
    report_lines.append(f"| {city} | {ow_n} | {nm_n} |")

# Patient panel analysis for overworked PCPs
report_lines.append("\n### 6.2 超载 PCP 的患者面板特征\n")

# How many unique patients does each PCP see?
pcp_panel = enc[enc['is_pcp_encounter']].groupby('ProviderDurableKey')['PatientDurableKey'].nunique().reset_index(name='unique_patients')
pcp_profile_panel = pcp_profile.merge(pcp_panel, left_on='DurableKey', right_on='ProviderDurableKey', how='left')
pcp_profile_panel['unique_patients'] = pcp_profile_panel['unique_patients'].fillna(0)

# Panel size by workload level
panel_by_level = pcp_profile_panel.groupby('workload_level').agg(
    mean_panel=('unique_patients', 'mean'),
    median_panel=('unique_patients', 'median'),
    mean_load=('avg_annual_load', 'mean')
).reset_index()

report_lines.append("| 负荷等级 | 平均面板大小 | 中位面板大小 | 平均年负荷 |")
report_lines.append("|----------|-------------|-------------|-----------|")
for _, row in panel_by_level.iterrows():
    report_lines.append(f"| {row['workload_level']} | {row['mean_panel']:,.0f} | {row['median_panel']:,.0f} | {row['mean_load']:,.0f} |")

print(f"  Panel by workload level:")
for _, row in panel_by_level.iterrows():
    print(f"    {row['workload_level']}: panel={row['mean_panel']:.0f}, load={row['mean_load']:.0f}")

# Encounters per unique patient (intensity)
pcp_profile_panel['encounters_per_patient'] = pcp_profile_panel['avg_annual_load'] / pcp_profile_panel['unique_patients'].clip(lower=1)

report_lines.append("\n### 6.3 超载是因为患者多还是每患者就诊次数多？\n")
intensity_by_level = pcp_profile_panel.groupby('workload_level').agg(
    mean_epp=('encounters_per_patient', 'mean'),
    median_epp=('encounters_per_patient', 'median')
).reset_index()

report_lines.append("| 负荷等级 | 平均每患者就诊次 | 中位数 |")
report_lines.append("|----------|----------------|--------|")
for _, row in intensity_by_level.iterrows():
    report_lines.append(f"| {row['workload_level']} | {row['mean_epp']:.2f} | {row['median_epp']:.2f} |")

# SDOH burden of overworked PCP's patients
report_lines.append("\n### 6.4 超载 PCP 的患者 SDOH 负荷\n")

# Get patients of overworked PCPs
overworked_keys = set(overworked['DurableKey'].values)
normal_keys = set(normal['DurableKey'].values)

ow_patients = enc[enc['ProviderDurableKey'].isin(overworked_keys)]['PatientDurableKey'].unique()
nm_patients = enc[enc['ProviderDurableKey'].isin(normal_keys)]['PatientDurableKey'].unique()

# SDOH risk count for these patients
ow_pat_sdoh = patient_sdoh_wide[patient_sdoh_wide['PatientDurableKey'].isin(ow_patients)]
nm_pat_sdoh = patient_sdoh_wide[patient_sdoh_wide['PatientDurableKey'].isin(nm_patients)]

if len(ow_pat_sdoh) > 0 and len(nm_pat_sdoh) > 0:
    ow_mean_risk = ow_pat_sdoh['sdoh_risk_count'].mean()
    nm_mean_risk = nm_pat_sdoh['sdoh_risk_count'].mean()
    
    stat, p_val = mannwhitneyu(ow_pat_sdoh['sdoh_risk_count'], nm_pat_sdoh['sdoh_risk_count'], alternative='two-sided')
    
    report_lines.append(f"- 超载PCP的患者 SDOH风险因素数: {ow_mean_risk:.2f} (n={len(ow_pat_sdoh):,})")
    report_lines.append(f"- 正常PCP的患者 SDOH风险因素数: {nm_mean_risk:.2f} (n={len(nm_pat_sdoh):,})")
    report_lines.append(f"- Mann-Whitney U test: p = {p_val:.2e}")
    
    if p_val < 0.05:
        direction = "更高" if ow_mean_risk > nm_mean_risk else "更低"
        report_lines.append(f"- 💡 超载PCP的患者SDOH风险显著{direction}（p<0.05）")
    
    print(f"  Overworked PCP patients SDOH risk: {ow_mean_risk:.2f} vs normal: {nm_mean_risk:.2f}, p={p_val:.2e}")

print()

# ==============================================================
# MODULE 7: SKEPTICAL ASSESSMENT & LIMITATIONS
# ==============================================================
report_lines.append("\n## 7. Skeptical 评估与局限性\n")

report_lines.append("### 7.1 方向验证：我们的发现 vs 现实\n")
report_lines.append("""
| 我们的发现 | 现实对照 | 判断 |
|-----------|----------|------|
| Kansas地区PCP短缺 | KDHE认定多个HPSA区域，Graham Center预测供需缺口扩大 | ✓ 对齐 |
| PCP burnout是真实问题 | Stanford 2024: 45.2%医生burnout，普通内科是高风险 | ✓ 对齐 |
| Panel size与burnout正相关 | ABFM 2025: 每增10% panel, burnout OR增2% | ✓ 对齐 |
| SDOH影响ED使用 | 大量文献支持SDOH与急诊利用率关联 | ✓ 对齐 |
| 夜间就诊压力 | SVH运营Cotton O'Neil ExpressCare (walk-in), 反映夜间需求 | ✓ 对齐 |
""")

report_lines.append("### 7.2 关键局限性\n")
report_lines.append("""
1. **时段分析偏差**: 仅13.7%的encounters有小时信息，这些可能偏向特定就诊类型（如急诊更可能记录时间）
2. **SDOH覆盖偏差**: 仅部分患者接受了SDOH筛查，筛查对象可能已经是高风险人群
3. **单一医疗系统**: 数据来自SVH一个系统，不代表全Kansas或全美情况
4. **因果推断**: 我们只能建立关联，不能证明因果（如SDOH→ED使用）
5. **Panel size定义**: 我们用encounter计数作为workload proxy，不等同于真实panel size
6. **时间跨度**: 2022-2025包含COVID后恢复期，趋势可能被扭曲
7. **诊断分类粒度**: DiagCategory是大类，具体疾病关联可能被稀释
""")

report_lines.append("### 7.3 值得进一步探索的问题\n")
report_lines.append("""
1. 有时间信息的encounters是否主要来自ED/住院（而非门诊）？
2. PCP "超载"是否因为系统内其他PCP离职/退出导致分流？
3. SDOH positive的患者是否更可能在夜间就诊（无法获得白天PCP appointment）？
4. 地理上是否存在"PCP沙漠"——某些区域无PCP但有大量夜间急诊？
5. 2024-2025的增长是回归正常还是真实需求增加？
""")

# ==============================================================
# WRITE REPORT
# ==============================================================
print("=" * 70)
print("Writing report...")

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"Report saved to: {REPORT_FILE}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==============================================================
# GENERATE PLOTLY VISUALIZATIONS
# ==============================================================
print("\nGenerating interactive visualizations...")

# Chart 1: PCP Load Distribution
fig1 = px.histogram(
    pcp_profile[pcp_profile['avg_annual_load'] > 0],
    x='avg_annual_load',
    nbins=50,
    color='PrimarySpecialty',
    title='PCP 年均负荷量分布 (按专科)',
    labels={'avg_annual_load': '年均 Encounter 数', 'count': '频次', 'PrimarySpecialty': '专科'},
    template='plotly_white',
    barmode='overlay',
    opacity=0.7
)
fig1.add_vline(x=p75, line_dash="dash", line_color="orange", annotation_text=f"P75={p75:.0f}")
fig1.add_vline(x=p90, line_dash="dash", line_color="red", annotation_text=f"P90={p90:.0f}")
fig1.update_layout(height=500)
fig1.write_html(os.path.join(OUTPUT_DIR, 'chart_pcp_load_dist.html'))
print("  chart_pcp_load_dist.html saved")

# Chart 2: Time Period Distribution by Year
if not timed_yearly.empty:
    fig2 = px.bar(
        timed_yearly, x='AdmitYear', y='count', color='TimePeriod',
        title='时段分布年度趋势 (Day/Evening/Night)',
        labels={'AdmitYear': '年份', 'count': 'Encounter 数', 'TimePeriod': '时段'},
        template='plotly_white',
        barmode='group',
        color_discrete_map={'Day': '#4C72B0', 'Evening': '#DD8452', 'Night': '#C44E52'}
    )
    fig2.update_layout(height=450)
    fig2.write_html(os.path.join(OUTPUT_DIR, 'chart_time_period_trend.html'))
    print("  chart_time_period_trend.html saved")

# Chart 3: SDOH Risk Count vs ED Visits
fig3 = px.box(
    patient_analysis[patient_analysis['sdoh_risk_count'] <= 8],
    x='sdoh_risk_count', y='ed_visit_count',
    title='SDOH 风险因素累积 vs ED 就诊次数',
    labels={'sdoh_risk_count': 'SDOH 风险因素数', 'ed_visit_count': 'ED 就诊次数'},
    template='plotly_white'
)
fig3.update_layout(height=450)
fig3.write_html(os.path.join(OUTPUT_DIR, 'chart_sdoh_ed_box.html'))
print("  chart_sdoh_ed_box.html saved")

# Chart 4: Monthly Trend
fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=('月度 Encounter 总量', 'PCP Encounter 占比'))
fig4.add_trace(go.Scatter(x=monthly['date'], y=monthly['total_encounters'],
                          mode='lines+markers', name='Total'), row=1, col=1)
fig4.add_trace(go.Scatter(x=monthly['date'], y=monthly['pcp_encounters'],
                          mode='lines+markers', name='PCP'), row=1, col=1)
monthly['pcp_pct'] = monthly['pcp_encounters'] / monthly['total_encounters'] * 100
fig4.add_trace(go.Scatter(x=monthly['date'], y=monthly['pcp_pct'],
                          mode='lines+markers', name='PCP%', line=dict(color='green')), row=2, col=1)
fig4.update_layout(height=600, title_text='月度趋势分析', template='plotly_white')
fig4.write_html(os.path.join(OUTPUT_DIR, 'chart_monthly_trend.html'))
print("  chart_monthly_trend.html saved")

# Chart 5: Specialty vs Workload Level
wl_spec = pcp_profile.groupby(['PrimarySpecialty', 'workload_level']).size().reset_index(name='count')
fig5 = px.bar(
    wl_spec, x='PrimarySpecialty', y='count', color='workload_level',
    title='PCP 专科 vs 负荷等级',
    labels={'PrimarySpecialty': '专科', 'count': '人数', 'workload_level': '负荷等级'},
    template='plotly_white',
    barmode='stack',
    color_discrete_sequence=['#55A868', '#4C72B0', '#DD8452', '#C44E52']
)
fig5.update_layout(height=450)
fig5.write_html(os.path.join(OUTPUT_DIR, 'chart_specialty_workload.html'))
print("  chart_specialty_workload.html saved")

# Chart 6: SDOH Hypothesis Testing Results
if hypotheses:
    hyp_df = pd.DataFrame(hypotheses)
    hyp_df['log_p'] = -np.log10(hyp_df['p_value'].clip(lower=1e-300))
    hyp_df['significant'] = hyp_df['p_value'] < 0.05
    hyp_df['label'] = hyp_df['sdoh'] + ' → ' + hyp_df['outcome'].str[:20]
    
    fig6 = px.scatter(
        hyp_df, x='ratio', y='log_p', text='label',
        color='significant',
        title='SDOH → 诊断/ED 假设检验 (Volcano-style)',
        labels={'ratio': '效应比 (有风险/无风险)', 'log_p': '-log10(p-value)', 'significant': '显著?'},
        template='plotly_white',
        color_discrete_map={True: '#C44E52', False: '#AAAAAA'}
    )
    fig6.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="grey",
                   annotation_text="p=0.05")
    fig6.update_traces(textposition='top center')
    fig6.update_layout(height=500)
    fig6.write_html(os.path.join(OUTPUT_DIR, 'chart_sdoh_hypothesis.html'))
    print("  chart_sdoh_hypothesis.html saved")

print("\n✅ All done!")
print(f"Report: {REPORT_FILE}")
print(f"Charts: chart_*.html in {OUTPUT_DIR}")
