"""
DataFest 2026 - Step 2: 特征工程 + 聚类 + 风险建模 + 报告
==========================================================
基于 output/ 文件夹中清理后的数据执行:
1. 患者特征矩阵构建（特征工程）
2. 聚类分析（KMeans + Silhouette + PCA）
3. 30天再入院风险建模（Random Forest）
4. SDOH 风险暴露分析
5. 时序与地理分析
6. 生成交互式 HTML 报告

输入: output/ (清理后CSV)
输出: output/analysis_report.html, output/patient_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============== 配置 ==============
OUTPUT_DIR = "/Users/aaronwen/Documents/Datafest/output"
REPORT_PATH = os.path.join(OUTPUT_DIR, "analysis_report.html")
CHUNK_SIZE = 500000

print("=" * 70)
print("DataFest 2026 - Step 2: 特征工程 + 聚类 + 风险建模")
print(f"数据来源: {OUTPUT_DIR}/")
print("=" * 70)

# =====================================================================
# PHASE 1: 加载清理后数据
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 1] 加载清理后数据")
print("=" * 70)

print("  加载 patients_cleaned.csv...")
patients = pd.read_csv(os.path.join(OUTPUT_DIR, "patients_cleaned.csv"))
print(f"    {len(patients):,} 行")

print("  加载 diagnosis_cleaned.csv...")
diagnosis = pd.read_csv(os.path.join(OUTPUT_DIR, "diagnosis_cleaned.csv"))
print(f"    {len(diagnosis):,} 行")

print("  加载 social_determinants_cleaned.csv...")
sdoh = pd.read_csv(os.path.join(OUTPUT_DIR, "social_determinants_cleaned.csv"))
print(f"    {len(sdoh):,} 行")

print("  加载 departments_cleaned.csv...")
departments = pd.read_csv(os.path.join(OUTPUT_DIR, "departments_cleaned.csv"))
print(f"    {len(departments):,} 行")

print("  加载 tigercensuscodes.csv...")
tiger = pd.read_csv(os.path.join(OUTPUT_DIR, "tigercensuscodes.csv"))
print(f"    {len(tiger):,} 行")

# =====================================================================
# PHASE 2: 特征工程 — 构建患者特征矩阵
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 2] 特征工程 — 构建患者特征矩阵")
print("=" * 70)

# 2a: 从 encounters (chunked) 聚合患者级特征
print("  2a: 从 encounters 聚合患者级就诊特征...")

patient_agg = {}
enc_path = os.path.join(OUTPUT_DIR, "encounters_cleaned.csv")

enc_cols = ['PatientDurableKey', 'Date', 'IsEdVisit', 'IsHospitalAdmission',
            'IsInpatientAdmission', 'PrimaryDiagnosisKey', 'DepartmentKey',
            'DischargeInstant', 'VisitType', 'Type']

for chunk in pd.read_csv(enc_path, usecols=enc_cols, chunksize=CHUNK_SIZE):
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
                'unique_visit_types': set(),
                'first_date': None,
                'last_date': None,
            }
        
        pa = patient_agg[pid]
        pa['total_encounters'] += len(group)
        pa['ed_visits'] += (group['IsEdVisit'] == True).sum()
        pa['hospital_admissions'] += (group['IsHospitalAdmission'] == True).sum()
        pa['inpatient_admissions'] += (group['IsInpatientAdmission'] == True).sum()
        
        pa['unique_diagnoses'].update(group['PrimaryDiagnosisKey'].dropna().unique())
        pa['unique_departments'].update(group['DepartmentKey'].dropna().unique())
        pa['unique_visit_types'].update(group['VisitType'].dropna().unique())
        
        dates = group['Date'].dropna()
        if len(dates) > 0:
            if pa['first_date'] is None or dates.min() < pa['first_date']:
                pa['first_date'] = dates.min()
            if pa['last_date'] is None or dates.max() > pa['last_date']:
                pa['last_date'] = dates.max()
    
    print(f"    患者数: {len(patient_agg):,}", end='\r')

print(f"\n    总患者数: {len(patient_agg):,}")

# 转为 DataFrame
print("  构建特征 DataFrame...")
features_list = []
for pid, f in patient_agg.items():
    span_days = (f['last_date'] - f['first_date']).days if f['first_date'] and f['last_date'] else 0
    features_list.append({
        'PatientDurableKey': pid,
        'total_encounters': f['total_encounters'],
        'ed_visits': f['ed_visits'],
        'hospital_admissions': f['hospital_admissions'],
        'inpatient_admissions': f['inpatient_admissions'],
        'unique_diagnoses': len(f['unique_diagnoses']),
        'unique_departments': len(f['unique_departments']),
        'unique_visit_types': len(f['unique_visit_types']),
        'span_days': span_days,
        'encounters_per_month': f['total_encounters'] / max(span_days / 30.44, 1),
        'ed_ratio': f['ed_visits'] / max(f['total_encounters'], 1),
        'inpatient_ratio': f['inpatient_admissions'] / max(f['total_encounters'], 1),
    })

patient_features = pd.DataFrame(features_list)

# 2b: 合并人口统计学
print("  2b: 合并人口统计学特征...")
patients_demo = patients[['PatientDurableKey', 'PatientBirthYearBin', 'SexAssignedAtBirth',
                           'FirstRace', 'OmbEthnicity', 'SmokingStatus', 'MaritalStatus']].copy()

# 编码关键人口特征
patients_demo['approx_age'] = 2025 - patients_demo['PatientBirthYearBin']
patients_demo['is_female'] = (patients_demo['SexAssignedAtBirth'] == 'Female').astype(int)
patients_demo['is_smoker'] = patients_demo['SmokingStatus'].isin(
    ['Current Every Day Smoker', 'Current Some Day Smoker', 'Heavy Tobacco Smoker', 'Light Tobacco Smoker']
).astype(int)
patients_demo['is_married'] = (patients_demo['MaritalStatus'] == 'Married').astype(int)

patient_features = patient_features.merge(
    patients_demo[['PatientDurableKey', 'approx_age', 'is_female', 'is_smoker', 'is_married']],
    on='PatientDurableKey', how='left'
)

# 2c: SDOH 特征
print("  2c: 计算 SDOH 暴露特征...")
sdoh_by_patient = sdoh.groupby('PatientDurableKey').agg(
    sdoh_total_screenings=('EncounterKey', 'count'),
    sdoh_unique_domains=('Domain', 'nunique'),
).reset_index()

# 每个 domain 的 0/1 标记
all_domains = sdoh['Domain'].dropna().unique()
print(f"    SDOH Domains 发现: {len(all_domains)} 个")
for domain in all_domains:
    domain_patients = set(sdoh[sdoh['Domain'] == domain]['PatientDurableKey'].unique())
    col_name = f"sdoh_{domain.replace(' ', '_').lower()}"
    sdoh_by_patient[col_name] = sdoh_by_patient['PatientDurableKey'].isin(domain_patients).astype(int)

patient_features = patient_features.merge(sdoh_by_patient, on='PatientDurableKey', how='left')

# SDOH 缺失填 0
sdoh_cols = [c for c in patient_features.columns if c.startswith('sdoh_')]
patient_features[sdoh_cols] = patient_features[sdoh_cols].fillna(0)

# 2d: 诊断多样性特征（ICD分类）
print("  2d: 诊断分类特征...")
# 诊断大类统计
def get_icd_chapter(code):
    """简单的ICD-10大类映射"""
    if pd.isna(code): return None
    code = str(code).upper()
    if code.startswith('E'): return 'Endocrine'
    elif code.startswith('F'): return 'Mental'
    elif code.startswith('I'): return 'Circulatory'
    elif code.startswith('J'): return 'Respiratory'
    elif code.startswith('K'): return 'Digestive'
    elif code.startswith('M'): return 'Musculoskeletal'
    elif code.startswith('N'): return 'Genitourinary'
    elif code.startswith('Z'): return 'HealthServices'
    elif code.startswith('R'): return 'Symptoms'
    else: return 'Other'

diagnosis['ICD_Chapter'] = diagnosis['DiagnosisValue'].apply(get_icd_chapter)

# 构建 DiagnosisKey → Chapter 映射
diag_key_to_chapter = dict(zip(diagnosis['DiagnosisKey'], diagnosis['ICD_Chapter']))

# 统计每个患者涉及的 ICD 大类数
print("    统计患者诊断大类分布（chunked）...")
patient_chapters = {}
for chunk in pd.read_csv(enc_path, usecols=['PatientDurableKey', 'PrimaryDiagnosisKey'], chunksize=CHUNK_SIZE):
    chunk['chapter'] = chunk['PrimaryDiagnosisKey'].map(diag_key_to_chapter)
    for pid, group in chunk.groupby('PatientDurableKey'):
        chapters = set(group['chapter'].dropna().unique())
        if pid not in patient_chapters:
            patient_chapters[pid] = set()
        patient_chapters[pid].update(chapters)

# 为关键大类创建标记
key_chapters = ['Mental', 'Endocrine', 'Circulatory', 'Respiratory']
chapter_features = []
for pid in patient_features['PatientDurableKey']:
    chs = patient_chapters.get(pid, set())
    row = {'PatientDurableKey': pid, 'diag_chapter_count': len(chs)}
    for ch in key_chapters:
        row[f'has_{ch.lower()}_diag'] = int(ch in chs)
    chapter_features.append(row)

chapter_df = pd.DataFrame(chapter_features)
patient_features = patient_features.merge(chapter_df, on='PatientDurableKey', how='left')

# 2e: 30天再入院标签
print("  2e: 计算 30天再入院标签...")
inpatient_chunks = []
for chunk in pd.read_csv(enc_path, usecols=['PatientDurableKey', 'Date', 'DischargeInstant', 'IsInpatientAdmission'],
                         chunksize=CHUNK_SIZE):
    mask = chunk['IsInpatientAdmission'] == True
    filtered = chunk[mask][['PatientDurableKey', 'Date', 'DischargeInstant']].copy()
    if len(filtered) > 0:
        inpatient_chunks.append(filtered)

inpatient_enc = pd.concat(inpatient_chunks, ignore_index=True)
inpatient_enc['Date'] = pd.to_datetime(inpatient_enc['Date'], errors='coerce')
inpatient_enc['DischargeInstant'] = pd.to_datetime(inpatient_enc['DischargeInstant'], errors='coerce')
inpatient_enc = inpatient_enc.sort_values(['PatientDurableKey', 'Date'])

inpatient_enc['NextAdmitDate'] = inpatient_enc.groupby('PatientDurableKey')['Date'].shift(-1)
inpatient_enc['DaysToNext'] = (inpatient_enc['NextAdmitDate'] - inpatient_enc['DischargeInstant']).dt.days

valid_discharges = inpatient_enc[inpatient_enc['DischargeInstant'].notna()]
readmit_30 = valid_discharges[valid_discharges['DaysToNext'].between(0, 30)]
readmit_patients = set(readmit_30['PatientDurableKey'].unique())
inpatient_patients = set(inpatient_enc['PatientDurableKey'].unique())

overall_readmit_rate = len(readmit_30) / max(len(valid_discharges), 1)
patient_features['readmit_30d'] = patient_features['PatientDurableKey'].isin(readmit_patients).astype(int)

print(f"    住院记录: {len(inpatient_enc):,}")
print(f"    有效出院: {len(valid_discharges):,}")
print(f"    30天再入院: {len(readmit_30):,} ({overall_readmit_rate*100:.2f}%)")
print(f"    再入院患者: {len(readmit_patients):,}")

# 保存特征矩阵
patient_features.to_csv(os.path.join(OUTPUT_DIR, "patient_features.csv"), index=False)
print(f"\n  ✅ 特征矩阵: {patient_features.shape} → output/patient_features.csv")

# =====================================================================
# PHASE 3: 聚类分析
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 3] 聚类分析 (KMeans + PCA)")
print("=" * 70)

# 选择聚类特征
cluster_cols = ['total_encounters', 'ed_visits', 'inpatient_admissions',
                'unique_diagnoses', 'unique_departments', 'span_days',
                'encounters_per_month', 'ed_ratio', 'inpatient_ratio',
                'approx_age', 'is_female', 'is_smoker',
                'sdoh_total_screenings', 'sdoh_unique_domains',
                'diag_chapter_count', 'has_mental_diag', 'has_endocrine_diag',
                'has_circulatory_diag']

# 确保都存在
cluster_cols = [c for c in cluster_cols if c in patient_features.columns]
cluster_cols += [c for c in sdoh_cols if c not in cluster_cols and c in patient_features.columns]

X_cluster = patient_features[cluster_cols].fillna(0).values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 抽样加速
n_sample = min(50000, len(X_scaled))
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), n_sample, replace=False)
X_sample = X_scaled[sample_idx]

# Silhouette score 选 k
print("  寻找最优聚类数 k (silhouette)...")
silhouette_scores = {}
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    score = silhouette_score(X_sample, labels, sample_size=min(10000, n_sample))
    silhouette_scores[k] = score
    print(f"    k={k}: silhouette={score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"  ✅ 最优 k = {best_k} (score={silhouette_scores[best_k]:.4f})")

# 全量聚类
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
patient_features['cluster'] = km_final.fit_predict(X_scaled)

# PCA 降维可视化
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
patient_features['pca_1'] = X_pca[:, 0]
patient_features['pca_2'] = X_pca[:, 1]
patient_features['pca_3'] = X_pca[:, 2]

print(f"  PCA 方差解释率: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
      f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%")

# 各簇画像
print(f"\n  各簇统计摘要:")
cluster_profile = patient_features.groupby('cluster').agg(
    count=('PatientDurableKey', 'count'),
    avg_encounters=('total_encounters', 'mean'),
    avg_ed_ratio=('ed_ratio', 'mean'),
    avg_inpatient_ratio=('inpatient_ratio', 'mean'),
    avg_diagnoses=('unique_diagnoses', 'mean'),
    avg_age=('approx_age', 'mean'),
    avg_sdoh=('sdoh_total_screenings', 'mean'),
    readmit_rate=('readmit_30d', 'mean'),
).round(3)
print(cluster_profile.to_string())

# =====================================================================
# PHASE 4: 30天再入院风险建模 (Random Forest)
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 4] 30天再入院风险建模 (Random Forest)")
print("=" * 70)

# 只对有住院记录的患者建模
model_df = patient_features[patient_features['PatientDurableKey'].isin(inpatient_patients)].copy()
print(f"  建模样本: {len(model_df):,} (有住院记录)")
print(f"  正例(再入院): {model_df['readmit_30d'].sum():,} ({model_df['readmit_30d'].mean()*100:.2f}%)")

# 特征列
feature_cols = [c for c in cluster_cols if c in model_df.columns]
X_model = model_df[feature_cols].fillna(0)
y_model = model_df['readmit_30d']

# 随机森林
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=30,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 交叉验证评估
print("  5-fold 交叉验证...")
cv_scores = cross_val_score(rf, X_model, y_model, cv=5, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 训练最终模型
rf.fit(X_model, y_model)

# 特征重要性
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 风险因子:")
for _, row in importances.head(15).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# =====================================================================
# PHASE 5: SDOH 风险暴露分析 + 统计检验
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 5] SDOH 风险暴露与再入院 — 统计检验")
print("=" * 70)

sdoh_analysis = []
for domain in all_domains:
    col = f"sdoh_{domain.replace(' ', '_').lower()}"
    if col not in patient_features.columns:
        continue
    
    # 只看住院患者
    exposed = model_df[model_df[col] == 1]
    not_exposed = model_df[model_df[col] == 0]
    
    if len(exposed) < 30:  # 样本太小跳过
        continue
    
    rate_exp = exposed['readmit_30d'].mean()
    rate_noexp = not_exposed['readmit_30d'].mean()
    
    # 卡方检验
    contingency = np.array([
        [exposed['readmit_30d'].sum(), len(exposed) - exposed['readmit_30d'].sum()],
        [not_exposed['readmit_30d'].sum(), len(not_exposed) - not_exposed['readmit_30d'].sum()]
    ])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    
    # 相对风险 (RR)
    rr = rate_exp / max(rate_noexp, 0.001)
    
    sdoh_analysis.append({
        'Domain': domain,
        'Exposed_N': len(exposed),
        'Exposed_Readmit_Rate': rate_exp * 100,
        'Not_Exposed_Rate': rate_noexp * 100,
        'Relative_Risk': rr,
        'Chi2': chi2,
        'P_Value': p_val,
        'Significant': p_val < 0.05
    })

sdoh_results = pd.DataFrame(sdoh_analysis).sort_values('Relative_Risk', ascending=False)
print("\n  SDOH Domain 与30天再入院率:")
print(f"  {'Domain':<25} {'暴露率%':>8} {'未暴露率%':>9} {'RR':>6} {'p值':>10} {'显著':>4}")
print("  " + "-" * 65)
for _, row in sdoh_results.iterrows():
    sig = "✓" if row['Significant'] else ""
    print(f"  {row['Domain']:<25} {row['Exposed_Readmit_Rate']:>7.2f}% {row['Not_Exposed_Rate']:>8.2f}% "
          f"{row['Relative_Risk']:>5.2f} {row['P_Value']:>10.2e} {sig:>4}")

# =====================================================================
# PHASE 6: 时序与地理分析
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 6] 时序与地理分析")
print("=" * 70)

# 6a: 月度趋势
print("  6a: 月度就诊趋势...")
monthly = {}
for chunk in pd.read_csv(enc_path, usecols=['Date', 'IsEdVisit', 'IsInpatientAdmission'], chunksize=CHUNK_SIZE):
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk['YM'] = chunk['Date'].dt.to_period('M').astype(str)
    
    agg = chunk.groupby('YM').agg(
        total=('Date', 'count'),
        ed=('IsEdVisit', lambda x: (x == True).sum()),
        inpatient=('IsInpatientAdmission', lambda x: (x == True).sum())
    )
    for ym, row in agg.iterrows():
        if ym not in monthly:
            monthly[ym] = {'total': 0, 'ed': 0, 'inpatient': 0}
        monthly[ym]['total'] += row['total']
        monthly[ym]['ed'] += row['ed']
        monthly[ym]['inpatient'] += row['inpatient']

monthly_df = pd.DataFrame(monthly).T.reset_index()
monthly_df.columns = ['YearMonth', 'Total', 'ED', 'Inpatient']
monthly_df = monthly_df.sort_values('YearMonth')
print(f"    {monthly_df['YearMonth'].iloc[0]} ~ {monthly_df['YearMonth'].iloc[-1]}")

# 6b: 地理分布
print("  6b: 地理分析...")
geo = patients[['PatientDurableKey', 'CensusBlockGroupFipsCode']].copy()
geo = geo[geo['CensusBlockGroupFipsCode'].notna() & (geo['CensusBlockGroupFipsCode'] != '*Unspecified')]
geo['readmit'] = geo['PatientDurableKey'].isin(readmit_patients).astype(int)

geo_summary = geo.groupby('CensusBlockGroupFipsCode').agg(
    patient_count=('PatientDurableKey', 'count'),
    readmit_count=('readmit', 'sum'),
    readmit_rate=('readmit', 'mean')
).reset_index()

# 统一类型后 merge
geo_summary['CensusBlockGroupFipsCode'] = geo_summary['CensusBlockGroupFipsCode'].astype(str)
tiger['GEOID'] = tiger['GEOID'].astype(str)
geo_summary = geo_summary.merge(tiger, left_on='CensusBlockGroupFipsCode', right_on='GEOID', how='left')
print(f"    有地理编码的患者: {len(geo):,}")
print(f"    唯一区域: {len(geo_summary):,}")

# 高风险区域
high_risk_geo = geo_summary[geo_summary['patient_count'] >= 50].nlargest(10, 'readmit_rate')
print(f"    Top 10 高再入院率区域 (>=50患者):")
for _, r in high_risk_geo.iterrows():
    print(f"      {r['CensusBlockGroupFipsCode']}: {r['readmit_rate']*100:.1f}% (n={r['patient_count']})")

# =====================================================================
# PHASE 7: 生成交互式 HTML 报告
# =====================================================================
print("\n" + "=" * 70)
print("[Phase 7] 生成交互式 HTML 报告")
print("=" * 70)

# --- 图1: 缺失率已在 Step1 处理，这里展示特征分布 ---
fig_dist = make_subplots(rows=2, cols=3, subplot_titles=[
    '就诊次数分布', 'ED 比例分布', '住院比例分布',
    '年龄分布', 'SDOH 筛查次数', '诊断大类数'
])
fig_dist.add_trace(go.Histogram(x=patient_features['total_encounters'].clip(0, 100), nbinsx=50, name='就诊次数', marker_color='#3498db'), row=1, col=1)
fig_dist.add_trace(go.Histogram(x=patient_features['ed_ratio'], nbinsx=30, name='ED比例', marker_color='#e74c3c'), row=1, col=2)
fig_dist.add_trace(go.Histogram(x=patient_features['inpatient_ratio'], nbinsx=30, name='住院比例', marker_color='#27ae60'), row=1, col=3)
fig_dist.add_trace(go.Histogram(x=patient_features['approx_age'].dropna().clip(0, 100), nbinsx=40, name='年龄', marker_color='#8e44ad'), row=2, col=1)
fig_dist.add_trace(go.Histogram(x=patient_features['sdoh_total_screenings'].clip(0, 50), nbinsx=30, name='SDOH筛查', marker_color='#f39c12'), row=2, col=2)
fig_dist.add_trace(go.Histogram(x=patient_features['diag_chapter_count'], nbinsx=10, name='诊断大类', marker_color='#1abc9c'), row=2, col=3)
fig_dist.update_layout(height=500, showlegend=False, title_text='患者特征分布', template='plotly_white')

# --- 图2: Silhouette Scores ---
fig_sil = go.Figure(go.Bar(
    x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),
    marker_color=['#e74c3c' if k == best_k else '#3498db' for k in silhouette_scores.keys()],
    text=[f"{v:.3f}" for v in silhouette_scores.values()], textposition='auto'
))
fig_sil.update_layout(title=f'聚类评估: Silhouette Score (最优k={best_k})',
                      xaxis_title='k', yaxis_title='Silhouette Score', template='plotly_white', height=350)

# --- 图3: PCA 聚类散点 ---
plot_n = min(25000, len(patient_features))
plot_df = patient_features.sample(plot_n, random_state=42)
fig_pca = px.scatter(plot_df, x='pca_1', y='pca_2', color=plot_df['cluster'].astype(str),
                     title=f'患者聚类 PCA 2D投影 (k={best_k})',
                     labels={'pca_1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                             'pca_2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                             'color': 'Cluster'},
                     template='plotly_white', opacity=0.5)
fig_pca.update_layout(height=600)

# --- 图4: 聚类雷达图 ---
radar_cols = ['total_encounters', 'ed_ratio', 'inpatient_ratio', 'unique_diagnoses',
              'sdoh_total_screenings', 'approx_age', 'has_mental_diag', 'has_circulatory_diag']
radar_cols = [c for c in radar_cols if c in patient_features.columns]
cluster_means = patient_features.groupby('cluster')[radar_cols].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-10)

fig_radar = go.Figure()
colors = px.colors.qualitative.Set2
for cid in range(best_k):
    vals = cluster_means_norm.loc[cid].values.tolist()
    vals.append(vals[0])
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=radar_cols + [radar_cols[0]], fill='toself',
        name=f'Cluster {cid}', line_color=colors[cid % len(colors)]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title='各簇特征画像 (归一化雷达图)', template='plotly_white', height=550)

# --- 图5: 风险因子 Top 20 ---
fig_imp = px.bar(importances.head(20), x='importance', y='feature', orientation='h',
                 title=f'30天再入院风险因子 Top 20 (RF, CV AUC={cv_scores.mean():.3f})',
                 labels={'importance': 'Feature Importance', 'feature': ''},
                 template='plotly_white', color='importance', color_continuous_scale='Reds')
fig_imp.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})

# --- 图6: SDOH 对比 ---
if len(sdoh_results) > 0:
    fig_sdoh = go.Figure()
    fig_sdoh.add_trace(go.Bar(x=sdoh_results['Domain'], y=sdoh_results['Exposed_Readmit_Rate'],
                              name='有SDOH风险暴露', marker_color='#e74c3c'))
    fig_sdoh.add_trace(go.Bar(x=sdoh_results['Domain'], y=sdoh_results['Not_Exposed_Rate'],
                              name='无SDOH风险暴露', marker_color='#3498db'))
    fig_sdoh.update_layout(title='SDOH 风险暴露与30天再入院率 (住院患者)',
                           barmode='group', template='plotly_white', height=450,
                           xaxis_tickangle=-25, yaxis_title='再入院率(%)')
else:
    fig_sdoh = go.Figure()

# --- 图7: 月度趋势 ---
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=monthly_df['YearMonth'], y=monthly_df['Total'],
                               mode='lines+markers', name='总就诊', line=dict(width=2.5, color='#3498db')))
fig_trend.add_trace(go.Scatter(x=monthly_df['YearMonth'], y=monthly_df['ED'],
                               mode='lines+markers', name='ED就诊', line=dict(width=2, color='#e74c3c')))
fig_trend.add_trace(go.Scatter(x=monthly_df['YearMonth'], y=monthly_df['Inpatient'],
                               mode='lines+markers', name='住院', line=dict(width=2, color='#27ae60')))
fig_trend.update_layout(title='月度就诊量趋势', template='plotly_white', height=450,
                        xaxis_title='月份', yaxis_title='就诊次数', hovermode='x unified')

# --- 图8: 地理散点 ---
geo_plot = geo_summary[geo_summary['patient_count'] >= 10].copy()
fig_geo = px.scatter(geo_plot, x='patient_count', y='readmit_rate',
                     size='patient_count', color='readmit_rate',
                     color_continuous_scale='RdYlGn_r',
                     title='Census Block Group: 患者数 vs 再入院率',
                     labels={'patient_count': '患者数', 'readmit_rate': '30天再入院率'},
                     template='plotly_white', height=500)

# === 组装 HTML ===
print("  组装 HTML...")

# 簇表格
cluster_table = ""
for cid in range(best_k):
    c = patient_features[patient_features['cluster'] == cid]
    cluster_table += f"""<tr>
        <td><strong>Cluster {cid}</strong></td>
        <td>{len(c):,}</td>
        <td>{c['total_encounters'].mean():.1f}</td>
        <td>{c['ed_ratio'].mean()*100:.1f}%</td>
        <td>{c['inpatient_ratio'].mean()*100:.1f}%</td>
        <td>{c['unique_diagnoses'].mean():.1f}</td>
        <td>{c['approx_age'].mean():.0f}</td>
        <td>{c['sdoh_total_screenings'].mean():.1f}</td>
        <td style="font-weight:bold;color:{'#e74c3c' if c['readmit_30d'].mean()>0.02 else '#27ae60'}">{c['readmit_30d'].mean()*100:.2f}%</td>
    </tr>"""

# SDOH 检验表格
sdoh_table = ""
for _, r in sdoh_results.iterrows():
    sig_style = "color:#e74c3c;font-weight:bold" if r['Significant'] else ""
    sdoh_table += f"""<tr>
        <td>{r['Domain']}</td>
        <td>{r['Exposed_N']:,}</td>
        <td>{r['Exposed_Readmit_Rate']:.2f}%</td>
        <td>{r['Not_Exposed_Rate']:.2f}%</td>
        <td>{r['Relative_Risk']:.2f}</td>
        <td style="{sig_style}">{r['P_Value']:.2e}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>DataFest 2026 - 数据挖掘分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif;
                max-width: 1400px; margin: 0 auto; padding: 30px; background: #f5f7fa; color: #2c3e50; line-height: 1.7; }}
        h1 {{ color: #1a1a2e; font-size: 2.2em; border-bottom: 4px solid #3498db; padding-bottom: 15px; }}
        h2 {{ color: #2c3e50; margin-top: 50px; border-left: 5px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 25px 0; }}
        .card {{ background: white; border-radius: 12px; padding: 20px; text-align: center;
                 box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-2px); }}
        .card .val {{ font-size: 2em; font-weight: 700; color: #3498db; }}
        .card .lbl {{ color: #7f8c8d; font-size: 0.85em; margin-top: 5px; }}
        .chart {{ background: white; border-radius: 12px; padding: 20px; margin: 25px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }}
        th {{ background: #3498db; color: white; padding: 12px; font-weight: 600; font-size: 0.9em; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #edf2f7; font-size: 0.9em; }}
        tr:hover {{ background: #f7fafc; }}
        .insight {{ background: #eaf4fe; border-left: 5px solid #3498db; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .warning {{ background: #fff8e1; border-left: 5px solid #f39c12; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .method {{ background: #f0fdf4; border-left: 5px solid #27ae60; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>DataFest 2026 — 数据挖掘分析报告</h1>
    <p style="color:#7f8c8d">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 基于清理后数据</p>

    <h2>1. 数据概览</h2>
    <div class="grid">
        <div class="card"><div class="val">{len(patient_features):,}</div><div class="lbl">唯一患者</div></div>
        <div class="card"><div class="val">{patient_features['total_encounters'].sum():,.0f}</div><div class="lbl">总就诊次数</div></div>
        <div class="card"><div class="val">{len(inpatient_enc):,}</div><div class="lbl">住院记录</div></div>
        <div class="card"><div class="val">{overall_readmit_rate*100:.1f}%</div><div class="lbl">30天再入院率</div></div>
        <div class="card"><div class="val">{len(readmit_patients):,}</div><div class="lbl">再入院患者</div></div>
        <div class="card"><div class="val">{len(all_domains)}</div><div class="lbl">SDOH Domains</div></div>
        <div class="card"><div class="val">{best_k}</div><div class="lbl">聚类数 (最优k)</div></div>
        <div class="card"><div class="val">{cv_scores.mean():.3f}</div><div class="lbl">RF CV AUC</div></div>
    </div>

    <h2>2. 患者特征分布</h2>
    <div class="method"><strong>方法:</strong> 对 {len(patient_features):,} 名患者构建 {len(cluster_cols)} 维特征矩阵，包含就诊模式、人口统计学、SDOH暴露、诊断分类。</div>
    <div class="chart" id="fig_dist"></div>

    <h2>3. 患者聚类分析</h2>
    <div class="method"><strong>方法:</strong> StandardScaler 标准化 → KMeans (k=3~8, silhouette score 选最优) → PCA 3D降维可视化。抽样 {n_sample:,} 进行 silhouette 评估。</div>
    <div class="chart" id="fig_sil"></div>
    <div class="chart" id="fig_pca"></div>
    <div class="chart" id="fig_radar"></div>
    <h3>各簇统计摘要</h3>
    <table>
        <tr><th>簇</th><th>患者数</th><th>平均就诊</th><th>ED比例</th><th>住院比例</th><th>诊断数</th><th>平均年龄</th><th>SDOH筛查</th><th>再入院率</th></tr>
        {cluster_table}
    </table>

    <h2>4. 30天再入院风险因子</h2>
    <div class="method"><strong>方法:</strong> Random Forest (n=300, max_depth=12, class_weight=balanced) 对 {len(model_df):,} 名住院患者建模。5-fold CV ROC-AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}。目标不是预测精度，而是特征重要性的可解释性排序。</div>
    <div class="chart" id="fig_imp"></div>
    <div class="insight"><strong>关键发现:</strong> Top 5 风险因子为: <strong>{', '.join(importances.head(5)['feature'].tolist())}</strong></div>

    <h2>5. SDOH 风险暴露分析</h2>
    <div class="method"><strong>方法:</strong> 对每个 SDOH Domain，比较有暴露 vs 无暴露的住院患者 30天再入院率，使用卡方检验评估显著性 (α=0.05)，计算相对风险 (RR)。</div>
    <div class="chart" id="fig_sdoh"></div>
    <h3>统计检验结果</h3>
    <table>
        <tr><th>SDOH Domain</th><th>暴露人数</th><th>暴露再入院率</th><th>未暴露再入院率</th><th>相对风险RR</th><th>p值</th></tr>
        {sdoh_table}
    </table>

    <h2>6. 时序趋势</h2>
    <div class="chart" id="fig_trend"></div>

    <h2>7. 地理分布</h2>
    <div class="chart" id="fig_geo"></div>
    <div class="insight"><strong>高风险区域:</strong> 在患者数≥50的 Census Block Group 中，再入院率最高达 {high_risk_geo['readmit_rate'].iloc[0]*100:.1f}%。</div>

    <h2>8. 方法论说明</h2>
    <div class="warning">
        <strong>⚠️ 统计注意事项:</strong><br>
        • 聚类分析使用 KMeans（假设球形簇），对非球形分布可能不理想<br>
        • RF 特征重要性反映预测贡献而非因果关系 (correlation ≠ causation)<br>
        • SDOH 暴露是观察性数据，存在选择偏差（被筛查的患者可能本身健康管理更积极）<br>
        • 30天再入院标签基于同一系统内记录，可能遗漏转院患者<br>
        • 样本量说明: 多重比较已报告但未做 Bonferroni 校正
    </div>

    <script>
        Plotly.newPlot('fig_dist', {fig_dist.to_json()}.data, {fig_dist.to_json()}.layout);
        Plotly.newPlot('fig_sil', {fig_sil.to_json()}.data, {fig_sil.to_json()}.layout);
        Plotly.newPlot('fig_pca', {fig_pca.to_json()}.data, {fig_pca.to_json()}.layout);
        Plotly.newPlot('fig_radar', {fig_radar.to_json()}.data, {fig_radar.to_json()}.layout);
        Plotly.newPlot('fig_imp', {fig_imp.to_json()}.data, {fig_imp.to_json()}.layout);
        Plotly.newPlot('fig_sdoh', {fig_sdoh.to_json()}.data, {fig_sdoh.to_json()}.layout);
        Plotly.newPlot('fig_trend', {fig_trend.to_json()}.data, {fig_trend.to_json()}.layout);
        Plotly.newPlot('fig_geo', {fig_geo.to_json()}.data, {fig_geo.to_json()}.layout);
    </script>
</body>
</html>"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"  ✅ 报告: {REPORT_PATH}")

# 保存最终特征矩阵
patient_features.to_csv(os.path.join(OUTPUT_DIR, "patient_features.csv"), index=False)
importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
sdoh_results.to_csv(os.path.join(OUTPUT_DIR, "sdoh_analysis.csv"), index=False)

print(f"\n" + "=" * 70)
print("Step 2 完成")
print("=" * 70)
print(f"  产出物:")
print(f"    {REPORT_PATH}")
print(f"    {OUTPUT_DIR}/patient_features.csv")
print(f"    {OUTPUT_DIR}/feature_importances.csv")
print(f"    {OUTPUT_DIR}/sdoh_analysis.csv")
print(f"\n  关键结论:")
print(f"    • 最优聚类 k={best_k}, silhouette={silhouette_scores[best_k]:.3f}")
print(f"    • 30天再入院率: {overall_readmit_rate*100:.2f}%")
print(f"    • RF CV AUC: {cv_scores.mean():.3f}")
print(f"    • Top 风险因子: {importances.iloc[0]['feature']}")
if len(sdoh_results[sdoh_results['Significant']]) > 0:
    print(f"    • 显著SDOH风险: {', '.join(sdoh_results[sdoh_results['Significant']]['Domain'].tolist())}")
print("=" * 70)
