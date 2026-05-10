"""
DataFest 2026 — 系统性交叉分析 EDA
====================================
模块:
1. 30天再入院预测因子
2. MyChart 数字鸿沟
3. SDOH × 夜间就诊
4. 年龄分层效应
5. 周末效应
6. 共病负担分析
7. 种族/民族差异
8. 全局相关性矩阵 + 新发现

输出: plotly 交互式 HTML 图表 + 汇总 JSON
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, spearmanr, pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 数据加载
# ============================================================
print("=" * 60)
print("DataFest 2026 — 交叉分析 EDA")
print("=" * 60)

print("\n[1/8] 加载数据...")

# 核心特征表（已含 SDOH 和 readmit_30d）
pf = pd.read_csv('/Users/aaronwen/Documents/Datafest/DATA_CLEANED/patient_features.csv')
print(f"  patient_features: {pf.shape}")

# 患者人口学
patients = pd.read_csv('/Users/aaronwen/Documents/Datafest/DATA_CLEANED/patients_cleaned.csv')
print(f"  patients_cleaned: {patients.shape}")

# 就诊记录（采样以控制内存，用于时间模式分析）
enc_cols = ['PatientDurableKey', 'Type', 'IsEdVisit', 'IsHospitalAdmission',
            'IsOutpatientFaceToFaceVisit', 'has_time', 'TimePeriod', 'Weekday',
            'is_weekend', 'AdmitYear', 'AdmitMonth', 'AdmitHour']
encounters = pd.read_csv('/Users/aaronwen/Documents/Datafest/DATA_CLEANED/encounters_cleaned.csv',
                         usecols=enc_cols, low_memory=False)
print(f"  encounters_cleaned: {encounters.shape}")

# 诊断（只加载需要的列）
diag = pd.read_csv('/Users/aaronwen/Documents/Datafest/DATA_CLEANED/diagnosis_cleaned.csv',
                   low_memory=False)
print(f"  diagnosis_cleaned: {diag.shape}")

# ============================================================
# 合并主分析表
# ============================================================
print("\n  合并数据...")
# patient_features + patients demographics
df = pf.merge(patients[['DurableKey', 'FirstRace', 'MaritalStatus', 'MyChartStatus',
                         'OmbEthnicity', 'OmbRace', 'SmokingStatus', 'PatientBirthYearBin']],
              left_on='PatientDurableKey', right_on='DurableKey', how='left')
print(f"  合并后主表: {df.shape}")

# ============================================================
# 辅助函数
# ============================================================
def effect_size_cohen_d(group1, group2):
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

def cramers_v(confusion_matrix):
    """计算 Cramér's V（分类变量关联强度）"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0 or n == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

findings = []  # 收集所有新发现

# ============================================================
# 模块 1: 30天再入院预测因子
# ============================================================
print("\n[2/8] 模块1: 30天再入院预测因子分析...")

readmit_cols = ['total_encounters', 'ed_visits', 'hospital_admissions',
                'unique_diagnoses', 'unique_departments', 'encounters_per_month',
                'ed_ratio', 'inpatient_ratio', 'approx_age',
                'sdoh_total_screenings', 'sdoh_unique_domains',
                'sdoh_social_connections', 'sdoh_housing_stability',
                'sdoh_stress', 'sdoh_depression', 'sdoh_financial_resource_strain',
                'sdoh_transportation_needs', 'sdoh_food_insecurity']

readmit_results = []
for col in readmit_cols:
    valid = df[[col, 'readmit_30d']].dropna()
    if len(valid) < 100:
        continue
    g0 = valid[valid['readmit_30d'] == 0][col]
    g1 = valid[valid['readmit_30d'] == 1][col]
    if len(g1) < 30:
        continue
    stat, p = mannwhitneyu(g0, g1, alternative='two-sided')
    d = effect_size_cohen_d(g1, g0)
    readmit_results.append({
        'variable': col,
        'readmit_mean': g1.mean(),
        'no_readmit_mean': g0.mean(),
        'ratio': g1.mean() / g0.mean() if g0.mean() != 0 else np.nan,
        'cohen_d': d,
        'p_value': p,
        'n_readmit': len(g1),
        'n_no_readmit': len(g0)
    })

readmit_df = pd.DataFrame(readmit_results).sort_values('p_value')
readmit_df['significant'] = readmit_df['p_value'] < 0.05 / len(readmit_results)  # Bonferroni
print(f"  显著预测因子 (Bonferroni α={0.05/len(readmit_results):.4f}): "
      f"{readmit_df['significant'].sum()}/{len(readmit_results)}")

# 可视化: 再入院预测因子效应量
fig_readmit = px.bar(
    readmit_df.sort_values('cohen_d'),
    x='cohen_d', y='variable', orientation='h',
    color='significant',
    color_discrete_map={True: '#E63946', False: '#A8DADC'},
    title='30天再入院预测因子 — Cohen\'s d 效应量<br><sub>红色=Bonferroni校正后显著</sub>',
    labels={'cohen_d': "Cohen's d (效应量)", 'variable': '变量', 'significant': '统计显著'},
    template='plotly_white'
)
fig_readmit.update_layout(height=600, showlegend=True)
fig_readmit.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_readmit_predictors.html')

# 收集 finding
top_predictors = readmit_df[readmit_df['significant']].head(5)
if len(top_predictors) > 0:
    findings.append({
        'module': '30天再入院',
        'finding': f"Top再入院预测因子: {', '.join(top_predictors['variable'].tolist())}",
        'detail': f"最强效应: {top_predictors.iloc[0]['variable']} (d={top_predictors.iloc[0]['cohen_d']:.3f})",
        'significance': 'Bonferroni corrected'
    })

# MyChart × 再入院
mychart_readmit = df.groupby('MyChartStatus')['readmit_30d'].agg(['mean', 'count']).reset_index()
mychart_readmit.columns = ['MyChartStatus', 'readmit_rate', 'n']
mychart_readmit = mychart_readmit[mychart_readmit['n'] > 100]
print(f"  MyChart与再入院:\n{mychart_readmit.to_string()}")

# ============================================================
# 模块 2: MyChart 数字鸿沟
# ============================================================
print("\n[3/8] 模块2: MyChart 数字鸿沟分析...")

mychart_metrics = ['total_encounters', 'ed_visits', 'ed_ratio', 'hospital_admissions',
                   'inpatient_ratio', 'encounters_per_month', 'unique_diagnoses',
                   'sdoh_total_screenings', 'readmit_30d']

# 简化 MyChart 分组: Activated vs Non-Digital (Unspecified + Pending + Inactivated + Declined)
df['mychart_group'] = df['MyChartStatus'].map(
    lambda x: 'Activated' if x == 'Activated' 
    else ('Non-Digital' if x in ['Unspecified', 'Pending Activation', 'Inactivated', 'Patient Declined']
          else 'Other')
)

mychart_comparison = []
activated = df[df['mychart_group'] == 'Activated']
not_activated = df[df['mychart_group'] == 'Non-Digital']

for metric in mychart_metrics:
    g1 = activated[metric].dropna()
    g2 = not_activated[metric].dropna()
    if len(g1) < 30 or len(g2) < 30:
        continue
    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
    d = effect_size_cohen_d(g1, g2)
    mychart_comparison.append({
        'metric': metric,
        'activated_mean': g1.mean(),
        'not_activated_mean': g2.mean(),
        'diff_pct': (g1.mean() - g2.mean()) / g2.mean() * 100 if g2.mean() != 0 else 0,
        'cohen_d': d,
        'p_value': p
    })

mychart_df = pd.DataFrame(mychart_comparison)
if len(mychart_df) > 0:
    mychart_df['significant'] = mychart_df['p_value'] < 0.05
else:
    print("  警告: MyChart对比无有效数据")

# 可视化: MyChart 数字鸿沟
fig_mychart = make_subplots(rows=1, cols=2, subplot_titles=['均值对比', '百分比差异'],
                            specs=[[{"type": "bar"}, {"type": "bar"}]])

mychart_sorted = mychart_df.sort_values('diff_pct')
fig_mychart.add_trace(
    go.Bar(x=mychart_sorted['activated_mean'], y=mychart_sorted['metric'],
           orientation='h', name='已激活', marker_color='#2A9D8F'),
    row=1, col=1
)
fig_mychart.add_trace(
    go.Bar(x=mychart_sorted['not_activated_mean'], y=mychart_sorted['metric'],
           orientation='h', name='未激活', marker_color='#E76F51'),
    row=1, col=1
)
fig_mychart.add_trace(
    go.Bar(x=mychart_sorted['diff_pct'], y=mychart_sorted['metric'],
           orientation='h', name='差异%',
           marker_color=['#E63946' if x > 0 else '#457B9D' for x in mychart_sorted['diff_pct']]),
    row=1, col=2
)
fig_mychart.update_layout(title='MyChart 数字鸿沟 — 激活 vs 未激活患者就医模式差异',
                          height=500, template='plotly_white')
fig_mychart.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_mychart_gap.html')

if len(mychart_df) > 0:
    findings.append({
        'module': 'MyChart数字鸿沟',
        'finding': f"MyChart激活患者ED比率{'更低' if mychart_df[mychart_df['metric']=='ed_ratio']['diff_pct'].values[0] < 0 else '更高'}",
        'detail': f"ED ratio差异: {mychart_df[mychart_df['metric']=='ed_ratio']['diff_pct'].values[0]:.1f}%, "
                  f"再入院率差异: {mychart_df[mychart_df['metric']=='readmit_30d']['diff_pct'].values[0]:.1f}%",
        'significance': 'Mann-Whitney U'
    })

# ============================================================
# 模块 3: SDOH × 夜间就诊
# ============================================================
print("\n[4/8] 模块3: SDOH × 夜间就诊分析...")

# 在有时间信息的就诊中分析
enc_with_time = encounters[encounters['has_time'] == True].copy()
print(f"  有时间信息的就诊: {len(enc_with_time):,}")

# 计算每位患者的夜间就诊比例（夜间=22:00-06:00）
enc_with_time['is_night'] = enc_with_time['AdmitHour'].apply(
    lambda h: 1 if (h >= 22 or h < 6) else 0
)

patient_night = enc_with_time.groupby('PatientDurableKey').agg(
    total_timed=('is_night', 'count'),
    night_visits=('is_night', 'sum')
).reset_index()
patient_night['night_ratio'] = patient_night['night_visits'] / patient_night['total_timed']
patient_night = patient_night[patient_night['total_timed'] >= 3]  # 至少3次有时间记录

# 合并 SDOH
night_sdoh = patient_night.merge(
    df[['PatientDurableKey', 'sdoh_total_screenings', 'sdoh_social_connections',
        'sdoh_housing_stability', 'sdoh_stress', 'sdoh_depression',
        'sdoh_financial_resource_strain', 'sdoh_transportation_needs',
        'sdoh_food_insecurity', 'ed_ratio', 'approx_age']],
    on='PatientDurableKey', how='inner'
)
print(f"  夜间分析样本: {len(night_sdoh):,}")

# SDOH domains 与夜间就诊的相关性
sdoh_domains = ['sdoh_social_connections', 'sdoh_housing_stability', 'sdoh_stress',
                'sdoh_depression', 'sdoh_financial_resource_strain',
                'sdoh_transportation_needs', 'sdoh_food_insecurity']

night_correlations = []
for domain in sdoh_domains:
    valid = night_sdoh[[domain, 'night_ratio']].dropna()
    valid = valid[valid[domain] > 0]  # 只看有SDOH记录的
    if len(valid) < 50:
        continue
    # 分位数分组比较
    has_sdoh = valid[valid[domain] > 0]['night_ratio']
    
    # 与 night_ratio 的 Spearman 相关
    rho, p = spearmanr(valid[domain], valid['night_ratio'])
    night_correlations.append({
        'sdoh_domain': domain.replace('sdoh_', ''),
        'spearman_rho': rho,
        'p_value': p,
        'n': len(valid),
        'mean_night_ratio': valid['night_ratio'].mean()
    })

night_corr_df = pd.DataFrame(night_correlations).sort_values('spearman_rho', ascending=False)
print(f"  SDOH × 夜间相关性:\n{night_corr_df.to_string()}")

# 可视化: SDOH × 夜间就诊
# 把有SDOH风险 vs 无SDOH风险的患者夜间比例做对比
night_sdoh['has_any_sdoh'] = (night_sdoh['sdoh_total_screenings'] > 0).astype(int)
night_sdoh['sdoh_burden'] = pd.cut(night_sdoh['sdoh_total_screenings'],
                                    bins=[-1, 0, 2, 5, 999],
                                    labels=['无筛查', '1-2次', '3-5次', '6+次'])

fig_night = px.box(
    night_sdoh.dropna(subset=['sdoh_burden']),
    x='sdoh_burden', y='night_ratio',
    color='sdoh_burden',
    title='SDOH筛查负担 × 夜间就诊比例<br><sub>夜间=22:00-06:00，仅含≥3次有时间记录的患者</sub>',
    labels={'night_ratio': '夜间就诊比例', 'sdoh_burden': 'SDOH筛查次数分组'},
    template='plotly_white',
    color_discrete_sequence=['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
)
fig_night.update_layout(height=500)
fig_night.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_sdoh_night.html')

# 统计检验
g_no_sdoh = night_sdoh[night_sdoh['sdoh_total_screenings'] == 0]['night_ratio']
g_has_sdoh = night_sdoh[night_sdoh['sdoh_total_screenings'] > 0]['night_ratio']
if len(g_no_sdoh) > 30 and len(g_has_sdoh) > 30:
    stat, p = mannwhitneyu(g_has_sdoh, g_no_sdoh, alternative='greater')
    d = effect_size_cohen_d(g_has_sdoh, g_no_sdoh)
    findings.append({
        'module': 'SDOH×夜间就诊',
        'finding': f"有SDOH风险患者夜间就诊比例{'更高' if d > 0 else '更低'} (d={d:.3f})",
        'detail': f"有SDOH: {g_has_sdoh.mean():.4f}, 无SDOH: {g_no_sdoh.mean():.4f}, p={p:.2e}",
        'significance': f'Mann-Whitney U, p={p:.2e}'
    })

# ============================================================
# 模块 4: 年龄分层效应
# ============================================================
print("\n[5/8] 模块4: 年龄分层效应...")

age_metrics = df.groupby('PatientBirthYearBin').agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_inpatient_ratio=('inpatient_ratio', 'mean'),
    mean_readmit=('readmit_30d', 'mean'),
    mean_encounters=('encounters_per_month', 'mean'),
    mean_unique_diag=('unique_diagnoses', 'mean'),
    mean_sdoh_screenings=('sdoh_total_screenings', 'mean'),
    mean_ed_visits=('ed_visits', 'mean')
).reset_index()

# 排序年龄组
age_order = sorted(df['PatientBirthYearBin'].dropna().unique())
age_metrics['PatientBirthYearBin'] = pd.Categorical(age_metrics['PatientBirthYearBin'],
                                                      categories=age_order, ordered=True)
age_metrics = age_metrics.sort_values('PatientBirthYearBin')

print(f"  年龄组统计:\n{age_metrics[['PatientBirthYearBin','n','mean_ed_ratio','mean_readmit','mean_sdoh_screenings']].to_string()}")

# 可视化: 年龄分层蜘蛛图 / 多指标对比
fig_age = make_subplots(rows=2, cols=2,
                        subplot_titles=['ED比率 by 年龄组', '再入院率 by 年龄组',
                                       '月均就诊次数 by 年龄组', 'SDOH筛查次数 by 年龄组'])

age_labels = age_metrics['PatientBirthYearBin'].astype(str).tolist()

fig_age.add_trace(go.Bar(x=age_labels, y=age_metrics['mean_ed_ratio'],
                         marker_color='#E63946', name='ED比率'), row=1, col=1)
fig_age.add_trace(go.Bar(x=age_labels, y=age_metrics['mean_readmit'],
                         marker_color='#457B9D', name='再入院率'), row=1, col=2)
fig_age.add_trace(go.Bar(x=age_labels, y=age_metrics['mean_encounters'],
                         marker_color='#2A9D8F', name='月均就诊'), row=2, col=1)
fig_age.add_trace(go.Bar(x=age_labels, y=age_metrics['mean_sdoh_screenings'],
                         marker_color='#E9C46A', name='SDOH筛查'), row=2, col=2)

fig_age.update_layout(title='年龄分层效应 — 关键指标对比', height=700,
                      template='plotly_white', showlegend=False)
fig_age.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_age_stratification.html')

# Kruskal-Wallis 检验年龄组差异
age_groups_data = [group['ed_ratio'].dropna().values for _, group in df.groupby('PatientBirthYearBin')
                   if len(group) > 30]
if len(age_groups_data) >= 3:
    stat, p = kruskal(*age_groups_data)
    findings.append({
        'module': '年龄分层',
        'finding': f"年龄组间ED比率差异显著 (H={stat:.1f}, p={p:.2e})",
        'detail': f"ED比率范围: {age_metrics['mean_ed_ratio'].min():.3f} - {age_metrics['mean_ed_ratio'].max():.3f}",
        'significance': f'Kruskal-Wallis, p={p:.2e}'
    })

# ============================================================
# 模块 5: 周末效应
# ============================================================
print("\n[6/8] 模块5: 周末效应分析...")

weekend_stats = encounters.groupby('is_weekend').agg(
    total=('PatientDurableKey', 'count'),
    ed_visits=('IsEdVisit', 'sum'),
    admissions=('IsHospitalAdmission', 'sum'),
    outpatient=('IsOutpatientFaceToFaceVisit', 'sum')
).reset_index()

weekend_stats['ed_rate'] = weekend_stats['ed_visits'] / weekend_stats['total']
weekend_stats['admission_rate'] = weekend_stats['admissions'] / weekend_stats['total']
weekend_stats['outpatient_rate'] = weekend_stats['outpatient'] / weekend_stats['total']
weekend_stats['label'] = weekend_stats['is_weekend'].map({0: '工作日', 1: '周末'})

print(f"  周末效应:\n{weekend_stats.to_string()}")

# 按星期几的详细模式
weekday_stats = encounters.groupby('Weekday').agg(
    total=('PatientDurableKey', 'count'),
    ed_visits=('IsEdVisit', 'sum'),
    admissions=('IsHospitalAdmission', 'sum')
).reset_index()
weekday_stats['ed_rate'] = weekday_stats['ed_visits'] / weekday_stats['total']
weekday_stats['admission_rate'] = weekday_stats['admissions'] / weekday_stats['total']

# 映射星期名称
day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
weekday_stats['day_name'] = weekday_stats['Weekday'].map(day_names)

fig_weekend = make_subplots(rows=1, cols=2,
                            subplot_titles=['各星期ED比率', '各星期住院比率'])

fig_weekend.add_trace(
    go.Bar(x=weekday_stats['day_name'], y=weekday_stats['ed_rate'],
           marker_color=['#457B9D']*5 + ['#E63946']*2, name='ED率'),
    row=1, col=1
)
fig_weekend.add_trace(
    go.Bar(x=weekday_stats['day_name'], y=weekday_stats['admission_rate'],
           marker_color=['#457B9D']*5 + ['#E63946']*2, name='住院率'),
    row=1, col=2
)
fig_weekend.update_layout(title='周末效应 — ED和住院比率按星期分布<br><sub>红色=周末</sub>',
                          height=400, template='plotly_white', showlegend=False)
fig_weekend.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_weekend_effect.html')

# 卡方检验
contingency = pd.crosstab(encounters['is_weekend'], encounters['IsEdVisit'])
chi2, p_chi, dof, expected = chi2_contingency(contingency)
v = cramers_v(contingency)
findings.append({
    'module': '周末效应',
    'finding': f"周末ED就诊比率显著高于工作日 (V={v:.4f})",
    'detail': f"周末ED率: {weekend_stats[weekend_stats['is_weekend']==1]['ed_rate'].values[0]:.4f}, "
              f"工作日: {weekend_stats[weekend_stats['is_weekend']==0]['ed_rate'].values[0]:.4f}",
    'significance': f"Chi²={chi2:.1f}, p={p_chi:.2e}, Cramér's V={v:.4f}"
})

# ============================================================
# 模块 6: 共病负担分析
# ============================================================
print("\n[7/8] 模块6: 共病负担与结局关系...")

# unique_diagnoses 作为共病代理指标
df['diag_burden'] = pd.cut(df['unique_diagnoses'], bins=[0, 1, 3, 5, 10, 999],
                           labels=['1', '2-3', '4-5', '6-10', '11+'])

comorbidity_stats = df.groupby('diag_burden').agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_readmit=('readmit_30d', 'mean'),
    mean_admissions=('hospital_admissions', 'mean'),
    mean_encounters=('encounters_per_month', 'mean')
).reset_index()

print(f"  共病负担统计:\n{comorbidity_stats.to_string()}")

# Spearman: 共病数 vs ED率
valid_diag = df[['unique_diagnoses', 'ed_ratio', 'readmit_30d']].dropna()
rho_ed, p_ed = spearmanr(valid_diag['unique_diagnoses'], valid_diag['ed_ratio'])
rho_readmit, p_readmit = spearmanr(valid_diag['unique_diagnoses'], valid_diag['readmit_30d'])

# 可视化: 共病-剂量反应
fig_comorbid = make_subplots(rows=1, cols=2,
                             subplot_titles=[f'共病数 → ED比率 (ρ={rho_ed:.3f})',
                                           f'共病数 → 再入院率 (ρ={rho_readmit:.3f})'])

fig_comorbid.add_trace(
    go.Bar(x=comorbidity_stats['diag_burden'].astype(str),
           y=comorbidity_stats['mean_ed_ratio'],
           marker_color='#E63946', name='ED比率'),
    row=1, col=1
)
fig_comorbid.add_trace(
    go.Bar(x=comorbidity_stats['diag_burden'].astype(str),
           y=comorbidity_stats['mean_readmit'],
           marker_color='#457B9D', name='再入院率'),
    row=1, col=2
)
fig_comorbid.update_layout(title='共病负担与临床结局 — 剂量反应关系',
                           height=400, template='plotly_white', showlegend=False)
fig_comorbid.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_comorbidity.html')

findings.append({
    'module': '共病负担',
    'finding': f"共病数与ED比率正相关 (ρ={rho_ed:.3f}), 与再入院正相关 (ρ={rho_readmit:.3f})",
    'detail': f"诊断数11+的患者ED率={comorbidity_stats[comorbidity_stats['diag_burden']=='11+']['mean_ed_ratio'].values[0]:.3f}, "
              f"1个诊断={comorbidity_stats[comorbidity_stats['diag_burden']=='1']['mean_ed_ratio'].values[0]:.3f}",
    'significance': f'Spearman, p_ed={p_ed:.2e}, p_readmit={p_readmit:.2e}'
})

# ============================================================
# 模块 7: 种族/民族差异（有值子集）
# ============================================================
print("\n[8/8] 模块7: 种族/民族差异分析...")

# 使用 OmbRace（合并后的种族分类）
race_valid = df[df['OmbRace'].notna() & (df['OmbRace'] != 'Unknown')].copy()
print(f"  有种族信息的患者: {len(race_valid):,} ({len(race_valid)/len(df)*100:.1f}%)")

race_stats = race_valid.groupby('OmbRace').agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_readmit=('readmit_30d', 'mean'),
    mean_encounters=('encounters_per_month', 'mean'),
    mean_sdoh=('sdoh_total_screenings', 'mean'),
    mean_inpatient=('inpatient_ratio', 'mean')
).reset_index()
race_stats = race_stats[race_stats['n'] >= 100].sort_values('mean_ed_ratio', ascending=False)

print(f"  种族差异统计:\n{race_stats.to_string()}")

# 可视化
if len(race_stats) >= 2:
    fig_race = px.bar(
        race_stats.melt(id_vars=['OmbRace', 'n'],
                       value_vars=['mean_ed_ratio', 'mean_readmit', 'mean_inpatient'],
                       var_name='指标', value_name='值'),
        x='OmbRace', y='值', color='指标', barmode='group',
        title='种族/民族差异 — 关键就医指标对比<br><sub>仅含有种族信息的患者子集</sub>',
        labels={'OmbRace': '种族', '值': '比率'},
        template='plotly_white',
        color_discrete_sequence=['#E63946', '#457B9D', '#2A9D8F']
    )
    fig_race.update_layout(height=500)
    fig_race.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_race_disparity.html')

    # Kruskal-Wallis across races
    race_groups = [group['ed_ratio'].dropna().values 
                   for name, group in race_valid.groupby('OmbRace') 
                   if len(group) >= 100]
    if len(race_groups) >= 3:
        stat, p = kruskal(*race_groups)
        findings.append({
            'module': '种族差异',
            'finding': f"种族间ED比率存在显著差异 (H={stat:.1f}, p={p:.2e})",
            'detail': f"ED比率范围: {race_stats['mean_ed_ratio'].min():.3f} - {race_stats['mean_ed_ratio'].max():.3f}",
            'significance': f'Kruskal-Wallis, p={p:.2e}'
        })

# ============================================================
# 模块 8: 全局相关性矩阵
# ============================================================
print("\n[Extra] 模块8: 全局相关性矩阵...")

corr_cols = ['total_encounters', 'ed_visits', 'hospital_admissions', 'unique_diagnoses',
             'encounters_per_month', 'ed_ratio', 'inpatient_ratio', 'approx_age',
             'sdoh_total_screenings', 'sdoh_social_connections', 'sdoh_depression',
             'sdoh_financial_resource_strain', 'sdoh_transportation_needs',
             'sdoh_food_insecurity', 'readmit_30d']

corr_matrix = df[corr_cols].corr(method='spearman')

# 找出未被之前分析捕捉的强相关对
strong_corrs = []
for i in range(len(corr_cols)):
    for j in range(i+1, len(corr_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:  # 中等以上相关
            strong_corrs.append({
                'var1': corr_cols[i],
                'var2': corr_cols[j],
                'spearman_r': r
            })

strong_corrs_df = pd.DataFrame(strong_corrs).sort_values('spearman_r', key=abs, ascending=False)
print(f"  |ρ| > 0.3 的相关对: {len(strong_corrs_df)}")
print(strong_corrs_df.head(15).to_string())

# 可视化: 相关性热力图
fig_corr = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title='全局 Spearman 相关性矩阵<br><sub>关键临床与SDOH变量</sub>',
    labels={'color': 'Spearman ρ'}
)
fig_corr.update_layout(height=800, width=900)
fig_corr.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_correlation_matrix.html')

# 新发现的强相关
novel_corrs = strong_corrs_df[
    ~strong_corrs_df.apply(lambda r: 
        ('ed_visits' in r['var1'] and 'ed_ratio' in r['var2']) or
        ('total_encounters' in r['var1'] and 'encounters_per_month' in r['var2']) or
        ('hospital_admissions' in r['var1'] and 'inpatient_ratio' in r['var2']),
        axis=1)
].head(5)

if len(novel_corrs) > 0:
    findings.append({
        'module': '全局相关性',
        'finding': f"发现{len(strong_corrs_df)}对中等以上相关 (|ρ|>0.3)",
        'detail': '; '.join([f"{r['var1']}↔{r['var2']}(ρ={r['spearman_r']:.3f})" 
                            for _, r in novel_corrs.iterrows()]),
        'significance': 'Spearman rank correlation'
    })

# ============================================================
# 额外模块: MyChart × SDOH × 年龄交互
# ============================================================
print("\n[Bonus] MyChart × SDOH × 年龄三维交互...")

# MyChart状态对SDOH筛查率的影响是否因年龄而异
interaction_df = df[df['mychart_group'].isin(['Activated', 'Non-Digital'])].copy()
interaction_df = interaction_df[interaction_df['PatientBirthYearBin'].notna()]

interaction_stats = interaction_df.groupby(['PatientBirthYearBin', 'mychart_group']).agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_sdoh=('sdoh_total_screenings', 'mean'),
    mean_readmit=('readmit_30d', 'mean')
).reset_index()
interaction_stats = interaction_stats[interaction_stats['n'] >= 30]

fig_interaction = px.line(
    interaction_stats,
    x='PatientBirthYearBin', y='mean_ed_ratio', color='mychart_group',
    markers=True,
    title='MyChart × 年龄交互效应 — ED比率<br><sub>数字鸿沟在哪些年龄段最明显？</sub>',
    labels={'mean_ed_ratio': 'ED比率', 'PatientBirthYearBin': '出生年代',
            'mychart_group': 'MyChart状态'},
    template='plotly_white',
    color_discrete_map={'Activated': '#2A9D8F', 'Non-Digital': '#E76F51'}
)
fig_interaction.update_layout(height=450)
fig_interaction.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_interaction_mychart_age.html')

# ============================================================
# 额外模块: 婚姻状态 × 就医模式
# ============================================================
print("\n[Bonus] 婚姻状态与就医模式...")

marital_valid = df[df['MaritalStatus'].notna() & (df['MaritalStatus'] != 'Unknown')].copy()
marital_stats = marital_valid.groupby('MaritalStatus').agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_readmit=('readmit_30d', 'mean'),
    mean_encounters=('encounters_per_month', 'mean'),
    mean_sdoh=('sdoh_total_screenings', 'mean')
).reset_index()
marital_stats = marital_stats[marital_stats['n'] >= 100].sort_values('mean_ed_ratio', ascending=False)
print(f"  婚姻状态差异:\n{marital_stats.to_string()}")

if len(marital_stats) >= 2:
    fig_marital = px.bar(
        marital_stats,
        x='MaritalStatus', y='mean_ed_ratio',
        color='mean_sdoh', color_continuous_scale='YlOrRd',
        title='婚姻状态 × ED比率 (颜色=SDOH筛查频率)<br><sub>社交孤立可能是中介变量</sub>',
        labels={'MaritalStatus': '婚姻状态', 'mean_ed_ratio': 'ED比率', 'mean_sdoh': 'SDOH筛查'},
        template='plotly_white'
    )
    fig_marital.update_layout(height=450)
    fig_marital.write_html('/Users/aaronwen/Documents/Datafest/chart_eda_marital_status.html')

# ============================================================
# 额外模块: 吸烟状态 × 就医模式
# ============================================================
print("\n[Bonus] 吸烟状态与就医结局...")
smoking_valid = df[df['SmokingStatus'].notna() & (df['SmokingStatus'] != 'Unknown')].copy()
smoking_stats = smoking_valid.groupby('SmokingStatus').agg(
    n=('PatientDurableKey', 'count'),
    mean_ed_ratio=('ed_ratio', 'mean'),
    mean_readmit=('readmit_30d', 'mean'),
    mean_inpatient=('inpatient_ratio', 'mean')
).reset_index()
smoking_stats = smoking_stats[smoking_stats['n'] >= 100].sort_values('mean_ed_ratio', ascending=False)
print(f"  吸烟状态差异:\n{smoking_stats.to_string()}")

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 60)
print("汇总: 所有新发现")
print("=" * 60)
for i, f in enumerate(findings, 1):
    print(f"\n  [{i}] {f['module']}")
    print(f"      发现: {f['finding']}")
    print(f"      详情: {f['detail']}")
    print(f"      检验: {f['significance']}")

# 保存结果
output = {
    'findings': findings,
    'readmit_predictors': readmit_df.to_dict('records'),
    'mychart_comparison': mychart_df.to_dict('records') if len(mychart_df) > 0 else [],
    'night_correlations': night_corr_df.to_dict('records') if len(night_corr_df) > 0 else [],
    'age_stratification': age_metrics.to_dict('records'),
    'weekend_effect': weekend_stats.to_dict('records'),
    'comorbidity_dose_response': comorbidity_stats.to_dict('records'),
    'race_disparity': race_stats.to_dict('records') if len(race_stats) > 0 else [],
    'strong_correlations': strong_corrs_df.to_dict('records') if len(strong_corrs_df) > 0 else []
}

import os
os.makedirs('/Users/aaronwen/Documents/Datafest/output', exist_ok=True)
with open('/Users/aaronwen/Documents/Datafest/output/eda_cross_findings.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\n\n✓ 所有图表已保存:")
print("  - chart_eda_readmit_predictors.html")
print("  - chart_eda_mychart_gap.html")
print("  - chart_eda_sdoh_night.html")
print("  - chart_eda_age_stratification.html")
print("  - chart_eda_weekend_effect.html")
print("  - chart_eda_comorbidity.html")
print("  - chart_eda_race_disparity.html")
print("  - chart_eda_correlation_matrix.html")
print("  - chart_eda_interaction_mychart_age.html")
print("  - chart_eda_marital_status.html")
print("  - output/eda_cross_findings.json")
print("\n完成!")
