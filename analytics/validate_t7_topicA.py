"""
DataFest 2026 - Topic 7 & Topic A 数据验证脚本
================================================
验证核心假设:
1. T7: T2D 首诊 → Depression/Anxiety 首诊的时间差中位数是否在 3-9 个月
2. Topic A: 30天再入院率是否显著，SDOH 特征差异
3. 协同: 心理共病患者是否更容易再入院
"""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
from datetime import timedelta

warnings.filterwarnings('ignore')

DATA_DIR = "/Users/aaronwen/Documents/Datafest/DATA"
OUTPUT_PATH = "/Users/aaronwen/Documents/Datafest/validation_report.html"

print("=" * 60)
print("DataFest 2026 - T7 & Topic A 数据验证")
print("=" * 60)

# =============================================================
# STEP 1: 加载 diagnosis 表，筛选目标诊断
# =============================================================
print("\n[Step 1] 加载 diagnosis.csv，筛选 T2D / Depression / Anxiety...")

diagnosis = pd.read_csv(os.path.join(DATA_DIR, "diagnosis.csv"))
print(f"  diagnosis 总行数: {len(diagnosis):,}")

# ICD-10 匹配
def classify_diagnosis(row):
    dv = str(row['DiagnosisValue']).upper()
    gn = str(row['GroupName']).lower()
    
    if dv.startswith('E11') or 'type 2 diabetes' in gn:
        return 'T2D'
    elif dv.startswith('F32') or dv.startswith('F33') or 'depressive' in gn:
        return 'Depression'
    elif dv.startswith('F41') or 'anxiety' in gn:
        return 'Anxiety'
    return None

diagnosis['Category'] = diagnosis.apply(classify_diagnosis, axis=1)
target_diag = diagnosis[diagnosis['Category'].notna()].copy()
print(f"  T2D 诊断码数: {(target_diag['Category']=='T2D').sum():,}")
print(f"  Depression 诊断码数: {(target_diag['Category']=='Depression').sum():,}")
print(f"  Anxiety 诊断码数: {(target_diag['Category']=='Anxiety').sum():,}")

# 构建 DiagnosisKey → Category 映射
diag_key_to_cat = dict(zip(target_diag['DiagnosisKey'], target_diag['Category']))
target_keys = set(target_diag['DiagnosisKey'].values)

# =============================================================
# STEP 2: Chunked 读取 encounters，构建目标子集
# =============================================================
print("\n[Step 2] Chunked 读取 encounters.csv...")

# 只需要这些列
usecols = ['Date', 'EncounterKey', 'PatientDurableKey', 'PrimaryDiagnosisKey',
           'IsEdVisit', 'IsHospitalAdmission', 'IsInpatientAdmission',
           'DischargeInstant', 'AdmitYear']

chunks = []
chunk_size = 500000
total_rows = 0
matched_rows = 0

for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"), 
                         usecols=usecols, chunksize=chunk_size):
    total_rows += len(chunk)
    # 筛选包含目标诊断的行
    mask = chunk['PrimaryDiagnosisKey'].isin(target_keys)
    filtered = chunk[mask].copy()
    matched_rows += len(filtered)
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"    已处理 {total_rows:,} 行, 匹配 {matched_rows:,} 行...", end='\r')

encounters_target = pd.concat(chunks, ignore_index=True)
print(f"\n  encounters 总行数: {total_rows:,}")
print(f"  匹配目标诊断的行数: {len(encounters_target):,}")

# 添加诊断分类
encounters_target['DiagCategory'] = encounters_target['PrimaryDiagnosisKey'].map(diag_key_to_cat)
encounters_target['Date'] = pd.to_datetime(encounters_target['Date'], errors='coerce')
encounters_target['DischargeInstant'] = pd.to_datetime(encounters_target['DischargeInstant'], errors='coerce')

print(f"  T2D encounters: {(encounters_target['DiagCategory']=='T2D').sum():,}")
print(f"  Depression encounters: {(encounters_target['DiagCategory']=='Depression').sum():,}")
print(f"  Anxiety encounters: {(encounters_target['DiagCategory']=='Anxiety').sum():,}")

# =============================================================
# STEP 3: T7 验证 - T2D 首诊 → 心理诊断时间差
# =============================================================
print("\n[Step 3] T7 验证: 计算 T2D 首诊 → Depression/Anxiety 首诊时间差...")

# 每个患者每个诊断类别的首诊日期
first_diag = encounters_target.groupby(['PatientDurableKey', 'DiagCategory'])['Date'].min().reset_index()
first_diag.columns = ['PatientDurableKey', 'DiagCategory', 'FirstDate']

# Pivot: 每个患者一行，三列（T2D/Depression/Anxiety 的首诊日期）
patient_first = first_diag.pivot(index='PatientDurableKey', columns='DiagCategory', values='FirstDate').reset_index()

# 有 T2D 首诊的患者
t2d_patients = patient_first[patient_first['T2D'].notna()].copy()
print(f"  有 T2D 诊断的唯一患者数: {len(t2d_patients):,}")

# 限定 2022 年首诊 T2D 的队列（更聚焦）
t2d_2022 = t2d_patients[t2d_patients['T2D'].dt.year == 2022].copy()
print(f"  2022 年首诊 T2D 的患者数: {len(t2d_2022):,}")

# 也看整体（不限年份）
t2d_all = t2d_patients.copy()

# 计算时间差（月）
def calc_gap_months(row):
    gaps = []
    if pd.notna(row.get('Depression')) and row['Depression'] > row['T2D']:
        gaps.append((row['Depression'] - row['T2D']).days / 30.44)
    if pd.notna(row.get('Anxiety')) and row['Anxiety'] > row['T2D']:
        gaps.append((row['Anxiety'] - row['T2D']).days / 30.44)
    if gaps:
        return min(gaps)  # 取最早的心理诊断
    return None

# 全体 T2D 患者中，后续有心理诊断的
t2d_all['MentalHealthGap_Months'] = t2d_all.apply(calc_gap_months, axis=1)
t2d_2022['MentalHealthGap_Months'] = t2d_2022.apply(calc_gap_months, axis=1)

# 统计
has_mh_all = t2d_all[t2d_all['MentalHealthGap_Months'].notna()]
has_mh_2022 = t2d_2022[t2d_2022['MentalHealthGap_Months'].notna()]

print(f"\n  --- 全体 T2D 患者 (首诊任意年份) ---")
print(f"  T2D 后有心理诊断的患者数: {len(has_mh_all):,} / {len(t2d_all):,} ({100*len(has_mh_all)/len(t2d_all):.1f}%)")
if len(has_mh_all) > 0:
    print(f"  时间差中位数: {has_mh_all['MentalHealthGap_Months'].median():.1f} 个月")
    print(f"  时间差均值: {has_mh_all['MentalHealthGap_Months'].mean():.1f} 个月")
    print(f"  时间差 25%分位: {has_mh_all['MentalHealthGap_Months'].quantile(0.25):.1f} 个月")
    print(f"  时间差 75%分位: {has_mh_all['MentalHealthGap_Months'].quantile(0.75):.1f} 个月")

print(f"\n  --- 2022 首诊 T2D 队列 ---")
print(f"  T2D 后有心理诊断的患者数: {len(has_mh_2022):,} / {len(t2d_2022):,} ({100*len(has_mh_2022)/max(len(t2d_2022),1):.1f}%)")
if len(has_mh_2022) > 0:
    print(f"  时间差中位数: {has_mh_2022['MentalHealthGap_Months'].median():.1f} 个月")
    print(f"  时间差均值: {has_mh_2022['MentalHealthGap_Months'].mean():.1f} 个月")
    print(f"  时间差 25%分位: {has_mh_2022['MentalHealthGap_Months'].quantile(0.25):.1f} 个月")
    print(f"  时间差 75%分位: {has_mh_2022['MentalHealthGap_Months'].quantile(0.75):.1f} 个月")

# =============================================================
# STEP 4: Topic A 验证 - 30 天再入院率
# =============================================================
print("\n[Step 4] Topic A 验证: 30 天再入院率...")

# 需要重新读取所有 inpatient encounters（不限于目标诊断）
print("  重新 chunked 读取 encounters（inpatient admissions）...")

inpatient_cols = ['Date', 'EncounterKey', 'PatientDurableKey', 'PrimaryDiagnosisKey',
                  'IsInpatientAdmission', 'IsEdVisit', 'DischargeInstant']

inpatient_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                         usecols=inpatient_cols, chunksize=chunk_size):
    # 只保留住院
    mask = chunk['IsInpatientAdmission'] == True
    filtered = chunk[mask].copy()
    if len(filtered) > 0:
        inpatient_chunks.append(filtered)

if inpatient_chunks:
    inpatient_enc = pd.concat(inpatient_chunks, ignore_index=True)
else:
    # 如果 IsInpatientAdmission 没有 True（可能是字符串），尝试其他方式
    inpatient_chunks2 = []
    for chunk in pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"),
                             usecols=inpatient_cols, chunksize=chunk_size):
        mask = chunk['IsInpatientAdmission'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
        filtered = chunk[mask].copy()
        if len(filtered) > 0:
            inpatient_chunks2.append(filtered)
    inpatient_enc = pd.concat(inpatient_chunks2, ignore_index=True) if inpatient_chunks2 else pd.DataFrame()

print(f"  Inpatient admissions 总数: {len(inpatient_enc):,}")

if len(inpatient_enc) > 0:
    inpatient_enc['Date'] = pd.to_datetime(inpatient_enc['Date'], errors='coerce')
    inpatient_enc['DischargeInstant'] = pd.to_datetime(inpatient_enc['DischargeInstant'], errors='coerce')
    inpatient_enc = inpatient_enc.sort_values(['PatientDurableKey', 'Date'])
    
    # 30天再入院标记
    inpatient_enc['NextAdmitDate'] = inpatient_enc.groupby('PatientDurableKey')['Date'].shift(-1)
    inpatient_enc['DaysToNext'] = (inpatient_enc['NextAdmitDate'] - inpatient_enc['DischargeInstant']).dt.days
    
    # 有效出院记录（有 DischargeInstant）
    valid_discharges = inpatient_enc[inpatient_enc['DischargeInstant'].notna()].copy()
    readmit_30 = valid_discharges[valid_discharges['DaysToNext'].between(0, 30)]
    
    readmit_rate = len(readmit_30) / max(len(valid_discharges), 1)
    print(f"  有效出院记录数: {len(valid_discharges):,}")
    print(f"  30天内再入院次数: {len(readmit_30):,}")
    print(f"  30天再入院率: {readmit_rate*100:.2f}%")
    
    # 再入院患者 vs 非再入院患者
    readmit_patients = set(readmit_30['PatientDurableKey'].unique())
    non_readmit_patients = set(valid_discharges['PatientDurableKey'].unique()) - readmit_patients
    print(f"  有再入院经历的患者数: {len(readmit_patients):,}")
    print(f"  无再入院经历的患者数: {len(non_readmit_patients):,}")
else:
    readmit_rate = 0
    readmit_patients = set()
    non_readmit_patients = set()
    valid_discharges = pd.DataFrame()
    readmit_30 = pd.DataFrame()

# =============================================================
# STEP 5: 协同验证 - 心理共病 × 再入院
# =============================================================
print("\n[Step 5] 协同验证: 心理共病患者是否更容易再入院...")

# 有心理诊断的患者集合
mental_health_patients = set()
if 'Depression' in patient_first.columns:
    mental_health_patients.update(
        patient_first[patient_first['Depression'].notna()]['PatientDurableKey'].values
    )
if 'Anxiety' in patient_first.columns:
    mental_health_patients.update(
        patient_first[patient_first['Anxiety'].notna()]['PatientDurableKey'].values
    )

print(f"  有心理诊断（Depression/Anxiety）的患者总数: {len(mental_health_patients):,}")

if len(readmit_patients) > 0:
    # 再入院患者中有心理共病的比例
    readmit_with_mh = readmit_patients & mental_health_patients
    non_readmit_with_mh = non_readmit_patients & mental_health_patients
    
    rate_readmit_mh = len(readmit_with_mh) / max(len(readmit_patients), 1)
    rate_non_readmit_mh = len(non_readmit_with_mh) / max(len(non_readmit_patients), 1)
    
    print(f"  再入院患者中有心理共病: {len(readmit_with_mh):,}/{len(readmit_patients):,} = {rate_readmit_mh*100:.1f}%")
    print(f"  非再入院患者中有心理共病: {len(non_readmit_with_mh):,}/{len(non_readmit_patients):,} = {rate_non_readmit_mh*100:.1f}%")
    
    # 卡方检验
    contingency = np.array([
        [len(readmit_with_mh), len(readmit_patients) - len(readmit_with_mh)],
        [len(non_readmit_with_mh), len(non_readmit_patients) - len(non_readmit_with_mh)]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(f"  卡方检验: chi2={chi2:.2f}, p={p_value:.4e}")
    print(f"  结论: {'显著' if p_value < 0.05 else '不显著'} (alpha=0.05)")

# =============================================================
# STEP 6: SDOH 分析
# =============================================================
print("\n[Step 6] 加载 SDOH 数据，对比再入院/非再入院患者...")

sdoh = pd.read_csv(os.path.join(DATA_DIR, "social_determinants.csv"))
print(f"  SDOH 记录数: {len(sdoh):,}")
print(f"  SDOH domains: {sdoh['Domain'].nunique()} 个")
print(f"  Domains: {sdoh['Domain'].unique().tolist()}")

if len(readmit_patients) > 0 and len(sdoh) > 0:
    # 再入院患者 vs 非再入院患者的 SDOH 对比
    sdoh_readmit = sdoh[sdoh['PatientDurableKey'].isin(readmit_patients)]
    sdoh_non_readmit = sdoh[sdoh['PatientDurableKey'].isin(non_readmit_patients)]
    
    # 按 Domain 统计回答率和阳性率
    domain_comparison = []
    for domain in sdoh['Domain'].unique():
        r_count = sdoh_readmit[sdoh_readmit['Domain'] == domain]['PatientDurableKey'].nunique()
        nr_count = sdoh_non_readmit[sdoh_non_readmit['Domain'] == domain]['PatientDurableKey'].nunique()
        r_rate = r_count / max(len(readmit_patients), 1)
        nr_rate = nr_count / max(len(non_readmit_patients), 1)
        domain_comparison.append({
            'Domain': domain,
            'Readmit_Patients': r_count,
            'NonReadmit_Patients': nr_count,
            'Readmit_Rate': r_rate,
            'NonReadmit_Rate': nr_rate
        })
    
    domain_df = pd.DataFrame(domain_comparison)
    print("\n  SDOH Domain 覆盖率对比（再入院 vs 非再入院）:")
    print(domain_df[['Domain', 'Readmit_Rate', 'NonReadmit_Rate']].to_string(index=False))

# =============================================================
# STEP 7: 生成 HTML 报告
# =============================================================
print("\n[Step 7] 生成 HTML 综合验证报告...")

# 图表1: T7 时间差分布
fig1 = go.Figure()
if len(has_mh_all) > 0:
    fig1.add_trace(go.Histogram(
        x=has_mh_all['MentalHealthGap_Months'],
        nbinsx=30,
        name='全体 T2D→心理诊断',
        marker_color='steelblue',
        opacity=0.7
    ))
if len(has_mh_2022) > 0:
    fig1.add_trace(go.Histogram(
        x=has_mh_2022['MentalHealthGap_Months'],
        nbinsx=20,
        name='2022首诊队列',
        marker_color='coral',
        opacity=0.7
    ))

# 添加关键阈值线
if len(has_mh_all) > 0:
    median_val = has_mh_all['MentalHealthGap_Months'].median()
    fig1.add_vline(x=median_val, line_dash="dash", line_color="red",
                   annotation_text=f"中位数: {median_val:.1f}月")
    fig1.add_vline(x=3, line_dash="dot", line_color="green",
                   annotation_text="3个月")
    fig1.add_vline(x=9, line_dash="dot", line_color="orange",
                   annotation_text="9个月")

fig1.update_layout(
    title="T7 验证: T2D 首诊 → Depression/Anxiety 首诊 时间差分布",
    xaxis_title="时间差（月）",
    yaxis_title="患者数",
    barmode='overlay',
    template='plotly_white'
)

# 图表2: 30天再入院率
fig2 = go.Figure()
if len(valid_discharges) > 0:
    # 按月统计再入院率
    valid_discharges_copy = valid_discharges.copy()
    valid_discharges_copy['DischargeMonth'] = valid_discharges_copy['DischargeInstant'].dt.to_period('M').astype(str)
    monthly_stats = valid_discharges_copy.groupby('DischargeMonth').agg(
        total=('EncounterKey', 'count'),
        readmit=('DaysToNext', lambda x: (x.between(0, 30)).sum())
    ).reset_index()
    monthly_stats['rate'] = monthly_stats['readmit'] / monthly_stats['total'] * 100
    
    fig2.add_trace(go.Bar(
        x=monthly_stats['DischargeMonth'],
        y=monthly_stats['rate'],
        marker_color='indianred',
        name='30天再入院率(%)'
    ))
    fig2.add_hline(y=readmit_rate*100, line_dash="dash", line_color="black",
                   annotation_text=f"总体平均: {readmit_rate*100:.1f}%")

fig2.update_layout(
    title="Topic A 验证: 30天再入院率月度趋势",
    xaxis_title="出院月份",
    yaxis_title="再入院率(%)",
    template='plotly_white'
)

# 图表3: 心理共病 × 再入院 对比
fig3 = go.Figure()
if len(readmit_patients) > 0:
    fig3.add_trace(go.Bar(
        x=['再入院患者', '非再入院患者'],
        y=[rate_readmit_mh*100, rate_non_readmit_mh*100],
        marker_color=['#e74c3c', '#2ecc71'],
        text=[f'{rate_readmit_mh*100:.1f}%', f'{rate_non_readmit_mh*100:.1f}%'],
        textposition='auto'
    ))
    fig3.update_layout(
        title=f"协同验证: 心理共病比例 (再入院 vs 非再入院) | χ²={chi2:.1f}, p={p_value:.2e}",
        yaxis_title="有心理共病的比例(%)",
        template='plotly_white'
    )

# 图表4: SDOH Domain 对比
fig4 = go.Figure()
if len(readmit_patients) > 0 and len(domain_df) > 0:
    fig4.add_trace(go.Bar(
        x=domain_df['Domain'],
        y=domain_df['Readmit_Rate']*100,
        name='再入院',
        marker_color='#e74c3c'
    ))
    fig4.add_trace(go.Bar(
        x=domain_df['Domain'],
        y=domain_df['NonReadmit_Rate']*100,
        name='非再入院',
        marker_color='#3498db'
    ))
    fig4.update_layout(
        title="SDOH Domain 覆盖率对比 (再入院 vs 非再入院)",
        xaxis_title="SDOH Domain",
        yaxis_title="被筛查比例(%)",
        barmode='group',
        template='plotly_white',
        xaxis_tickangle=-45
    )

# 图表5: 时间差的箱线图（按是否再入院分组）
fig5 = go.Figure()
if len(has_mh_all) > 0 and len(readmit_patients) > 0:
    has_mh_all_copy = has_mh_all.copy()
    has_mh_all_copy['Readmitted'] = has_mh_all_copy['PatientDurableKey'].isin(readmit_patients)
    
    for label, group in has_mh_all_copy.groupby('Readmitted'):
        fig5.add_trace(go.Box(
            y=group['MentalHealthGap_Months'],
            name='有再入院' if label else '无再入院',
            marker_color='#e74c3c' if label else '#2ecc71'
        ))
    fig5.update_layout(
        title="T2D→心理诊断时间差: 再入院 vs 非再入院患者",
        yaxis_title="时间差（月）",
        template='plotly_white'
    )

# 构建 HTML 报告
html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>DataFest 2026 - T7 & Topic A 数据验证报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
        }}
        .summary-box {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .verdict {{
            font-size: 1.3em;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .verdict-pass {{
            background: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
        }}
        .verdict-fail {{
            background: #f8d7da;
            color: #721c24;
            border-left: 5px solid #dc3545;
        }}
        .verdict-caution {{
            background: #fff3cd;
            color: #856404;
            border-left: 5px solid #ffc107;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 1px 5px rgba(0,0,0,0.08);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .caveat {{
            background: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 12px 18px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f1f2f6;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <h1>🔬 DataFest 2026 - T7 & Topic A 数据验证报告</h1>
    <p style="color:#7f8c8d;">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>📊 关键数据概览</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{len(t2d_all):,}</div>
            <div class="stat-label">有 T2D 诊断的唯一患者</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(t2d_2022):,}</div>
            <div class="stat-label">2022年首诊 T2D 患者</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(has_mh_all):,}</div>
            <div class="stat-label">T2D后有心理诊断的患者</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{readmit_rate*100:.1f}%</div>
            <div class="stat-label">30天再入院率</div>
        </div>
    </div>

    <h2>✅ 验证结论总结</h2>
    <div class="summary-box">
        <h3>Topic 7: T2D → 心理诊断时间差</h3>
"""

# 动态判断结论
if len(has_mh_all) > 0:
    median_all = has_mh_all['MentalHealthGap_Months'].median()
    if 3 <= median_all <= 9:
        verdict_t7 = "pass"
        verdict_t7_text = f"✅ 假设成立: 中位时间差 = {median_all:.1f} 个月，在 3-9 个月区间内"
    elif median_all < 3:
        verdict_t7 = "fail"
        verdict_t7_text = f"❌ 假设不成立: 中位时间差 = {median_all:.1f} 个月，小于 3 个月（建议的筛查时点可能太晚）"
    else:
        verdict_t7 = "caution"
        verdict_t7_text = f"⚠️ 需要注意: 中位时间差 = {median_all:.1f} 个月，大于 9 个月（需排查数据 rollout artifact）"
    
    comorbidity_rate = len(has_mh_all) / len(t2d_all) * 100
else:
    verdict_t7 = "fail"
    verdict_t7_text = "❌ 数据不足，无法验证"
    median_all = 0
    comorbidity_rate = 0

html_content += f"""
        <div class="verdict verdict-{verdict_t7}">
            {verdict_t7_text}
        </div>
        <ul>
            <li>T2D 患者中 {comorbidity_rate:.1f}% 后续被诊断 Depression 或 Anxiety</li>
            <li>全体: 中位数 = {median_all:.1f} 月 | 均值 = {has_mh_all['MentalHealthGap_Months'].mean():.1f} 月</li>
            <li>IQR: [{has_mh_all['MentalHealthGap_Months'].quantile(0.25):.1f}, {has_mh_all['MentalHealthGap_Months'].quantile(0.75):.1f}] 月</li>
        </ul>
"""

# Topic A 结论
if readmit_rate > 0:
    if readmit_rate >= 0.10:
        verdict_a = "pass"
        verdict_a_text = f"✅ 30天再入院率 = {readmit_rate*100:.1f}%，有显著的改善空间"
    elif readmit_rate >= 0.05:
        verdict_a = "caution"
        verdict_a_text = f"⚠️ 30天再入院率 = {readmit_rate*100:.1f}%，中等水平，仍有故事可讲"
    else:
        verdict_a = "fail"
        verdict_a_text = f"❌ 30天再入院率 = {readmit_rate*100:.1f}%，偏低，Topic A 可能缺乏足够的改善空间"
else:
    verdict_a = "fail"
    verdict_a_text = "❌ 无法计算再入院率（无 inpatient 数据）"

html_content += f"""
        <h3>Topic A: 30天再入院率</h3>
        <div class="verdict verdict-{verdict_a}">
            {verdict_a_text}
        </div>
"""

# 协同结论
if len(readmit_patients) > 0:
    html_content += f"""
        <h3>协同验证: 心理共病 × 再入院</h3>
        <div class="verdict verdict-{'pass' if p_value < 0.05 else 'fail'}">
            {'✅' if p_value < 0.05 else '❌'} 心理共病在再入院患者中的比例 ({rate_readmit_mh*100:.1f}%) vs 非再入院 ({rate_non_readmit_mh*100:.1f}%) — χ²={chi2:.1f}, p={p_value:.2e}
        </div>
        <p>{'心理共病显著增加再入院风险，T7+A 的协同叙事有数据支撑。' if p_value < 0.05 else '心理共病与再入院的关联不显著，需要重新考虑 T7+A 的协同逻辑。'}</p>
"""

html_content += """
    </div>
    
    <h2>📈 详细图表</h2>
    
    <h3>图1: T2D → 心理诊断时间差分布</h3>
    <div class="chart-container" id="chart1"></div>
    
    <h3>图2: 30天再入院率月度趋势</h3>
    <div class="chart-container" id="chart2"></div>
    
    <h3>图3: 心理共病 × 再入院</h3>
    <div class="chart-container" id="chart3"></div>
    
    <h3>图4: SDOH Domain 覆盖率对比</h3>
    <div class="chart-container" id="chart4"></div>
    
    <h3>图5: 时间差箱线图（按再入院分组）</h3>
    <div class="chart-container" id="chart5"></div>
    
    <h2>⚠️ Skeptical 注意点</h2>
    <div class="caveat">
        <strong>1. 数据 rollout artifact:</strong> SDOH 问卷在 2022-2023 年期间可能有逐步推广的过程。如果心理诊断的时间差恰好等于 SDOH 推广周期，结果可能是假阳性。
    </div>
    <div class="caveat">
        <strong>2. 首诊定义偏差:</strong> 数据只覆盖 SVH 系统内的就诊记录。患者可能在 SVH 外已被诊断 T2D 或 Depression，此处看到的"首诊"不一定是真正的首诊。
    </div>
    <div class="caveat">
        <strong>3. 存活偏差:</strong> 只有留在 SVH 系统内的患者才会被观察到后续心理诊断。那些诊断后就离开系统的患者（可能是最严重的）被遗漏了。
    </div>
    <div class="caveat">
        <strong>4. 样本量警告:</strong> 如果 2022 首诊队列过小（<100），统计推断可能不够稳健。
    </div>
"""

html_content += f"""
    <script>
        var chart1 = {fig1.to_json()};
        var chart2 = {fig2.to_json()};
        var chart3 = {fig3.to_json()};
        var chart4 = {fig4.to_json()};
        var chart5 = {fig5.to_json()};
        
        Plotly.newPlot('chart1', chart1.data, chart1.layout);
        Plotly.newPlot('chart2', chart2.data, chart2.layout);
        Plotly.newPlot('chart3', chart3.data, chart3.layout);
        Plotly.newPlot('chart4', chart4.data, chart4.layout);
        Plotly.newPlot('chart5', chart5.data, chart5.layout);
    </script>
</body>
</html>
"""

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ HTML 报告已保存至: {OUTPUT_PATH}")

# =============================================================
# 最终结论
# =============================================================
print("\n" + "=" * 60)
print("最终验证结论")
print("=" * 60)
if len(has_mh_all) > 0:
    print(f"\n[T7] 时间差中位数: {median_all:.1f} 个月")
    if 3 <= median_all <= 9:
        print("  → ✅ Topic 7 假设成立，建议继续推进")
    elif median_all < 3:
        print("  → ⚠️ 时间差偏短，'第3个月筛查'的建议时间点可能太晚")
        print("  → 建议: 调整为'诊断后第1个月即启动心理筛查'")
    else:
        print("  → ⚠️ 时间差偏长，需检查是否为数据 rollout 造成的假象")
        print("  → 建议: 切换到 T7+G (Silent Drop Off)")

print(f"\n[Topic A] 30天再入院率: {readmit_rate*100:.1f}%")
if readmit_rate >= 0.05:
    print("  → ✅ 再入院率可观，Topic A 有数据支撑")
else:
    print("  → ❌ 再入院率偏低，Topic A 可能缺乏足够改善空间")

if len(readmit_patients) > 0 and p_value < 0.05:
    print(f"\n[T7+A 协同] 心理共病显著关联再入院 (p={p_value:.2e})")
    print("  → ✅ T7+A 组合有数据支撑")
elif len(readmit_patients) > 0:
    print(f"\n[T7+A 协同] 心理共病与再入院关联不显著 (p={p_value:.4f})")
    print("  → ❌ 需重新考虑组合策略")

print("\n" + "=" * 60)
