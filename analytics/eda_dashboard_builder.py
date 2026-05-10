"""
DataFest 2026 — EDA 汇总 Dashboard（整合所有新发现）
========================================================
生成一个包含全部发现的交互式 HTML Dashboard 报告
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 加载所有发现
with open('/Users/aaronwen/Documents/Datafest/output/eda_cross_findings.json', 'r') as f:
    data = json.load(f)

# ============================================================
# 创建汇总 Dashboard HTML
# ============================================================

# 1. 再入院预测因子图
readmit_df = pd.DataFrame(data['readmit_predictors'])
readmit_df = readmit_df.sort_values('cohen_d')

fig1 = go.Figure()
colors = ['#E63946' if sig else '#A8DADC' for sig in readmit_df['significant']]
fig1.add_trace(go.Bar(
    x=readmit_df['cohen_d'], y=readmit_df['variable'],
    orientation='h', marker_color=colors,
    text=[f"d={d:.3f}" for d in readmit_df['cohen_d']],
    textposition='outside'
))
fig1.update_layout(
    title="30天再入院预测因子 — Cohen's d 效应量<br><sub>全部18个因子均通过Bonferroni校正(α=0.003)</sub>",
    xaxis_title="Cohen's d", yaxis_title="",
    template='plotly_white', height=550
)

# 2. MyChart 数字鸿沟
mychart_df = pd.DataFrame(data['mychart_comparison'])
if len(mychart_df) > 0:
    mychart_sorted = mychart_df.sort_values('diff_pct')
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=mychart_sorted['diff_pct'], y=mychart_sorted['metric'],
        orientation='h',
        marker_color=['#E63946' if x > 0 else '#2A9D8F' for x in mychart_sorted['diff_pct']],
        text=[f"{x:+.1f}%" for x in mychart_sorted['diff_pct']],
        textposition='outside'
    ))
    fig2.update_layout(
        title="MyChart 数字鸿沟 — 激活 vs 非数字用户差异(%)<br><sub>正值=激活用户更高 | 负值=激活用户更低</sub>",
        xaxis_title="百分比差异", yaxis_title="",
        template='plotly_white', height=450
    )
else:
    fig2 = go.Figure()
    fig2.add_annotation(text="No data", showarrow=False)

# 3. 年龄分层
age_df = pd.DataFrame(data['age_stratification'])
# 过滤掉样本量太小的组
age_df = age_df[age_df['n'] >= 100]
age_df['age_label'] = age_df['PatientBirthYearBin'].astype(str)

fig3 = make_subplots(rows=2, cols=2,
                     subplot_titles=['ED比率', '再入院率', '月均就诊', 'SDOH筛查次数'])
fig3.add_trace(go.Scatter(x=age_df['age_label'], y=age_df['mean_ed_ratio'],
                          mode='lines+markers', marker_color='#E63946', name='ED比率'),
               row=1, col=1)
fig3.add_trace(go.Scatter(x=age_df['age_label'], y=age_df['mean_readmit'],
                          mode='lines+markers', marker_color='#457B9D', name='再入院'),
               row=1, col=2)
fig3.add_trace(go.Scatter(x=age_df['age_label'], y=age_df['mean_encounters'],
                          mode='lines+markers', marker_color='#2A9D8F', name='月均就诊'),
               row=2, col=1)
fig3.add_trace(go.Scatter(x=age_df['age_label'], y=age_df['mean_sdoh_screenings'],
                          mode='lines+markers', marker_color='#E9C46A', name='SDOH筛查'),
               row=2, col=2)
fig3.update_layout(title="年龄分层效应 — 出生年代 vs 关键指标", height=600,
                   template='plotly_white', showlegend=False)

# 4. 周末效应
weekend_df = pd.DataFrame(data['weekend_effect'])
fig4 = go.Figure(data=[
    go.Bar(name='ED率', x=['工作日', '周末'],
           y=[weekend_df[weekend_df['is_weekend']==0]['ed_rate'].values[0],
              weekend_df[weekend_df['is_weekend']==1]['ed_rate'].values[0]],
           marker_color=['#457B9D', '#E63946']),
    go.Bar(name='住院率', x=['工作日', '周末'],
           y=[weekend_df[weekend_df['is_weekend']==0]['admission_rate'].values[0],
              weekend_df[weekend_df['is_weekend']==1]['admission_rate'].values[0]],
           marker_color=['#A8DADC', '#E9C46A'])
])
fig4.update_layout(title="周末效应 — ED率与住院率对比<br><sub>周末ED率是工作日的8.7倍 (Cramér's V=0.219)</sub>",
                   barmode='group', template='plotly_white', height=400)

# 5. 共病剂量反应
comorbid_df = pd.DataFrame(data['comorbidity_dose_response'])
fig5 = make_subplots(rows=1, cols=2, subplot_titles=['共病数→ED比率 (ρ=0.219)', '共病数→再入院率 (ρ=0.139)'])
fig5.add_trace(go.Bar(x=comorbid_df['diag_burden'].astype(str), y=comorbid_df['mean_ed_ratio'],
                      marker_color='#E63946'), row=1, col=1)
fig5.add_trace(go.Bar(x=comorbid_df['diag_burden'].astype(str), y=comorbid_df['mean_readmit'],
                      marker_color='#457B9D'), row=1, col=2)
fig5.update_layout(title="共病负担与临床结局 — 剂量反应关系", height=400,
                   template='plotly_white', showlegend=False)

# 6. 种族差异
race_df = pd.DataFrame(data['race_disparity'])
if len(race_df) > 0:
    race_df = race_df.sort_values('mean_ed_ratio', ascending=True)
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=race_df['mean_ed_ratio'], y=race_df['OmbRace'],
                          orientation='h', marker_color='#E63946', name='ED比率'))
    fig6.add_trace(go.Bar(x=race_df['mean_readmit'], y=race_df['OmbRace'],
                          orientation='h', marker_color='#457B9D', name='再入院率'))
    fig6.update_layout(title="种族/民族差异 — ED比率与再入院率<br><sub>有种族信息子集(61.4%)</sub>",
                       barmode='group', template='plotly_white', height=450)
else:
    fig6 = go.Figure()

# 7. 强相关性对（排除trivial的）
corr_list = pd.DataFrame(data['strong_correlations'])
# 过滤掉显而易见的相关（同一 domain 的 SDOH 变量之间）
trivial_pairs = set()
sdoh_vars = ['sdoh_social_connections', 'sdoh_housing_stability', 'sdoh_stress',
             'sdoh_depression', 'sdoh_financial_resource_strain', 
             'sdoh_transportation_needs', 'sdoh_food_insecurity', 'sdoh_total_screenings']
for i, v1 in enumerate(sdoh_vars):
    for v2 in sdoh_vars[i+1:]:
        trivial_pairs.add((v1, v2))
        trivial_pairs.add((v2, v1))

# 也排除 encounters<->diagnoses 等明显的
trivial_pairs.update([
    ('total_encounters', 'ed_visits'), ('total_encounters', 'hospital_admissions'),
    ('total_encounters', 'unique_diagnoses'), ('total_encounters', 'encounters_per_month'),
    ('ed_visits', 'hospital_admissions'), ('ed_visits', 'unique_diagnoses')
])

if len(corr_list) > 0:
    novel = corr_list[~corr_list.apply(lambda r: (r['var1'], r['var2']) in trivial_pairs, axis=1)]
    novel = novel.sort_values('spearman_r', key=abs, ascending=False).head(15)
    
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(
        x=novel['spearman_r'], y=[f"{r['var1']} ↔ {r['var2']}" for _, r in novel.iterrows()],
        orientation='h',
        marker_color=['#E63946' if r > 0 else '#457B9D' for r in novel['spearman_r']],
        text=[f"ρ={r:.3f}" for r in novel['spearman_r']],
        textposition='outside'
    ))
    fig7.update_layout(title="非平凡的强相关对 (|Spearman ρ| > 0.3)<br><sub>排除同类SDOH变量和显而易见的encounter相关</sub>",
                       xaxis_title="Spearman ρ", template='plotly_white', height=500)
else:
    fig7 = go.Figure()

# ============================================================
# 合成 Full Dashboard HTML
# ============================================================
dashboard_html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>DataFest 2026 — 交叉分析 EDA Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #1d3557; }}
.header {{ text-align: center; padding: 30px 0; border-bottom: 3px solid #E63946; margin-bottom: 30px; }}
.header h1 {{ font-size: 2.2em; color: #1d3557; margin: 0; }}
.header p {{ color: #457B9D; font-size: 1.1em; margin-top: 8px; }}
.findings-summary {{ background: #fff; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.findings-summary h2 {{ color: #E63946; border-bottom: 2px solid #A8DADC; padding-bottom: 10px; }}
.finding-item {{ margin: 12px 0; padding: 12px 15px; border-left: 4px solid #2A9D8F; background: #f8f9fa; border-radius: 0 8px 8px 0; }}
.finding-item strong {{ color: #1d3557; }}
.finding-item .detail {{ color: #457B9D; font-size: 0.9em; margin-top: 4px; }}
.chart-section {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.chart-section h3 {{ color: #1d3557; margin-top: 0; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }}
@media (max-width: 1200px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
.kpi-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px; }}
.kpi-card {{ flex: 1; min-width: 180px; background: #fff; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.kpi-card .value {{ font-size: 2em; font-weight: bold; color: #E63946; }}
.kpi-card .label {{ color: #457B9D; font-size: 0.85em; margin-top: 5px; }}
</style>
</head>
<body>
<div class="header">
<h1>DataFest 2026 — 交叉分析 EDA Dashboard</h1>
<p>8大模块 × 10+交互式图表 × 全面统计检验 | 生成时间: 2026-05-01</p>
</div>

<div class="kpi-row">
<div class="kpi-card"><div class="value">18/18</div><div class="label">再入院显著预测因子<br>(Bonferroni校正)</div></div>
<div class="kpi-card"><div class="value">-63.6%</div><div class="label">MyChart激活者<br>ED比率更低</div></div>
<div class="kpi-card"><div class="value">8.7×</div><div class="label">周末ED率<br>vs 工作日</div></div>
<div class="kpi-card"><div class="value">ρ=0.219</div><div class="label">共病数↔ED比率<br>Spearman相关</div></div>
<div class="kpi-card"><div class="value">3.6×</div><div class="label">种族间ED比率<br>最大差距</div></div>
</div>

<div class="findings-summary">
<h2>核心新发现汇总</h2>
"""

for i, finding in enumerate(data['findings'], 1):
    dashboard_html += f"""
<div class="finding-item">
<strong>[{i}] {finding['module']}</strong>: {finding['finding']}
<div class="detail">{finding['detail']} | {finding['significance']}</div>
</div>
"""

dashboard_html += """
</div>

<div class="grid-2">
<div class="chart-section"><div id="chart1"></div></div>
<div class="chart-section"><div id="chart2"></div></div>
</div>
<div class="chart-section"><div id="chart3"></div></div>
<div class="grid-2">
<div class="chart-section"><div id="chart4"></div></div>
<div class="chart-section"><div id="chart5"></div></div>
</div>
<div class="chart-section"><div id="chart6"></div></div>
<div class="chart-section"><div id="chart7"></div></div>

<script>
"""

# 将 plotly 图表转换为 JSON 并嵌入
for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6, fig7], 1):
    fig_json = fig.to_json()
    dashboard_html += f"Plotly.newPlot('chart{i}', {fig_json});\n"

dashboard_html += """
</script>
</body>
</html>
"""

with open('/Users/aaronwen/Documents/Datafest/dashboard_eda_cross_analysis.html', 'w') as f:
    f.write(dashboard_html)

print("✓ Dashboard 已生成: dashboard_eda_cross_analysis.html")
