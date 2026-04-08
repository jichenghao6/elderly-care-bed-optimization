import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('问题四：政策建议效果评估与实施路径', fontsize=16, fontweight='bold')

# 图1: 政策建议优先级矩阵
ax1 = axes[0, 0]
policies = ['动态预测\n机制', '公建民营\n推广', '护理型床位\n建设', '社会资本\n激励', '智慧养老\n平台']
urgency = [9, 8.5, 9.5, 7.5, 8]      # 紧迫性
impact = [8.5, 9, 9.5, 8, 8.5]        # 影响力
cost = [3, 6, 8, 4, 7]                # 实施成本（气泡大小）

colors_policy = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#A8DADC']
scatter = ax1.scatter(urgency, impact, s=[c*50 for c in cost], c=colors_policy, alpha=0.7, edgecolors='black', linewidth=1.5)

for i, policy in enumerate(policies):
    ax1.annotate(policy, (urgency[i], impact[i]), textcoords="offset points", 
                xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('紧迫性评分', fontsize=11)
ax1.set_ylabel('影响力评分', fontsize=11)
ax1.set_title('(a) 政策建议优先级矩阵（气泡大小=实施成本）', fontsize=12, fontweight='bold')
ax1.set_xlim(6.5, 10)
ax1.set_ylim(7.5, 10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=8.5, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5)

# 添加象限标签
ax1.text(9.25, 9.75, '高紧迫性\n高影响力', ha='center', va='center', fontsize=9, style='italic', alpha=0.7)
ax1.text(7.25, 9.75, '低紧迫性\n高影响力', ha='center', va='center', fontsize=9, style='italic', alpha=0.7)

# 图2: 政策实施时间轴
ax2 = axes[0, 1]
years = ['2025', '2026', '2027', '2028', '2029', '2030']
policy_implementation = {
    '动态预测机制': [80, 95, 100, 100, 100, 100],
    '公建民营推广': [20, 45, 70, 85, 95, 100],
    '护理型床位建设': [30, 50, 70, 85, 95, 100],
    '社会资本激励': [40, 60, 75, 85, 90, 95],
    '智慧养老平台': [10, 30, 55, 75, 90, 100]
}

colors_time = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#A8DADC']
for i, (policy, values) in enumerate(policy_implementation.items()):
    ax2.plot(years, values, 'o-', color=colors_time[i], linewidth=2.5, markersize=6, label=policy)

ax2.set_xlabel('年份', fontsize=11)
ax2.set_ylabel('实施完成度(%)', fontsize=11)
ax2.set_title('(b) 政策建议实施路径规划', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

# 图3: 预期效果评估
ax3 = axes[1, 0]
categories = ['床位供给\n满足率', '财政投入\n效率', '区域公平\n指数', '就业带动\n效应', '服务质量\n满意度']
before = [72, 65, 68, 60, 70]   # 实施前
after = [95, 88, 85, 90, 92]    # 实施后

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, before, width, label='现状(2025)', color='#E63946', alpha=0.8)
bars2 = ax3.bar(x + width/2, after, width, label='目标(2030)', color='#2A9D8F', alpha=0.8)

ax3.set_ylabel('评分(分)', fontsize=11)
ax3.set_title('(c) 政策实施前后关键指标对比', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=9)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 105)

# 添加提升幅度标签
for i, (b, a) in enumerate(zip(before, after)):
    improvement = ((a - b) / b) * 100
    ax3.text(i, max(b, a) + 3, f'+{improvement:.0f}%', ha='center', fontsize=8, fontweight='bold', color='#2B2D42')

# 图4: 投资回报分析
ax4 = axes[1, 1]
investment_type = ['政府财政\n投入', '社会资本\n投入', '总投入']
roi_values = [3.2, 8.5, 5.8]  # 投资回报率
colors_roi = ['#E63946', '#2A9D8F', '#F4A261']

bars = ax4.bar(investment_type, roi_values, color=colors_roi, alpha=0.8, width=0.6)
ax4.set_ylabel('投资回报率(倍)', fontsize=11)
ax4.set_title('(d) 不同投资主体预期回报率(2025-2030)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, roi_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{val:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/kimi/output/figure8_policy_evaluation.png', dpi=300, bbox_inches='tight')
print("图8已保存至: /mnt/kimi/output/figure8_policy_evaluation.png")
plt.show()