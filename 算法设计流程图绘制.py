import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 颜色方案
colors = {
    'input': '#E8F4F8',      # 浅蓝 - 输入
    'process': '#B8E0D2',    # 薄荷绿 - 处理
    'model': '#D4C5E9',      # 淡紫 - 模型
    'output': '#F8D7DA',     # 浅粉 - 输出
    'decision': '#FCE7CD',   # 浅橙 - 决策
    'arrow': '#2B2D42'       # 深灰 - 箭头
}

# ==================== 标题 ====================
ax.text(8, 11.5, '问题四：养老服务床位规划关键算法体系', 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#2B2D42')

# ==================== 第一层：数据输入 ====================
# 数据输入框
input_box = FancyBboxPatch((1, 9.5), 3, 1.2, boxstyle="round,pad=0.05", 
                           facecolor=colors['input'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(input_box)
ax.text(2.5, 10.1, '历史数据输入', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(2.5, 9.75, '人口数据 | 经济数据 | 养老数据', ha='center', va='center', fontsize=9)

# ==================== 第二层：预测模块 ====================
# GM(1,1)模型
gm_box = FancyBboxPatch((0.5, 7.5), 3.5, 1.5, boxstyle="round,pad=0.05",
                        facecolor=colors['model'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(gm_box)
ax.text(2.25, 8.6, 'GM(1,1)灰色预测', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2.25, 8.25, 'x^(0)(k)+az^(1)(k)=b', ha='center', va='center', fontsize=9, family='monospace')
ax.text(2.25, 7.95, 'x̂^(1)(k+1)=(x^(0)(1)-b/a)e^(-ak)+b/a', ha='center', va='center', fontsize=8, family='monospace')

# Logistic模型
log_box = FancyBboxPatch((4.5, 7.5), 3, 1.5, boxstyle="round,pad=0.05",
                         facecolor=colors['model'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(log_box)
ax.text(6, 8.6, 'Logistic模型', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(6, 8.25, 'P(t)=K/(1+e^(-r(t-t0)))', ha='center', va='center', fontsize=9, family='monospace')
ax.text(6, 7.95, '对比验证预测精度', ha='center', va='center', fontsize=9)

# 多元回归
reg_box = FancyBboxPatch((8.5, 7.5), 3, 1.5, boxstyle="round,pad=0.05",
                         facecolor=colors['model'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(reg_box)
ax.text(10, 8.6, '多元回归模型', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(10, 8.25, 'α=β₀+β₁GDP+β₂U+β₃A', ha='center', va='center', fontsize=9, family='monospace')
ax.text(10, 7.95, '估计入住率变化趋势', ha='center', va='center', fontsize=9)

# 马尔可夫链
markov_box = FancyBboxPatch((12.5, 7.5), 3, 1.5, boxstyle="round,pad=0.05",
                            facecolor=colors['model'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(markov_box)
ax.text(14, 8.6, '马尔可夫链模型', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(14, 8.25, 'λ(t+1)=λ(t)·P', ha='center', va='center', fontsize=9, family='monospace')
ax.text(14, 7.95, '分解养老结构比例', ha='center', va='center', fontsize=9)

# 预测结果汇总
pred_result = FancyBboxPatch((5.5, 5.8), 5, 1.2, boxstyle="round,pad=0.05",
                             facecolor=colors['output'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(pred_result)
ax.text(8, 6.4, '2025-2030年养老床位需求预测结果', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8, 6.05, '总需求: 1030-1150万张 | 机构养老增速: 14%/年', ha='center', va='center', fontsize=9)

# ==================== 第三层：优化决策模块 ====================
# 企业优化
corp_box = FancyBboxPatch((1, 3.8), 5, 1.5, boxstyle="round,pad=0.05",
                          facecolor=colors['decision'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(corp_box)
ax.text(3.5, 4.9, '企业视角：线性规划模型', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(3.5, 4.55, 'max Π=cᵀx  s.t. Ax≤b, x≥0', ha='center', va='center', fontsize=9, family='monospace')
ax.text(3.5, 4.2, '单纯形法求解 | 高端养老ROI最优', ha='center', va='center', fontsize=9)

# 政府优化
gov_box = FancyBboxPatch((7.5, 3.8), 7.5, 1.5, boxstyle="round,pad=0.05",
                         facecolor=colors['decision'], edgecolor='#2B2D42', linewidth=2)
ax.add_patch(gov_box)
ax.text(11.25, 4.9, '政府视角：多目标优化模型', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(11.25, 4.55, 'min Z=w₁G+w₂E+w₃I-w₄J', ha='center', va='center', fontsize=9, family='monospace')
ax.text(11.25, 4.2, '差分进化算法 | 供需缺口↓ 财政可持续 区域公平↑ 就业促进↑', ha='center', va='center', fontsize=9)

# ==================== 第四层：政策输出 ====================
policy_box = FancyBboxPatch((3, 1.5), 10, 1.5, boxstyle="round,pad=0.05",
                            facecolor=colors['output'], edgecolor='#2B2D42', linewidth=3)
ax.add_patch(policy_box)
ax.text(8, 2.55, '政策建议输出', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(8, 2.15, '①动态预测机制 ②公建民营推广 ③护理型床位建设 ④社会资本激励 ⑤智慧养老平台', 
        ha='center', va='center', fontsize=10)

# ==================== 绘制连接箭头 ====================
arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2, mutation_scale=15)

# 输入到预测模型
ax.annotate('', xy=(2.25, 9), xytext=(2.5, 9.5), arrowprops=arrow_style)
ax.annotate('', xy=(6, 9), xytext=(2.5, 9.5), arrowprops=arrow_style)
ax.annotate('', xy=(10, 9), xytext=(2.5, 9.5), arrowprops=arrow_style)
ax.annotate('', xy=(14, 9), xytext=(2.5, 9.5), arrowprops=arrow_style)

# 预测模型到结果
ax.annotate('', xy=(6.5, 7), xytext=(2.25, 7.5), arrowprops=arrow_style)
ax.annotate('', xy=(7.5, 7), xytext=(6, 7.5), arrowprops=arrow_style)
ax.annotate('', xy=(8.5, 7), xytext=(10, 7.5), arrowprops=arrow_style)
ax.annotate('', xy=(9.5, 7), xytext=(14, 7.5), arrowprops=arrow_style)

# 预测结果到优化
ax.annotate('', xy=(3.5, 5.3), xytext=(6.5, 5.8), arrowprops=arrow_style)
ax.annotate('', xy=(11.25, 5.3), xytext=(9.5, 5.8), arrowprops=arrow_style)

# 优化到政策
ax.annotate('', xy=(6, 3), xytext=(3.5, 3.8), arrowprops=arrow_style)
ax.annotate('', xy=(10, 3), xytext=(11.25, 3.8), arrowprops=arrow_style)

# ==================== 添加算法特征标注 ====================
# 精度检验标注
ax.text(2.25, 7.2, 'C<0.35,P>0.95(一级精度)', ha='center', va='center', 
        fontsize=8, style='italic', color='#6B7280', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 稳态标注
ax.text(14, 7.2, '稳态:88%-8%-4%', ha='center', va='center',
        fontsize=8, style='italic', color='#6B7280', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 算法复杂度标注
ax.text(3.5, 3.5, 'O(n³)', ha='center', va='center',
        fontsize=8, style='italic', color='#6B7280', family='monospace')
ax.text(11.25, 3.5, 'O(NP·D·T_max)', ha='center', va='center',
        fontsize=8, style='italic', color='#6B7280', family='monospace')

plt.tight_layout()
plt.savefig('/mnt/kimi/output/figure7_algorithm_framework.png', dpi=300, bbox_inches='tight')
print("图7已保存至: /mnt/kimi/output/figure7_algorithm_framework.png")
plt.show()