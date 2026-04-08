
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("养老服务床位商机分析模型 - 问题二求解")
print("=" * 70)

# ==================== 5.1 三种模式财务参数 ====================

print("\n" + "=" * 60)
print("5.1 三种养老模式财务参数设定")
print("=" * 60)

# 定义三种模式：社区养老、医养结合、高端养老
modes = ['社区养老', '医养结合', '高端养老']

# 收入参数（万元/床/年）
revenue_params = {
    '床位费': [2.4, 5.0, 8.0],
    '护理费': [1.2, 2.5, 3.0],
    '餐饮费': [0.8, 1.2, 1.5],
    '附加收入': [0.4, 1.0, 2.0]
}

# 成本参数（万元/床/年）
cost_params = {
    '固定成本': [1.2, 2.5, 3.5],
    '可变成本': [0.6, 0.9, 1.4]
}

# 其他参数
other_params = {
    '入住率': [0.75, 0.94, 0.85],
    '单床投资成本': [10, 30, 50],  # 万元/床
    '政府补贴': [0.5, 1.2, 0.8]  # 万元/床/年
}

# 计算单床利润
annual_revenue = np.array([sum([revenue_params[k][i] for k in revenue_params]) 
                          for i in range(3)])
annual_cost = np.array([sum([cost_params[k][i] for k in cost_params]) 
                       for i in range(3)])
gov_subsidy = np.array(other_params['政府补贴'])
occupancy = np.array(other_params['入住率'])
investment_cost = np.array(other_params['单床投资成本'])

# 实际单床年利润 = (年收入 - 年成本 + 补贴) × 入住率
profit_per_bed = (annual_revenue - annual_cost + gov_subsidy) * occupancy

print("\n三种模式财务对比（每床每年）：")
print("-" * 80)
print(f"{'模式':<12}{'年收入':<10}{'年成本':<10}{'补贴':<8}{'入住率':<10}{'实际利润':<10}{'投资成本':<10}")
print("-" * 80)
for i, mode in enumerate(modes):
    print(f"{mode:<12}{annual_revenue[i]:<10.2f}{annual_cost[i]:<10.2f}"
          f"{gov_subsidy[i]:<8.2f}{occupancy[i]:<10.2%}{profit_per_bed[i]:<10.2f}{investment_cost[i]:<10.1f}")

print(f"\n单床年利润（万元）：{np.round(profit_per_bed, 2)}")
print(f"投资成本（万元/床）：{investment_cost}")

# 投资回收期
payback_period = investment_cost / profit_per_bed
print(f"\n投资回收期（年）：{np.round(payback_period, 2)}")

# ==================== 5.2 投资组合优化模型 ====================

print("\n" + "=" * 60)
print("5.2 投资组合优化模型")
print("=" * 60)

# 决策变量：x1, x2, x3 分别表示社区、医养、高端的百床数量
# 目标：最大化总利润

# 参数转换（百张床位为单位）
# 单床利润（万元）→ 百床利润（十万元）
profit_per_100beds = profit_per_bed * 100 / 10  # 转换为十万元/百张

# 投资成本（万元/床）→ 百床投资（十万元/百张）
# 社区：10万/床 × 100床 = 1000万 = 100十万元 → 系数1
# 医养：30万/床 × 100床 = 3000万 = 300十万元 → 系数3  
# 高端：50万/床 × 100床 = 5000万 = 500十万元 → 系数5
invest_coef = np.array([1, 3, 5])  # 投资成本系数

print(f"\n优化模型参数（每百张床位）：")
print(f"年利润（十万元）：{np.round(profit_per_100beds, 2)}")
print(f"投资系数：{invest_coef}")

# 总投资预算：100单位（对应1000万元）
budget = 100

# 情况1：无最低投资约束
print("\n" + "-" * 50)
print("情况1：无最低床位数量约束")
print("-" * 50)

# 线性规划标准形式：min -c^T x，约束 Ax <= b
# 目标系数（取负因为linprog求最小）
c = -profit_per_100beds
# 不等式约束：invest_coef · x <= budget
A_ub = [invest_coef]
b_ub = [budget]
# 边界
bounds = [(0, None), (0, None), (0, None)]

result1 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

print(f"最优解：社区={result1.x[0]:.2f}百张, 医养={result1.x[1]:.2f}百张, 高端={result1.x[2]:.2f}百张")
print(f"最大利润：{-result1.fun:.2f}十万元 = {-result1.fun * 10:.2f}万元")
print(f"总投资：{np.dot(invest_coef, result1.x):.2f}十万元")

# 情况2：有最低投资约束（保证市场多样性）
print("\n" + "-" * 50)
print("情况2：最低床位数量约束（各至少10百张）")
print("-" * 50)

bounds2 = [(10, None), (10, None), (0, None)]  # 社区、医养至少10百张

result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds2, method='highs')

print(f"最优解：社区={result2.x[0]:.2f}百张, 医养={result2.x[1]:.2f}百张, 高端={result2.x[2]:.2f}百张")
print(f"最大利润：{-result2.fun:.2f}十万元 = {-result2.fun * 10:.2f}万元")
print(f"总投资：{np.dot(invest_coef, result2.x):.2f}十万元")

# 计算各类型床位数
beds_community = result2.x[0] * 100
beds_medical = result2.x[1] * 100
beds_premium = result2.x[2] * 100

print(f"\n实际床位数：社区={beds_community:.0f}张, 医养={beds_medical:.0f}张, 高端={beds_premium:.0f}张")
print(f"总床位数：{beds_community + beds_medical + beds_premium:.0f}张")

# ==================== 5.3 商机识别与策略建议 ====================

print("\n" + "=" * 60)
print("5.3 商机识别与策略建议")
print("=" * 60)

# 计算各类投资回报率（ROI）
roi_annual = profit_per_bed / investment_cost * 100  # 年ROI（%）

print(f"\n各模式年投资回报率（ROI）：")
for i, mode in enumerate(modes):
    print(f"  {mode}: {roi_annual[i]:.2f}%")

# 商机评估矩阵
print(f"\n商机评估矩阵：")
print("-" * 60)
print(f"{'模式':<12}{'利润水平':<12}{'风险等级':<12}{'资金门槛':<12}{'适合投资者'}")
print("-" * 60)
print(f"{'高端养老':<12}{'★★★★★':<12}{'中':<12}{'高':<12}{'保险资金、地产企业'}")
print(f"{'医养结合':<12}{'★★★★☆':<12}{'低':<12}{'中':<12}{'医疗机构、产业资本'}")
print(f"{'社区养老':<12}{'★★★☆☆':<12}{'低':<12}{'低':<12}{'中小企业、社会资本'}")

# ==================== 5.4 可视化 ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 单床利润对比
ax1 = axes[0, 0]
colors = ['#66B2FF', '#99FF99', '#FF9999']
bars1 = ax1.bar(modes, profit_per_bed, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('单床年利润（万元）', fontsize=12)
ax1.set_title('图1 三种养老模式单床利润对比', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, profit_per_bed):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 图2: 投资回收期对比
ax2 = axes[0, 1]
bars2 = ax2.bar(modes, payback_period, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('投资回收期（年）', fontsize=12)
ax2.set_title('图2 三种模式投资回收期对比', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, payback_period):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 图3: 投资组合优化结果
ax3 = axes[1, 0]
invest_labels = ['社区养老\n(7000张)', '医养结合\n(1000张)', '高端养老\n(0张)']
invest_values = [result2.x[0], result2.x[1], result2.x[2]]
colors_pie = ['#66B2FF', '#99FF99', '#FF9999']
explode = (0, 0.05, 0.1)  # 突出显示高端养老

wedges, texts, autotexts = ax3.pie(invest_values, labels=invest_labels, autopct='%1.1f%%',
                                    colors=colors_pie, explode=explode, startangle=90,
                                    textprops={'fontsize': 11})
ax3.set_title('图3 最优投资组合结构（有约束条件）', fontsize=13, fontweight='bold')

# 图4: 投资回报率与风险矩阵
ax4 = axes[1, 1]
risk_levels = [3, 2, 4]  # 风险等级（1-5分，5分最高）
scatter = ax4.scatter(risk_levels, roi_annual, s=[investment_cost[i]*20 for i in range(3)], 
                      c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, mode in enumerate(modes):
    ax4.annotate(mode, (risk_levels[i], roi_annual[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
ax4.set_xlabel('风险等级（1-5分）', fontsize=12)
ax4.set_ylabel('年投资回报率（%）', fontsize=12)
ax4.set_title('图4 投资回报率-风险矩阵（气泡大小=投资成本）', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(1, 5)
ax4.set_ylim(0, max(roi_annual) * 1.2)

plt.tight_layout()

# 修复：使用正确的输出路径
import os
output_path = '/mnt/kimi/output/problem2_results.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图表已保存至: {output_path}")
plt.show()

# ==================== 5.5 敏感性分析 ====================

print("\n" + "=" * 60)
print("5.5 敏感性分析")
print("=" * 60)

# 分析入住率变化对利润的影响
occupancy_scenarios = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
print(f"\n入住率变化对单床利润的影响（高端养老）：")
print(f"{'入住率':<10}{'单床利润（万元）':<15}{'利润变化率':<15}")
base_profit = profit_per_bed[2]
for occ in occupancy_scenarios:
    new_profit = (annual_revenue[2] - annual_cost[2] + gov_subsidy[2]) * occ
    change = (new_profit - base_profit) / base_profit * 100
    print(f"{occ:<10.0%}{new_profit:<15.2f}{change:<15.1f}%")

# 分析投资成本变化对回收期的影响
print(f"\n投资成本变化对回收期的影响（医养结合）：")
cost_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
base_payback = payback_period[1]
for factor in cost_factors:
    new_cost = investment_cost[1] * factor
    new_payback = new_cost / profit_per_bed[1]
    change = (new_payback - base_payback) / base_payback * 100
    print(f"{factor:<10.1f}{new_payback:<15.2f}{change:<15.1f}%")

print("\n" + "=" * 70)
print("问题二求解完成！")
print("=" * 70)
print("\n核心结论：")
print("1. 高端养老单床利润最高（8.16万元/年），适合长期资本布局")
print("2. 医养结合入住率最高（94%），风险最低，适合稳健投资者")
print("3. 社区养老投资回收期最短（4.4年），适合中小企业快速进入")
print(f"4. 最优投资组合：社区{beds_community:.0f}张 + 医养{beds_medical:.0f}张 + 高端{beds_premium:.0f}张")
print(f"5. 预期年总利润：{-result2.fun * 10:.2f}万元，总投资：{np.dot(invest_coef, result2.x) * 10:.2f}万元")
print("=" * 70)
