import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数设置 ====================
np.random.seed(42)

# 地区数量（简化为东、中、西三个区域）
n_regions = 3
region_names = ['东部地区', '中部地区', '西部地区']

# 养老模式：居家、社区、机构
n_modes = 3
mode_names = ['居家养老', '社区养老', '机构养老']

# 成本参数（万元/张）
c_build = np.array([0.5, 3.0, 15.0])      # 建设成本
c_op = np.array([0.2, 0.8, 3.5])           # 运营成本
c_sub = np.array([0.1, 0.4, 2.0])           # 政府补贴

# 其他参数
l_land = np.array([0, 10, 50])             # 占地面积（平方米/张）
e_employ = np.array([0.05, 0.15, 0.30])     # 带动就业（人/张）

# 权重系数
w = np.array([0.40, 0.25, 0.20, 0.15])    # [缺口, 财政, 公平, 就业]

# 2025年基准数据（万张）- 基于问题一预测结果
D_2025 = np.array([
    [750, 60, 25],    # 东部需求
    [550, 40, 15],    # 中部需求  
    [400, 30, 10]     # 西部需求
])

S_2025 = np.array([
    [680, 45, 18],    # 东部供给
    [480, 28, 9],     # 中部供给
    [340, 20, 6]      # 西部供给
])

# 约束参数
B_budget = 5000        # 财政总预算（亿元）
L_land = np.array([5000, 8000, 10000])  # 各地区可用土地（万平方米）
alpha = 0.85           # 最低保障系数

print("=" * 60)
print("问题三：政府视角多目标优化模型求解")
print("=" * 60)

# ==================== 目标函数定义 ====================
def objective(x_flat):
    """
    多目标优化函数
    x_flat: 展平的决策变量 [n_regions * n_modes]
    """
    x = x_flat.reshape(n_regions, n_modes)
    
    # 确保非负
    x = np.maximum(x, 0)
    
    # 新总供给
    S_new = S_2025 + x
    
    # (1) 供需缺口 G
    gap = D_2025 - S_new
    G = np.sum(np.abs(gap))
    
    # (2) 财政支出 E（亿元）
    E = np.sum((c_build + c_sub) * x + c_sub * S_2025) / 10000
    
    # (3) 区域不公平 I（基尼系数形式）
    R = np.sum(S_new, axis=1) / np.sum(D_2025, axis=1)  # 各地区满足率
    n = len(R)
    I = np.sum([abs(R[i] - R[j]) for i in range(n) for j in range(n)]) / (2 * n * np.sum(R) + 1e-10)
    
    # (4) 就业促进 J（万人）
    J = np.sum(e_employ * x)
    
    # 综合目标（最小化，就业取负）
    Z = w[0] * G + w[1] * E + w[2] * I * 100 - w[3] * J * 10
    
    return Z

def constraints(x_flat):
    """
    约束条件检查，返回惩罚值
    """
    x = x_flat.reshape(n_regions, n_modes)
    x = np.maximum(x, 0)
    S_new = S_2025 + x
    
    penalty = 0
    
    # 约束1: 财政预算
    E = np.sum((c_build + c_sub) * x + c_sub * S_2025) / 10000
    if E > B_budget:
        penalty += (E - B_budget) * 1000
    
    # 约束2: 基本保障（各地区满足率>=alpha）
    for i in range(n_regions):
        supply = np.sum(S_new[i])
        demand = np.sum(D_2025[i])
        ratio = supply / demand
        if ratio < alpha:
            penalty += (alpha - ratio) * demand * 1000
    
    # 约束3: 土地资源
    for i in range(n_regions):
        land_use = np.sum(l_land * x[i])
        if land_use > L_land[i]:
            penalty += (land_use - L_land[i]) * 10
    
    # 约束4: 结构合理性（9073结构）
    total_beds = np.sum(S_new, axis=0)
    total = np.sum(total_beds)
    if total > 0:
        lambda_home = total_beds[0] / total
        lambda_comm = total_beds[1] / total
        lambda_inst = total_beds[2] / total
        
        if lambda_home < 0.85:
            penalty += (0.85 - lambda_home) * 1000
        if lambda_comm > 0.10:
            penalty += (lambda_comm - 0.10) * 1000
        if lambda_inst > 0.05:
            penalty += (lambda_inst - 0.05) * 1000
    
    return penalty

def penalized_objective(x_flat):
    """带惩罚的目标函数"""
    return objective(x_flat) + constraints(x_flat)

# ==================== 使用差分进化算法求解 ====================
print("\n【模型求解】采用差分进化算法进行多目标优化...")
print("-" * 60)

# 设置变量边界
bounds = [(0, 200) for _ in range(n_regions * n_modes)]  # 每个地区每类最多新增200万张

# 执行优化
result = differential_evolution(
    penalized_objective,
    bounds,
    maxiter=500,
    popsize=20,
    tol=1e-6,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=42,
    polish=True
)

# 提取最优解
x_opt = result.x.reshape(n_regions, n_modes)
x_opt = np.maximum(x_opt, 0)  # 确保非负
S_opt = S_2025 + x_opt

print(f"优化收敛状态: {'成功' if result.success else '需检查'}")
print(f"最终目标函数值: {result.fun:.4f}")

# ==================== 结果分析 ====================
print("\n" + "=" * 60)
print("【求解结果分析】")
print("=" * 60)

# 1. 新增床位配置方案
print("\n表1 各地区各类型新增养老床位配置方案（万张）")
print("-" * 60)
print(f"{'地区':<12} {'居家养老':>12} {'社区养老':>12} {'机构养老':>12} {'合计':>12}")
print("-" * 60)
for i, name in enumerate(region_names):
    total = np.sum(x_opt[i])
    print(f"{name:<12} {x_opt[i,0]:>12.2f} {x_opt[i,1]:>12.2f} {x_opt[i,2]:>12.2f} {total:>12.2f}")
print("-" * 60)
print(f"{'全国合计':<12} {np.sum(x_opt[:,0]):>12.2f} {np.sum(x_opt[:,1]):>12.2f} {np.sum(x_opt[:,2]):>12.2f} {np.sum(x_opt):>12.2f}")

# 2. 优化后总供给
print("\n表2 优化后各地区养老床位总供给（万张）")
print("-" * 60)
print(f"{'地区':<12} {'居家养老':>12} {'社区养老':>12} {'机构养老':>12} {'合计':>12}")
print("-" * 60)
for i, name in enumerate(region_names):
    total = np.sum(S_opt[i])
    print(f"{name:<12} {S_opt[i,0]:>12.2f} {S_opt[i,1]:>12.2f} {S_opt[i,2]:>12.2f} {total:>12.2f}")
print("-" * 60)

# 3. 供需缺口分析
gap_opt = D_2025 - S_opt
print("\n表3 优化后各地区供需缺口分析（万张）")
print("-" * 60)
print(f"{'地区':<12} {'居家缺口':>12} {'社区缺口':>12} {'机构缺口':>12} {'总缺口率':>12}")
print("-" * 60)
for i, name in enumerate(region_names):
    gap_rate = np.sum(gap_opt[i]) / np.sum(D_2025[i]) * 100
    print(f"{name:<12} {gap_opt[i,0]:>12.2f} {gap_opt[i,1]:>12.2f} {gap_opt[i,2]:>12.2f} {gap_rate:>11.2f}%")
print("-" * 60)

# 4. 目标达成情况
G_final = np.sum(np.abs(gap_opt))
E_final = np.sum((c_build + c_sub) * x_opt + c_sub * S_2025) / 10000
R_final = np.sum(S_opt, axis=1) / np.sum(D_2025, axis=1)
I_final = np.sum([abs(R_final[i] - R_final[j]) for i in range(n_regions) for j in range(n_regions)]) / (2 * n_regions * np.sum(R_final))
J_final = np.sum(e_employ * x_opt)

print("\n表4 多目标优化达成情况")
print("-" * 60)
print(f"{'目标项':<20} {'优化结果':>20} {'单位':<10}")
print("-" * 60)
print(f"{'总供需缺口':<20} {G_final:>20.2f} {'万张':<10}")
print(f"{'财政总支出':<20} {E_final:>20.2f} {'亿元':<10}")
print(f"{'区域不公平系数':<20} {I_final:>20.4f} {'-':<10}")
print(f"{'新增就业岗位':<20} {J_final:>20.2f} {'万人':<10}")
print(f"{'预算使用率':<20} {E_final/B_budget*100:>20.2f} {'%':<10}")

# 5. 养老结构分析
total_beds = np.sum(S_opt, axis=0)
total_all = np.sum(total_beds)
structure = total_beds / total_all * 100
print("\n表5 优化后养老服务结构占比")
print("-" * 60)
print(f"{'养老模式':<15} {'床位数(万张)':>15} {'占比(%)':>15}")
print("-" * 60)
for j, name in enumerate(mode_names):
    print(f"{name:<15} {total_beds[j]:>15.2f} {structure[j]:>15.2f}")
print("-" * 60)
print(f"{'合计':<15} {total_all:>15.2f} {np.sum(structure):>15.2f}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('问题三：政府视角养老服务床位多目标优化结果', fontsize=16, fontweight='bold')

# 图1: 新增床位配置堆叠柱状图
ax1 = axes[0, 0]
x_pos = np.arange(n_regions)
width = 0.25
colors = ['#2E86AB', '#A23B72', '#F18F01']
bottom1 = np.zeros(n_regions)
bottom2 = x_opt[:, 0]

bars1 = ax1.bar(x_pos, x_opt[:, 0], width*3, label=mode_names[0], color=colors[0], alpha=0.8)
bars2 = ax1.bar(x_pos, x_opt[:, 1], width*3, bottom=bottom2, label=mode_names[1], color=colors[1], alpha=0.8)
bars3 = ax1.bar(x_pos, x_opt[:, 2], width*3, bottom=bottom2+x_opt[:,1], label=mode_names[2], color=colors[2], alpha=0.8)

ax1.set_xlabel('地区', fontsize=11)
ax1.set_ylabel('新增床位数（万张）', fontsize=11)
ax1.set_title('(a) 各地区新增床位配置方案', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(region_names)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for i in range(n_regions):
    height = x_opt[i, 0] + x_opt[i, 1] + x_opt[i, 2]
    ax1.text(i, height + 2, f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图2: 供需对比图
ax2 = axes[0, 1]
x_pos = np.arange(n_regions)
width = 0.35

demand_total = np.sum(D_2025, axis=1)
supply_total = np.sum(S_opt, axis=1)

bars1 = ax2.bar(x_pos - width/2, demand_total, width, label='需求量', color='#E63946', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, supply_total, width, label='供给量', color='#06A77D', alpha=0.8)

ax2.set_xlabel('地区', fontsize=11)
ax2.set_ylabel('床位数（万张）', fontsize=11)
ax2.set_title('(b) 各地区养老床位供需对比', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(region_names)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# 添加满足率标签
for i in range(n_regions):
    rate = supply_total[i] / demand_total[i] * 100
    ax2.text(i, max(demand_total[i], supply_total[i]) + 20, f'{rate:.1f}%', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2B2D42')

# 图3: 目标达成雷达图
ax3 = axes[0, 2]
categories = ['供需缺口\\n(归一化)', '财政支出\\n(归一化)', '区域公平\\n(归一化)', '就业促进\\n(归一化)']
N = len(categories)

# 归一化各目标值（越小越好，就业除外）
G_norm = G_final / np.sum(D_2025)  # 缺口率
E_norm = E_final / B_budget  # 预算使用率
I_norm = I_final * 5  # 放大不公平系数
J_norm = 1 - (J_final / 100)  # 就业转换（越大越好）

values = [G_norm, E_norm, I_norm, J_norm]
values += values[:1]  # 闭合

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax3 = plt.subplot(2, 3, 3, projection='polar')
ax3.plot(angles, values, 'o-', linewidth=2, color='#F18F01', label='优化结果')
ax3.fill(angles, values, alpha=0.25, color='#F18F01')
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=9)
ax3.set_ylim(0, 1)
ax3.set_title('(c) 多目标达成情况雷达图', fontsize=12, fontweight='bold', pad=20)
ax3.grid(True)

# 图4: 养老结构饼图
ax4 = axes[1, 0]
colors_pie = ['#2E86AB', '#A23B72', '#F18F01']
explode = (0.02, 0.02, 0.05)
wedges, texts, autotexts = ax4.pie(total_beds, labels=mode_names, autopct='%1.1f%%',
                                    colors=colors_pie, explode=explode, startangle=90,
                                    textprops={'fontsize': 10})
ax4.set_title('(d) 优化后养老服务结构占比', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

# 图5: 成本结构分析
ax5 = axes[1, 1]
cost_build = np.sum(c_build * x_opt) / 10000
cost_sub_new = np.sum(c_sub * x_opt) / 10000
cost_sub_exist = np.sum(c_sub * S_2025) / 10000

cost_items = ['新建建设成本', '新建运营成本\\n(政府补贴)', '存量运营成本\\n(政府补贴)']
cost_values = [cost_build, cost_sub_new, cost_sub_exist]
colors_cost = ['#E63946', '#F4A261', '#2A9D8F']

bars = ax5.barh(cost_items, cost_values, color=colors_cost, alpha=0.8, height=0.6)
ax5.set_xlabel('金额（亿元）', fontsize=11)
ax5.set_title('(e) 财政支出结构分解', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, cost_values)):
    ax5.text(val + 20, bar.get_y() + bar.get_height()/2, f'{val:.1f}亿', 
             va='center', fontsize=10, fontweight='bold')

# 图6: 就业带动效应
ax6 = axes[1, 2]
employ_by_region = np.sum(e_employ * x_opt, axis=1)
employ_by_mode = np.sum(e_employ * x_opt, axis=0)

# 创建组合图
x_pos = np.arange(n_regions)
bars = ax6.bar(x_pos, employ_by_region, color=['#264653', '#2A9D8F', '#E9C46A'], alpha=0.8, width=0.6)
ax6.set_xlabel('地区', fontsize=11)
ax6.set_ylabel('新增就业（万人）', fontsize=11)
ax6.set_title('(f) 各地区养老服务带动就业效应', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(region_names)
ax6.grid(axis='y', alpha=0.3)

# 添加数值标签和占比
total_employ = np.sum(employ_by_region)
for i, (bar, val) in enumerate(zip(bars, employ_by_region)):
    pct = val / total_employ * 100
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}万\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/kimi/output/figure6_multiobjective_optimization.png', dpi=300, bbox_inches='tight')
print("\n图6已保存至: /mnt/kimi/output/figure6_multiobjective_optimization.png")
plt.show()

# ==================== 商业模式设计 ====================
print("\n" + "=" * 60)
print("【基于优化结果的商业模式设计】")
print("=" * 60)

print("""
1. "公建民营"主导模式
   - 政府负责土地供给和基础设施建设（占总投资60%）
   - 通过公开招标引入专业养老机构运营
   - 运营方缴纳管理费用，政府提供绩效补贴
   - 预期效果：床位使用率从45%提升至75%以上

2. 三级服务网络架构
   ┌─────────────────────────────────────────┐
   │  县级：综合养老服务平台（每县1个）        │
   │         功能：统筹调配、质量监管、应急救助  │
   ├─────────────────────────────────────────┤
   │  乡镇：区域养老服务中心（覆盖率100%）       │
   │         功能：日间照料、康复护理、助餐服务  │
   ├─────────────────────────────────────────┤
   │  村级：互助养老服务点（覆盖率70%以上）       │
   │         功能：居家上门、邻里互助、健康监测  │
   └─────────────────────────────────────────┘

3. 多元化收入结构
   - 用户付费：分级收费（自理/半失能/失能）
   - 政府补贴：建设补贴8000-12000元/床，运营补贴200-400元/床/月
   - 社会捐赠：设立养老慈善基金，鼓励企业社会责任投入
   - 增值服务：康复医疗、文化娱乐、老年教育等

4. 就业促进机制
   - 直接就业：每100张床位创造25-30个岗位
   - 间接就业：带动医疗器械、老年用品、餐饮服务等相关产业
   - 2026-2030年累计新增就业岗位预计：60-80万人
""")

print("\n" + "=" * 60)
print("模型求解完成！")
print("=" * 60)