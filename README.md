# diagnostics-ai
diagnostics+ai实例：预测糖尿病紧张情况
白盒模型
主要是用了线性回归的模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import seaborn as sns

# 设置中文字体（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_linear_regression_weights_plot():
    """
    生成线性回归模型权重可视化图（类似图2.4）
    展示特征权重及其对预测的影响方向
    """
    # 生成模拟数据 - 使用糖尿病数据集类似的特征
    feature_names = ['BMI', '血压', '血糖', '胆固醇', '年龄', '性别', '胰岛素', '皮肤厚度']

    # 创建模拟数据集
    X, y = make_regression(n_samples=1000, n_features=len(feature_names),
                           noise=0.1, random_state=42)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 获取特征权重（系数）
    weights = model.coef_
    intercept = model.intercept_

    # 创建可视化图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 图1：特征权重条形图
    colors = ['red' if w < 0 else 'blue' for w in weights]
    bars = ax1.barh(feature_names, weights, color=colors, alpha=0.7)
    ax1.set_xlabel('特征权重', fontsize=12)
    ax1.set_title('线性回归模型特征权重分析', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 在条形上添加数值标签
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width >= 0 else width - 0.01
        ax1.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                 f'{weight:.3f}', ha='left' if width >= 0 else 'right',
                 va='center', fontsize=10)

    # 图2：权重绝对值排序（特征重要性）
    feature_importance = np.abs(weights)
    sorted_idx = np.argsort(feature_importance)[::-1]

    ax2.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx],
             color='green', alpha=0.7)
    ax2.set_xlabel('特征重要性（权重绝对值）', fontsize=12)
    ax2.set_title('特征重要性排序', fontsize=14, fontweight='bold')

    # 添加网格使图表更易读
    ax1.grid(axis='x', alpha=0.3)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('linear_regression_weights.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印模型统计信息
    print("=" * 50)
    print("线性回归模型权重分析报告")
    print("=" * 50)
    print(f"截距项 (bias): {intercept:.4f}")
    print("\n各特征权重:")
    for name, weight in zip(feature_names, weights):
        direction = "正向影响" if weight > 0 else "负向影响"
        print(f"{name}: {weight:.4f} ({direction})")

    print(f"\n最重要的特征: {feature_names[sorted_idx[0]]} (权重绝对值: {feature_importance[sorted_idx[0]]:.4f})")

    return weights, feature_names


def create_simple_weights_plot():
    """
    创建简化的权重图（更接近传统论文风格）
    """
    # 简化版特征名称
    simple_features = ['特征A', '特征B', '特征C', '特征D', '特征E']

    # 模拟权重数据
    np.random.seed(42)
    weights = np.random.uniform(-1, 1, len(simple_features))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(simple_features, weights, color=['red' if w < 0 else 'blue' for w in weights])

    plt.xlabel('特征名称', fontsize=12)
    plt.ylabel('权重值', fontsize=12)
    plt.title('线性回归模型特征权重', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 添加数值标签
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{weight:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('simple_weights_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    return weights


if __name__ == "__main__":
    print("开始生成线性回归权重可视化图...")

    # 生成详细的可视化
    weights, features = generate_linear_regression_weights_plot()

    print("\n" + "=" * 50)
    print("简化版权重图生成中...")
    print("=" * 50)

    # 生成简化版
    simple_weights = create_simple_weights_plot()

    print("可视化完成！图片已保存为 'linear_regression_weights.png' 和 'simple_weights_plot.png'")
