"""
模型轻量化模块测试脚本
验证剪枝、量化、蒸馏等功能
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimization import (
    ModelProfiler,
    ModelPruner,
    ModelQuantizer,
    KnowledgeDistiller,
    ModelOptimizer
)


# 创建示例模型
class SimpleACCModel(nn.Module):
    """简单的ACC模型用于测试"""
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class LargeACCModel(nn.Module):
    """大型ACC模型（教师模型）"""
    def __init__(self, input_dim=5, hidden_dim=256, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def test_model_profiler():
    """测试模型分析器"""
    print("=" * 60)
    print("模型分析器测试")
    print("=" * 60)

    profiler = ModelProfiler()

    # 创建模型
    model = SimpleACCModel()

    # 分析模型
    model_info = profiler.profile_model(model, "SimpleACCModel")
    profiler.print_profile(model_info)

    # 测量推理时间
    input_shape = (1, 5)
    inference_stats = profiler.measure_inference_time(model, input_shape, num_runs=100)

    print("\n推理时间统计:")
    print(f"  平均时间: {inference_stats['mean_ms']:.3f} ms")
    print(f"  标准差: {inference_stats['std_ms']:.3f} ms")
    print(f"  最小时间: {inference_stats['min_ms']:.3f} ms")
    print(f"  最大时间: {inference_stats['max_ms']:.3f} ms")
    print(f"  FPS: {inference_stats['fps']:.1f}")

    return profiler, model_info


def test_model_pruning():
    """测试模型剪枝"""
    print("\n" + "=" * 60)
    print("模型剪枝测试")
    print("=" * 60)

    profiler = ModelProfiler()
    pruner = ModelPruner(pruning_ratio=0.3)

    # 创建模型
    original_model = SimpleACCModel()

    # 分析原始模型
    original_info = profiler.profile_model(original_model, "Original")
    print("\n原始模型:")
    profiler.print_profile(original_info)

    # 权重剪枝
    pruned_model = pruner.weight_pruning(original_model)

    # 分析剪枝模型
    pruning_stats = pruner.get_pruning_statistics(pruned_model)
    print("\n剪枝统计:")
    print(f"  总参数: {pruning_stats['total_params']:,}")
    print(f"  零参数: {pruning_stats['zero_params']:,}")
    print(f"  剩余参数: {pruning_stats['remaining_params']:,}")
    print(f"  稀疏度: {pruning_stats['sparsity_percent']:.2f}%")

    # 测量推理时间对比
    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(original_model, input_shape)
    pruned_time = profiler.measure_inference_time(pruned_model, input_shape)

    print("\n推理时间对比:")
    print(f"  原始模型: {original_time['mean_ms']:.3f} ms")
    print(f"  剪枝模型: {pruned_time['mean_ms']:.3f} ms")
    print(f"  加速比: {original_time['mean_ms'] / pruned_time['mean_ms']:.2f}x")

    return pruner, pruning_stats


def test_model_quantization():
    """测试模型量化"""
    print("\n" + "=" * 60)
    print("模型量化测试")
    print("=" * 60)

    profiler = ModelProfiler()
    quantizer = ModelQuantizer(quantization_type="dynamic")

    # 创建模型
    original_model = SimpleACCModel()

    # 分析原始模型
    original_info = profiler.profile_model(original_model, "Original")
    print("\n原始模型:")
    profiler.print_profile(original_info)

    # 动态量化
    quantized_model = quantizer.dynamic_quantization(original_model)

    # 分析量化模型
    quantized_info = profiler.profile_model(quantized_model, "Quantized")
    print("\n量化模型:")
    profiler.print_profile(quantized_info)

    # 计算压缩比
    compression_ratio = original_info.size_mb / max(quantized_info.size_mb, 0.001)
    size_reduction = (1 - quantized_info.size_mb / max(original_info.size_mb, 0.001)) * 100

    print("\n量化效果:")
    print(f"  原始大小: {original_info.size_mb:.2f} MB")
    print(f"  量化大小: {quantized_info.size_mb:.2f} MB")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  大小减少: {size_reduction:.2f}%")

    return quantizer, quantized_info


def test_knowledge_distillation():
    """测试知识蒸馏"""
    print("\n" + "=" * 60)
    print("知识蒸馏测试")
    print("=" * 60)

    profiler = ModelProfiler()
    distiller = KnowledgeDistiller(temperature=3.0, alpha=0.7)

    # 创建教师和学生模型
    teacher_model = LargeACCModel()
    student_model = SimpleACCModel()

    # 分析模型大小
    teacher_info = profiler.profile_model(teacher_model, "Teacher")
    student_info = profiler.profile_model(student_model, "Student")

    print("\n教师模型:")
    profiler.print_profile(teacher_info)
    print("\n学生模型:")
    profiler.print_profile(student_info)

    # 模拟蒸馏过程（简化版）
    print("\n模拟知识蒸馏过程...")

    # 创建模拟数据（回归问题）
    num_samples = 100
    train_data = torch.randn(num_samples, 5)
    train_labels = torch.randn(num_samples, 1)  # 回归标签

    # 模拟蒸馏训练
    for epoch in range(3):
        # 教师预测
        teacher_output = teacher_model(train_data)

        # 学生预测
        student_output = student_model(train_data)

        # 使用MSE损失模拟蒸馏
        soft_loss = torch.nn.functional.mse_loss(student_output, teacher_output)
        hard_loss = torch.nn.functional.mse_loss(student_output, train_labels)
        loss = distiller.alpha * soft_loss + (1 - distiller.alpha) * hard_loss

        print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

    print("\n蒸馏完成！")

    return distiller


def test_optimization_pipeline():
    """测试优化流水线"""
    print("\n" + "=" * 60)
    print("优化流水线测试")
    print("=" * 60)

    optimizer = ModelOptimizer()
    profiler = ModelProfiler()

    # 创建模型
    model = SimpleACCModel()

    # 优化配置
    config = {
        'pruning': True,
        'pruning_ratio': 0.3,
        'quantization': True,
        'quantization_type': 'dynamic'
    }

    print("\n优化配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 执行优化
    optimized_model, result = optimizer.optimize_pipeline(model, config)

    # 打印结果
    print("\n优化结果:")
    print(f"  原始大小: {result.original_size:.2f} MB")
    print(f"  优化大小: {result.optimized_size:.2f} MB")
    print(f"  压缩比: {result.compression_ratio:.2f}x")
    print(f"  大小减少: {(1 - result.optimized_size / result.original_size) * 100:.2f}%")

    return optimizer, result


def visualize_optimization_results():
    """可视化优化结果"""
    profiler = ModelProfiler()
    pruner = ModelPruner(pruning_ratio=0.3)
    quantizer = ModelQuantizer()

    # 创建模型
    original = SimpleACCModel()
    pruned = pruner.weight_pruning(SimpleACCModel())
    quantized = quantizer.dynamic_quantization(SimpleACCModel())

    # 分析模型
    original_info = profiler.profile_model(original, "Original")
    pruned_info = profiler.profile_model(pruned, "Pruned")
    quantized_info = profiler.profile_model(quantized, "Quantized")

    # 测量推理时间
    input_shape = (1, 5)
    original_time = profiler.measure_inference_time(original, input_shape)
    pruned_time = profiler.measure_inference_time(pruned, input_shape)
    quantized_time = profiler.measure_inference_time(quantized, input_shape)

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 模型大小对比
    ax1 = axes[0]
    models = ['Original', 'Pruned', 'Quantized']
    sizes = [original_info.size_mb, pruned_info.size_mb, quantized_info.size_mb]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars1 = ax1.bar(models, sizes, color=colors)
    ax1.set_ylabel('Size (MB)')
    ax1.set_title('Model Size Comparison')
    ax1.grid(True, alpha=0.3)

    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{size:.3f}',
                ha='center', va='bottom')

    # 参数量对比
    ax2 = axes[1]
    params = [original_info.parameters, pruned_info.parameters, quantized_info.parameters]

    bars2 = ax2.bar(models, params, color=colors)
    ax2.set_ylabel('Parameters')
    ax2.set_title('Parameter Count Comparison')
    ax2.grid(True, alpha=0.3)

    for bar, param in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 100,
                f'{param:,}',
                ha='center', va='bottom')

    # 推理时间对比
    ax3 = axes[2]
    times = [original_time['mean_ms'], pruned_time['mean_ms'], quantized_time['mean_ms']]

    bars3 = ax3.bar(models, times, color=colors)
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Time Comparison')
    ax3.grid(True, alpha=0.3)

    for bar, time in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{time:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150)
    print("\n可视化结果已保存到 optimization_comparison.png")
    plt.show()


def comprehensive_comparison():
    """综合对比测试"""
    print("\n" + "=" * 60)
    print("综合对比测试")
    print("=" * 60)

    profiler = ModelProfiler()

    # 创建不同大小的模型
    models = {
        'Small': SimpleACCModel(hidden_dim=32),
        'Medium': SimpleACCModel(hidden_dim=64),
        'Large': LargeACCModel(),
    }

    print("\n模型对比:")
    print("-" * 60)
    print(f"{'Model':<10} {'Params':<12} {'Size (MB)':<10} {'Layers':<8}")
    print("-" * 60)

    for name, model in models.items():
        info = profiler.profile_model(model, name)
        print(f"{name:<10} {info.parameters:<12,} {info.size_mb:<10.3f} {info.layers:<8}")

    # 推理时间对比
    input_shape = (1, 5)
    print("\n推理时间对比:")
    print("-" * 60)
    print(f"{'Model':<10} {'Time (ms)':<12} {'FPS':<10}")
    print("-" * 60)

    for name, model in models.items():
        stats = profiler.measure_inference_time(model, input_shape)
        print(f"{name:<10} {stats['mean_ms']:<12.3f} {stats['fps']:<10.1f}")


if __name__ == "__main__":
    # 运行所有测试
    test_model_profiler()
    test_model_pruning()
    test_model_quantization()
    test_knowledge_distillation()
    test_optimization_pipeline()

    # 可视化
    visualize_optimization_results()

    # 综合对比
    comprehensive_comparison()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)