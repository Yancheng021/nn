"""
模型轻量化模块 - Model Optimization Module
实现模型压缩、量化、剪枝等轻量化技术
用于减小模型体积、加速推理、降低内存占用
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import os


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    parameters: int
    size_mb: float
    layers: int
    trainable_params: int


@dataclass
class OptimizationResult:
    """优化结果"""
    original_size: float
    optimized_size: float
    compression_ratio: float
    speedup: float
    accuracy_loss: float
    method: str


class ModelProfiler:
    """
    模型性能分析器
    分析模型的参数量、大小、推理速度等
    """

    def __init__(self):
        """初始化模型分析器"""
        self.results: Dict[str, Any] = {}

    def profile_model(self, model: nn.Module, model_name: str = "model") -> ModelInfo:
        """
        分析模型信息

        Args:
            model: PyTorch模型
            model_name: 模型名称

        Returns:
            ModelInfo: 模型信息
        """
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 计算模型大小（MB）
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024 * 1024)

        # 计算层数
        num_layers = len(list(model.modules()))

        return ModelInfo(
            name=model_name,
            parameters=total_params,
            size_mb=total_size,
            layers=num_layers,
            trainable_params=trainable_params
        )

    def measure_inference_time(self,
                               model: nn.Module,
                               input_shape: Tuple[int, ...],
                               num_runs: int = 100,
                               device: str = "cpu") -> Dict[str, float]:
        """
        测量推理时间

        Args:
            model: PyTorch模型
            input_shape: 输入形状
            num_runs: 运行次数
            device: 设备类型

        Returns:
            推理时间统计
        """
        model = model.to(device)
        model.eval()

        # 创建随机输入
        dummy_input = torch.randn(*input_shape).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 测量时间
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                times.append(end - start)

        # 统计
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            'mean_ms': mean_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000,
            'fps': 1.0 / mean_time
        }

    def compare_models(self,
                      original: nn.Module,
                      optimized: nn.Module,
                      input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        对比原始模型和优化模型

        Args:
            original: 原始模型
            optimized: 优化模型
            input_shape: 输入形状

        Returns:
            对比结果
        """
        # 分析模型信息
        original_info = self.profile_model(original, "original")
        optimized_info = self.profile_model(optimized, "optimized")

        # 测量推理时间
        original_time = self.measure_inference_time(original, input_shape)
        optimized_time = self.measure_inference_time(optimized, input_shape)

        # 计算改进
        size_reduction = (1 - optimized_info.size_mb / original_info.size_mb) * 100
        speedup = original_time['mean_ms'] / optimized_time['mean_ms']

        return {
            'original': {
                'params': original_info.parameters,
                'size_mb': original_info.size_mb,
                'inference_ms': original_time['mean_ms']
            },
            'optimized': {
                'params': optimized_info.parameters,
                'size_mb': optimized_info.size_mb,
                'inference_ms': optimized_time['mean_ms']
            },
            'improvement': {
                'size_reduction_percent': size_reduction,
                'speedup': speedup,
                'params_reduction': original_info.parameters - optimized_info.parameters
            }
        }

    def print_profile(self, model_info: ModelInfo):
        """打印模型信息"""
        print("\n" + "=" * 60)
        print(f"模型分析: {model_info.name}")
        print("=" * 60)
        print(f"总参数量: {model_info.parameters:,}")
        print(f"可训练参数: {model_info.trainable_params:,}")
        print(f"模型大小: {model_info.size_mb:.2f} MB")
        print(f"层数: {model_info.layers}")
        print("=" * 60)


class ModelPruner:
    """
    模型剪枝器
    实现权重剪枝和结构化剪枝
    """

    def __init__(self, pruning_ratio: float = 0.3):
        """
        初始化剪枝器

        Args:
            pruning_ratio: 剪枝比例（0-1）
        """
        self.pruning_ratio = pruning_ratio

    def weight_pruning(self,
                       model: nn.Module,
                       threshold: Optional[float] = None) -> nn.Module:
        """
        权重剪枝（非结构化剪枝）
        将小于阈值的权重置零

        Args:
            model: PyTorch模型
            threshold: 剪枝阈值，如果为None则根据比例计算

        Returns:
            剪枝后的模型
        """
        # 创建模型副本
        pruned_model = model

        for name, param in pruned_model.named_parameters():
            if param.dim() > 1:  # 只剪枝权重，不剪枝bias
                # 计算阈值
                if threshold is None:
                    abs_weights = torch.abs(param.data)
                    sorted_weights = torch.sort(abs_weights.view(-1))[0]
                    threshold_index = int(len(sorted_weights) * self.pruning_ratio)
                    threshold = sorted_weights[threshold_index].item()

                # 剪枝
                mask = torch.abs(param.data) > threshold
                param.data.mul_(mask.float())

        return pruned_model

    def structured_pruning(self,
                          model: nn.Module,
                          layer_pruning_ratios: Optional[Dict[str, float]] = None) -> nn.Module:
        """
        结构化剪枝
        剪枝整个神经元或卷积核

        Args:
            model: PyTorch模型
            layer_pruning_ratios: 各层的剪枝比例

        Returns:
            剪枝后的模型
        """
        if layer_pruning_ratios is None:
            layer_pruning_ratios = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                ratio = layer_pruning_ratios.get(name, self.pruning_ratio)
                # 剪枝神经元
                num_neurons = module.out_features
                num_to_prune = int(num_neurons * ratio)

                # 计算神经元重要性（基于权重绝对值之和）
                importance = torch.sum(torch.abs(module.weight.data), dim=1)
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]

                # 置零
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0

            elif isinstance(module, nn.Conv2d):
                ratio = layer_pruning_ratios.get(name, self.pruning_ratio)
                # 剪枝卷积核
                num_filters = module.out_channels
                num_to_prune = int(num_filters * ratio)

                # 计算卷积核重要性
                importance = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))
                _, indices = torch.sort(importance)
                prune_indices = indices[:num_to_prune]

                # 置零
                module.weight.data[prune_indices] = 0

        return model

    def get_pruning_statistics(self, model: nn.Module) -> Dict[str, float]:
        """
        获取剪枝统计信息

        Args:
            model: PyTorch模型

        Returns:
            统计信息
        """
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        sparsity = zero_params / total_params * 100

        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'remaining_params': total_params - zero_params,
            'sparsity_percent': sparsity
        }


class ModelQuantizer:
    """
    模型量化器
    实现动态量化和静态量化
    """

    def __init__(self, quantization_type: str = "dynamic"):
        """
        初始化量化器

        Args:
            quantization_type: 量化类型 ('dynamic', 'static')
        """
        self.quantization_type = quantization_type

    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """
        动态量化
        在推理时动态量化权重和激活

        Args:
            model: PyTorch模型

        Returns:
            量化后的模型
        """
        # 应用动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model

    def static_quantization(self,
                           model: nn.Module,
                           calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        静态量化
        需要校准数据来确定量化参数

        Args:
            model: PyTorch模型
            calibration_data: 校准数据

        Returns:
            量化后的模型
        """
        # 准备量化
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # 校准
        if calibration_data is not None:
            with torch.no_grad():
                model(calibration_data)

        # 转换为量化模型
        torch.quantization.convert(model, inplace=True)

        return model

    def quantize_model(self,
                      model: nn.Module,
                      calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        量化模型

        Args:
            model: PyTorch模型
            calibration_data: 校准数据（仅用于静态量化）

        Returns:
            量化后的模型
        """
        if self.quantization_type == "dynamic":
            return self.dynamic_quantization(model)
        elif self.quantization_type == "static":
            return self.static_quantization(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")


class KnowledgeDistiller:
    """
    知识蒸馏器
    从大模型（教师）向小模型（学生）转移知识
    """

    def __init__(self,
                 temperature: float = 3.0,
                 alpha: float = 0.7):
        """
        初始化知识蒸馏器

        Args:
            temperature: 温度参数，用于软化输出
            alpha: 软标签损失的权重
        """
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self,
                         student_output: torch.Tensor,
                         teacher_output: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """
        计算蒸馏损失

        Args:
            student_output: 学生模型输出
            teacher_output: 教师模型输出
            labels: 真实标签

        Returns:
            蒸馏损失
        """
        # 软标签损失（KL散度）
        soft_teacher = torch.softmax(teacher_output / self.temperature, dim=1)
        soft_student = torch.log_softmax(student_output / self.temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(
            soft_student, soft_teacher, reduction='batchmean'
        ) * (self.temperature ** 2)

        # 硬标签损失
        hard_loss = torch.nn.functional.cross_entropy(student_output, labels)

        # 综合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def distill(self,
               teacher_model: nn.Module,
               student_model: nn.Module,
               train_loader: Any,
               epochs: int = 10,
               learning_rate: float = 0.001) -> nn.Module:
        """
        执行知识蒸馏

        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            train_loader: 训练数据加载器
            epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            训练后的学生模型
        """
        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (data, labels) in enumerate(train_loader):
                # 教师模型预测
                with torch.no_grad():
                    teacher_output = teacher_model(data)

                # 学生模型预测
                student_output = student_model(data)

                # 计算蒸馏损失
                loss = self.distillation_loss(student_output, teacher_output, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return student_model


class ModelOptimizer:
    """
    模型优化器
    综合使用剪枝、量化、蒸馏等技术优化模型
    """

    def __init__(self):
        """初始化模型优化器"""
        self.profiler = ModelProfiler()
        self.pruner = ModelPruner()
        self.quantizer = ModelQuantizer()
        self.distiller = KnowledgeDistiller()

    def optimize_pipeline(self,
                         model: nn.Module,
                         optimization_config: Dict[str, Any]) -> Tuple[nn.Module, OptimizationResult]:
        """
        优化流水线

        Args:
            model: PyTorch模型
            optimization_config: 优化配置

        Returns:
            (优化后的模型, 优化结果)
        """
        original_info = self.profiler.profile_model(model, "original")

        optimized_model = model

        # 1. 剪枝
        if optimization_config.get('pruning', False):
            pruning_ratio = optimization_config.get('pruning_ratio', 0.3)
            self.pruner.pruning_ratio = pruning_ratio
            optimized_model = self.pruner.weight_pruning(optimized_model)
            print(f"剪枝完成，比例: {pruning_ratio}")

        # 2. 量化
        if optimization_config.get('quantization', False):
            quantization_type = optimization_config.get('quantization_type', 'dynamic')
            self.quantizer.quantization_type = quantization_type
            optimized_model = self.quantizer.quantize_model(optimized_model)
            print(f"量化完成，类型: {quantization_type}")

        optimized_info = self.profiler.profile_model(optimized_model, "optimized")

        # 计算优化结果
        compression_ratio = original_info.size_mb / max(optimized_info.size_mb, 0.001)
        size_reduction = (1 - optimized_info.size_mb / max(original_info.size_mb, 0.001)) * 100

        result = OptimizationResult(
            original_size=original_info.size_mb,
            optimized_size=optimized_info.size_mb,
            compression_ratio=compression_ratio,
            speedup=1.0,  # 需要实际测量
            accuracy_loss=0.0,  # 需要实际评估
            method=str(optimization_config)
        )

        return optimized_model, result

    def save_optimized_model(self,
                            model: nn.Module,
                            path: str,
                            format: str = "pytorch"):
        """
        保存优化后的模型

        Args:
            model: PyTorch模型
            path: 保存路径
            format: 保存格式 ('pytorch', 'onnx', 'torchscript')
        """
        if format == "pytorch":
            torch.save(model.state_dict(), path)
        elif format == "torchscript":
            scripted_model = torch.jit.script(model)
            scripted_model.save(path)
        elif format == "onnx":
            # 需要输入示例
            dummy_input = torch.randn(1, 5)  # ACC模型的输入维度
            torch.onnx.export(model, dummy_input, path)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"模型已保存到: {path}")