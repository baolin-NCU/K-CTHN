import torch
from thop import profile
from AFECNN_tensor import AFECNN
from torch.nn.utils import prune
import numpy as np

def apply_pruning(model, pruning_rate=0.5):
    # 对卷积层和全连接层进行L1非结构化剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, 'weight')  # 永久移除被剪枝的权重
            
    return model

def count_nonzero_parameters(model):
    return sum([torch.sum(p != 0) for p in model.parameters()])

def main():
    # 使用更真实的输入尺寸 (batch_size, channels, signal_length)
    input_shape = (2, 128)  # 双通道IQ信号，128个采样点
    
    # 创建并加载剪枝后的模型
    model = AFECNN(num_classes=11)
    
    # 加载剪枝后的权重（确保路径正确）
    pruned_weights = torch.load('IQ_and_accumulation_4/pruned/IQ_and_accumulation_4_pruned.wts.h5')
    model.load_state_dict(pruned_weights)
    
    # 应用结构化剪枝
    model = apply_pruning(model, pruning_rate=0.5)
    
    # 生成随机输入
    input_tensor = torch.randn(1, *input_shape)
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    
    # 打印结果
    print(f"Model FLOPs: {flops/1e6:.2f}M")
    print(f"Trainable Parameters: {params/1e6:.2f}M")

if __name__ == "__main__":
    main()
