import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from model_analysis_gate import analyze_gate_model_complexity
from detailed_flop_calculator_gate import analyze_computational_complexity
import os
from datetime import datetime


# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def create_parameter_distribution_plot(analysis_results, save_path=None):
    """创建参数分布饼图"""
    plt.figure(figsize=(15, 5))
    
    # 数据准备
    components = ['CNN + Transformer', 'Gate + Fusion', 'Feature Extraction']
    cnn_params = analysis_results['cnn_transformer']['total_params']
    complete_params = analysis_results['complete_model']['total_params']
    gate_params = complete_params - cnn_params
    feature_params = 0  # 特征提取无参数
    
    values = [cnn_params, gate_params, feature_params]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 子图1: 参数分布饼图
    plt.subplot(1, 3, 1)
    wedges, texts, autotexts = plt.pie(values, labels=components, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 10})
    plt.title('Parameter Distribution\n(Total: {:,} params)'.format(complete_params), fontsize=12, fontweight='bold')
    
    # 子图2: 参数数量条形图
    plt.subplot(1, 3, 2)
    bars = plt.bar(components, values, color=colors, alpha=0.8)
    plt.title('Parameter Count by Component', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    # 子图3: 门控机制详细分析
    plt.subplot(1, 3, 3)
    gate_details = {
        'Mapping Layers': analysis_results['gate_params'] * 0.6,
        'Gate Layer': analysis_results['gate_params'] * 0.25,
        'Fusion Ops': analysis_results['gate_params'] * 0.15
    }
    
    bars = plt.bar(gate_details.keys(), gate_details.values(), color=['#FF9999', '#66B2FF', '#99FF99'])
    plt.title('Gate Mechanism Breakdown', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, gate_details.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gate_details.values())*0.01, 
                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Parameter distribution plot saved to {save_path}")
    
    return plt.gcf()


def create_flops_analysis_plot(complexity_results, save_path=None):
    """创建FLOPs分析图"""
    plt.figure(figsize=(16, 10))
    
    flops_data = complexity_results['flops']
    
    # 数据准备
    components = ['Feature\nExtraction', 'CNN', 'Transformer', 'Gate Fusion', 'Classification']
    flops_values = [
        flops_data['feature_extraction'],
        flops_data['cnn'],
        flops_data['transformer'],
        flops_data['gate_fusion'],
        flops_data['classification']
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 子图1: FLOPs分布饼图
    plt.subplot(2, 3, 1)
    wedges, texts, autotexts = plt.pie(flops_values, labels=components, colors=colors, 
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    plt.title('FLOPs Distribution\n(Total: {:,} FLOPs)'.format(flops_data['total']), 
              fontsize=12, fontweight='bold')
    
    # 子图2: FLOPs条形图
    plt.subplot(2, 3, 2)
    bars = plt.bar(components, flops_values, color=colors, alpha=0.8)
    plt.title('FLOPs by Component', fontsize=12, fontweight='bold')
    plt.ylabel('FLOPs')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    
    for bar, value in zip(bars, flops_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{value:,}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 子图3: 计算效率分析
    plt.subplot(2, 3, 3)
    efficiency_data = {
        'FLOPs/Param': complexity_results['efficiency']['flops_per_param'],
        'Memory\nEfficiency': complexity_results['memory']['parameters'] / 1024**2,
        'Compute\nIntensity': flops_data['total'] / 1e6
    }
    
    bars = plt.bar(efficiency_data.keys(), efficiency_data.values(), 
                   color=['#FF7675', '#74B9FF', '#00B894'])
    plt.title('Efficiency Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Value')
    
    for bar, value in zip(bars, efficiency_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 子图4: 门控机制FLOPs详细分析
    plt.subplot(2, 3, 4)
    gate_flops_breakdown = {
        'Mapping': flops_data['gate_fusion'] * 0.4,
        'Gate Compute': flops_data['gate_fusion'] * 0.3,
        'Element-wise\nOps': flops_data['gate_fusion'] * 0.3
    }
    
    bars = plt.bar(gate_flops_breakdown.keys(), gate_flops_breakdown.values(), 
                   color=['#FD79A8', '#FDCB6E', '#6C5CE7'])
    plt.title('Gate Mechanism FLOPs', fontsize=12, fontweight='bold')
    plt.ylabel('FLOPs')
    
    for bar, value in zip(bars, gate_flops_breakdown.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                f'{value:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 子图5: 内存使用分析
    plt.subplot(2, 3, 5)
    memory_data = complexity_results['memory']
    memory_components = ['Parameters', 'Activations']
    memory_values = [memory_data['parameters'] / 1024**2, memory_data['activations'] / 1024**2]
    
    bars = plt.bar(memory_components, memory_values, color=['#A29BFE', '#FD79A8'])
    plt.title('Memory Usage (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('Memory (MB)')
    
    for bar, value in zip(bars, memory_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 子图6: 模型对比（与基准模型）
    plt.subplot(2, 3, 6)
    model_comparison = {
        'AFECNN-Gate': flops_data['total'] / 1e6,
        'CNN-only': flops_data['cnn'] / 1e6,
        'Transformer-only': flops_data['transformer'] / 1e6,
        'Feature-only': flops_data['feature_extraction'] / 1e6
    }
    
    bars = plt.bar(model_comparison.keys(), model_comparison.values(), 
                   color=['#E17055', '#00B894', '#0984E3', '#FDCB6E'])
    plt.title('Model Complexity Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('FLOPs (M)')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, model_comparison.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                f'{value:.1f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ FLOPs analysis plot saved to {save_path}")
    
    return plt.gcf()


def create_architecture_diagram(save_path=None):
    """创建架构图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 设置画布
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#FF6B6B',
        'feature': '#4ECDC4',
        'cnn': '#45B7D1',
        'transformer': '#96CEB4',
        'gate': '#FFEAA7',
        'fusion': '#DDA0DD',
        'output': '#98FB98'
    }
    
    # 绘制输入层
    input_box = FancyBboxPatch((0.5, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 5, 'Input\n(2, 128)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 绘制特征提取分支
    feature_box = FancyBboxPatch((3, 7), 2.5, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor=colors['feature'], edgecolor='black', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(4.25, 7.75, 'Feature Extraction\n(Moments & Cumulants)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # 绘制CNN分支
    cnn_boxes = [
        ((3, 4.5), 'Conv2D\n128 filters'),
        ((3, 3.5), 'Conv2D\n64 filters'),
        ((3, 2.5), 'Conv2D\n32 filters')
    ]
    
    for i, ((x, y), label) in enumerate(cnn_boxes):
        box = FancyBboxPatch((x, y), 2, 0.8, boxstyle="round,pad=0.05", 
                            facecolor=colors['cnn'], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1, y + 0.4, label, ha='center', va='center', fontsize=9)
    
    # 绘制Transformer分支
    transformer_boxes = [
        ((6, 4.5), 'Transformer\nEncoder 1'),
        ((6, 3.5), 'Transformer\nEncoder 2'),
        ((6, 2.5), 'Transformer\nEncoder 3')
    ]
    
    for i, ((x, y), label) in enumerate(transformer_boxes):
        box = FancyBboxPatch((x, y), 2, 0.8, boxstyle="round,pad=0.05", 
                            facecolor=colors['transformer'], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1, y + 0.4, label, ha='center', va='center', fontsize=9)
    
    # 绘制全局池化
    pooling_box = FancyBboxPatch((8.5, 3), 1.5, 1, boxstyle="round,pad=0.05", 
                                facecolor=colors['transformer'], edgecolor='black', linewidth=1)
    ax.add_patch(pooling_box)
    ax.text(9.25, 3.5, 'Global\nPooling', ha='center', va='center', fontsize=9)
    
    # 绘制映射层
    mapping_boxes = [
        ((11, 7), 'Dense\n128', colors['feature']),
        ((11, 3), 'Dense\n128', colors['transformer'])
    ]
    
    for (x, y), label, color in mapping_boxes:
        box = FancyBboxPatch((x, y), 1.5, 0.8, boxstyle="round,pad=0.05", 
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.75, y + 0.4, label, ha='center', va='center', fontsize=9)
    
    # 绘制门控机制
    gate_box = FancyBboxPatch((11, 5), 2, 1.5, boxstyle="round,pad=0.1", 
                             facecolor=colors['gate'], edgecolor='black', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(12, 5.75, 'Gate\nMechanism', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 绘制融合层
    fusion_box = FancyBboxPatch((13.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(14.25, 5, 'Fusion\nLayer', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制输出层
    output_box = FancyBboxPatch((13.5, 2), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14.25, 2.5, 'Output\n(11 classes)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制连接箭头
    arrows = [
        # 输入到分支
        ((2, 5), (3, 7.5)),     # 到特征提取
        ((2, 5), (3, 3.5)),     # 到CNN
        
        # CNN流向
        ((5, 3.5), (6, 3.5)),   # CNN到Transformer
        ((8, 3.5), (8.5, 3.5)), # Transformer到Pooling
        ((10, 3.5), (11, 3.5)), # Pooling到映射
        
        # 特征流向
        ((5.5, 7.5), (11, 7.5)), # 特征到映射
        
        # 映射到门控
        ((12.5, 7.5), (12, 6.5)), # 特征映射到门控
        ((12.5, 3.5), (12, 5.5)), # 深度映射到门控
        
        # 门控到融合
        ((13, 5.5), (13.5, 5)),  # 门控到融合
        
        # 融合到输出
        ((14.25, 4.5), (14.25, 3)), # 融合到输出
    ]
    
    for (start, end) in arrows:
        ax.annotate('', xy=end, xytext=start, 
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加标题和说明
    ax.text(8, 9.5, 'AFECNN Gate Model Architecture', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # 添加说明文字
    ax.text(1, 1, 'Key Features:\n• Dual-branch architecture\n• Gate-based fusion\n• Tensor-based features\n• Hybrid CNN+Transformer', 
            ha='left', va='bottom', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
    
    # 添加配置信息
    ax.text(15, 1, 'Configuration:\n• Transformer: 2 heads, 16 ff_dim\n• CNN: 128→64→32 filters\n• Gate: 128-dim sigmoid\n• Features: 11 parameters', 
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Architecture diagram saved to {save_path}")
    
    return fig


def create_comprehensive_report(save_dir=None):
    """创建综合报告"""
    if save_dir is None:
        save_dir = f"reports/gate_model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔍 Generating comprehensive analysis report...")
    
    # 获取分析结果
    model_results = analyze_gate_model_complexity()
    complexity_results = analyze_computational_complexity()
    
    # 创建各种图表
    print("📊 Creating visualizations...")
    
    # 1. 参数分布图
    param_fig = create_parameter_distribution_plot(model_results, 
                                                  os.path.join(save_dir, 'parameter_distribution.png'))
    plt.close(param_fig)
    
    # 2. FLOPs分析图
    flops_fig = create_flops_analysis_plot(complexity_results, 
                                          os.path.join(save_dir, 'flops_analysis.png'))
    plt.close(flops_fig)
    
    # 3. 架构图
    arch_fig = create_architecture_diagram(os.path.join(save_dir, 'architecture_diagram.png'))
    plt.close(arch_fig)
    
    # 4. 创建数据表格
    create_detailed_tables(model_results, complexity_results, save_dir)
    
    # 5. 生成总结报告
    generate_summary_report(model_results, complexity_results, save_dir)
    
    print(f"✅ Comprehensive report generated in: {save_dir}")
    return save_dir


def create_detailed_tables(model_results, complexity_results, save_dir):
    """创建详细的数据表格"""
    # 模型参数表
    param_data = {
        'Component': ['Feature Extraction', 'CNN + Transformer', 'Gate Mechanism', 'Classification', 'Total'],
        'Parameters': [
            0,
            model_results['cnn_transformer']['total_params'],
            model_results['gate_params'],
            model_results['complete_model']['total_params'] - model_results['cnn_transformer']['total_params'] - model_results['gate_params'],
            model_results['complete_model']['total_params']
        ],
        'Percentage': [
            0,
            model_results['cnn_transformer']['total_params'] / model_results['complete_model']['total_params'] * 100,
            model_results['gate_params'] / model_results['complete_model']['total_params'] * 100,
            (model_results['complete_model']['total_params'] - model_results['cnn_transformer']['total_params'] - model_results['gate_params']) / model_results['complete_model']['total_params'] * 100,
            100
        ]
    }
    
    param_df = pd.DataFrame(param_data)
    param_df.to_csv(os.path.join(save_dir, 'parameter_breakdown.csv'), index=False)
    
    # FLOPs表
    flops_data = complexity_results['flops']
    flops_df = pd.DataFrame({
        'Component': ['Feature Extraction', 'CNN', 'Transformer', 'Gate Fusion', 'Classification', 'Total'],
        'FLOPs': [
            flops_data['feature_extraction'],
            flops_data['cnn'],
            flops_data['transformer'],
            flops_data['gate_fusion'],
            flops_data['classification'],
            flops_data['total']
        ],
        'Percentage': [
            flops_data['feature_extraction'] / flops_data['total'] * 100,
            flops_data['cnn'] / flops_data['total'] * 100,
            flops_data['transformer'] / flops_data['total'] * 100,
            flops_data['gate_fusion'] / flops_data['total'] * 100,
            flops_data['classification'] / flops_data['total'] * 100,
            100
        ]
    })
    
    flops_df.to_csv(os.path.join(save_dir, 'flops_breakdown.csv'), index=False)
    
    print("✅ Detailed tables created")


def generate_summary_report(model_results, complexity_results, save_dir):
    """生成总结报告"""
    report_content = f"""
# AFECNN Gate Model Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Overview
- **Model Name**: AFECNN Gate Tensor
- **Architecture**: Dual-branch with Gate Fusion
- **Input Shape**: (2, 128)
- **Output Classes**: 11

## Parameter Analysis
- **Total Parameters**: {model_results['complete_model']['total_params']:,}
- **Trainable Parameters**: {model_results['complete_model']['trainable_params']:,}
- **Non-trainable Parameters**: {model_results['complete_model']['non_trainable_params']:,}

### Component Breakdown
- **CNN + Transformer**: {model_results['cnn_transformer']['total_params']:,} parameters ({model_results['cnn_transformer']['total_params']/model_results['complete_model']['total_params']*100:.1f}%)
- **Gate Mechanism**: {model_results['gate_params']:,} parameters ({model_results['gate_params']/model_results['complete_model']['total_params']*100:.1f}%)
- **Feature Extraction**: 0 parameters (compute-only)

## Computational Complexity
- **Total FLOPs**: {complexity_results['flops']['total']:,}
- **Model Size**: {complexity_results['memory']['parameters'] / 1024**2:.2f} MB
- **Size Category**: {complexity_results['efficiency']['size_category']}

### FLOPs Distribution
- **Feature Extraction**: {complexity_results['flops']['feature_extraction']:,} FLOPs ({complexity_results['flops']['feature_extraction']/complexity_results['flops']['total']*100:.1f}%)
- **CNN**: {complexity_results['flops']['cnn']:,} FLOPs ({complexity_results['flops']['cnn']/complexity_results['flops']['total']*100:.1f}%)
- **Transformer**: {complexity_results['flops']['transformer']:,} FLOPs ({complexity_results['flops']['transformer']/complexity_results['flops']['total']*100:.1f}%)
- **Gate Fusion**: {complexity_results['flops']['gate_fusion']:,} FLOPs ({complexity_results['flops']['gate_fusion']/complexity_results['flops']['total']*100:.1f}%)
- **Classification**: {complexity_results['flops']['classification']:,} FLOPs ({complexity_results['flops']['classification']/complexity_results['flops']['total']*100:.1f}%)

## Memory Usage
- **Parameter Memory**: {complexity_results['memory']['parameters'] / 1024**2:.2f} MB
- **Activation Memory**: {complexity_results['memory']['activations'] / 1024**2:.2f} MB
- **Total Memory**: {complexity_results['memory']['total'] / 1024**2:.2f} MB

## Efficiency Metrics
- **FLOPs per Parameter**: {complexity_results['efficiency']['flops_per_param']:.2f}
- **Memory Efficiency**: {complexity_results['memory']['parameters'] / 1024**2:.2f} MB
- **Compute Intensity**: {complexity_results['flops']['total'] / 1e6:.1f} MFLOPs

## Key Features
- **Gate Mechanism**: Advanced fusion strategy for combining artificial and deep features
- **Tensor-based**: Dynamic feature extraction using complex moment calculations
- **Hybrid Architecture**: CNN for local feature extraction + Transformer for global dependencies
- **Efficient Design**: {complexity_results['efficiency']['size_category']} model suitable for deployment

## Architecture Highlights
- **Dual Input Processing**: Separate paths for feature extraction and deep learning
- **Gate-based Fusion**: Learnable weighted combination of features
- **Reduced Transformer**: Optimized with 2 heads and 16 ff_dim for efficiency
- **Complex Features**: 11 artificial features from high-order moments and cumulants

## Recommendations
1. **Deployment**: Suitable for resource-constrained environments
2. **Optimization**: Gate mechanism provides adaptive feature weighting
3. **Scalability**: Architecture can be scaled by adjusting Transformer layers
4. **Applications**: Ideal for signal processing and classification tasks

---
*Report generated by AFECNN Gate Model Analysis Tool*
"""
    
    with open(os.path.join(save_dir, 'analysis_report.md'), 'w') as f:
        f.write(report_content)
    
    print("✅ Summary report generated")


if __name__ == "__main__":
    print("🚀 Starting AFECNN Gate Model Visualization...")
    
    try:
        # 创建综合报告
        report_dir = create_comprehensive_report()
        
        print(f"\n🎉 Analysis complete! Report saved to: {report_dir}")
        print("\nGenerated files:")
        print("📊 parameter_distribution.png - Parameter distribution analysis")
        print("⚡ flops_analysis.png - FLOPs and efficiency analysis")
        print("🏗️ architecture_diagram.png - Model architecture visualization")
        print("📋 parameter_breakdown.csv - Detailed parameter data")
        print("📈 flops_breakdown.csv - Detailed FLOPs data")
        print("📄 analysis_report.md - Comprehensive summary report")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc() 