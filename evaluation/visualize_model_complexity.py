import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from detailed_flop_calculator import comprehensive_model_analysis

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def create_complexity_visualizations():
    """
    创建模型复杂度可视化图表
    """
    # 获取分析结果
    results = comprehensive_model_analysis()
    
    # 创建一个大图，包含多个子图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 参数量分布饼图
    ax1 = plt.subplot(2, 3, 1)
    params_data = [
        results['cnn_params'],
        results['transformer_params'],
        results['fusion_params']
    ]
    params_labels = ['CNN Branch', 'Transformer Branch', 'Fusion Layer']
    colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax1.pie(params_data, labels=params_labels, colors=colors1, 
                                      autopct='%1.1f%%', startangle=90)
    ax1.set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    
    # 2. FLOPs分布饼图
    ax2 = plt.subplot(2, 3, 2)
    flops_data = [
        results['feature_flops'],
        results['cnn_flops'],
        results['transformer_flops'],
        results['fusion_flops']
    ]
    flops_labels = ['Feature Extraction', 'CNN Branch', 'Transformer Branch', 'Fusion Layer']
    colors2 = ['#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A3FF']
    
    wedges2, texts2, autotexts2 = ax2.pie(flops_data, labels=flops_labels, colors=colors2, 
                                         autopct='%1.1f%%', startangle=90)
    ax2.set_title('FLOPs Distribution', fontsize=14, fontweight='bold')
    
    # 3. 参数量和FLOPs对比柱状图
    ax3 = plt.subplot(2, 3, 3)
    components = ['Feature\nExtraction', 'CNN\nBranch', 'Transformer\nBranch', 'Fusion\nLayer']
    params_values = [0, results['cnn_params'], results['transformer_params'], results['fusion_params']]
    flops_values = [results['feature_flops'], results['cnn_flops'], 
                   results['transformer_flops'], results['fusion_flops']]
    
    x = np.arange(len(components))
    width = 0.35
    
    # 归一化处理以便对比
    params_normalized = np.array(params_values) / max(params_values) * 100
    flops_normalized = np.array(flops_values) / max(flops_values) * 100
    
    bars1 = ax3.bar(x - width/2, params_normalized, width, label='Parameters (%)', color='#FF6B6B', alpha=0.8)
    bars2 = ax3.bar(x + width/2, flops_normalized, width, label='FLOPs (%)', color='#4ECDC4', alpha=0.8)
    
    ax3.set_xlabel('Model Components')
    ax3.set_ylabel('Normalized Percentage (%)')
    ax3.set_title('Parameters vs FLOPs Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 详细的计算量分解
    ax4 = plt.subplot(2, 3, 4)
    
    # 创建堆叠柱状图显示每个分支的详细计算量
    branch_names = ['Feature\nExtraction', 'CNN\nBranch', 'Transformer\nBranch', 'Fusion\nLayer']
    
    # 估算各分支内部的计算分解
    feature_breakdown = [results['feature_flops']]
    cnn_breakdown = [results['cnn_flops'] * 0.7, results['cnn_flops'] * 0.3]  # 卷积层 vs BatchNorm
    transformer_breakdown = [results['transformer_flops'] * 0.6, results['transformer_flops'] * 0.3, 
                           results['transformer_flops'] * 0.1]  # 注意力 vs FFN vs LayerNorm
    fusion_breakdown = [results['fusion_flops'] * 0.95, results['fusion_flops'] * 0.05]  # Dense vs Softmax
    
    # 创建堆叠柱状图
    categories = ['Feature\nExtraction', 'CNN\nBranch', 'Transformer\nBranch', 'Fusion\nLayer']
    
    # 为了显示，我们需要将数据转换为相同长度的数组
    max_len = max(len(feature_breakdown), len(cnn_breakdown), len(transformer_breakdown), len(fusion_breakdown))
    
    data_matrix = np.zeros((len(categories), max_len))
    data_matrix[0, :len(feature_breakdown)] = feature_breakdown
    data_matrix[1, :len(cnn_breakdown)] = cnn_breakdown
    data_matrix[2, :len(transformer_breakdown)] = transformer_breakdown
    data_matrix[3, :len(fusion_breakdown)] = fusion_breakdown
    
    # 绘制堆叠柱状图
    bottom = np.zeros(len(categories))
    colors_stack = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    for i in range(max_len):
        ax4.bar(categories, data_matrix[:, i], bottom=bottom, color=colors_stack[i], alpha=0.8)
        bottom += data_matrix[:, i]
    
    ax4.set_ylabel('FLOPs')
    ax4.set_title('FLOPs Breakdown by Component', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 内存使用分析
    ax5 = plt.subplot(2, 3, 5)
    
    # 内存使用数据
    input_memory = 2 * 128 * 4  # bytes
    feature_memory = 11 * 4
    cnn_memory = 17 * 32 * 4
    fusion_memory = 256 * 4
    param_memory = results['total_params'] * 4
    
    memory_categories = ['Input\nData', 'Feature\nOutput', 'CNN\nOutput', 'Fusion\nBuffer', 'Parameters']
    memory_values = [input_memory, feature_memory, cnn_memory, fusion_memory, param_memory]
    memory_values_mb = [x / 1024 / 1024 for x in memory_values]
    
    bars = ax5.bar(memory_categories, memory_values_mb, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax5.set_ylabel('Memory Usage (MB)')
    ax5.set_title('Memory Usage Analysis', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}MB', ha='center', va='bottom', fontsize=9)
    
    # 6. 效率对比图
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算效率指标
    efficiency_data = {
        'Component': ['Feature\nExtraction', 'CNN\nBranch', 'Transformer\nBranch', 'Fusion\nLayer'],
        'Params_per_FLOP': [
            0 if results['feature_flops'] == 0 else 0,
            results['cnn_params'] / results['cnn_flops'] if results['cnn_flops'] > 0 else 0,
            results['transformer_params'] / results['transformer_flops'] if results['transformer_flops'] > 0 else 0,
            results['fusion_params'] / results['fusion_flops'] if results['fusion_flops'] > 0 else 0
        ],
        'Output_Efficiency': [11, 32, 32, 11]  # 输出特征维度
    }
    
    # 创建双轴图
    ax6_twin = ax6.twinx()
    
    # 绘制参数效率
    bars1 = ax6.bar(efficiency_data['Component'], efficiency_data['Params_per_FLOP'], 
                   alpha=0.7, color='#FF6B6B', label='Params/FLOP')
    ax6.set_ylabel('Parameters per FLOP', color='#FF6B6B')
    ax6.tick_params(axis='y', labelcolor='#FF6B6B')
    
    # 绘制输出效率
    line1 = ax6_twin.plot(efficiency_data['Component'], efficiency_data['Output_Efficiency'], 
                         'o-', color='#4ECDC4', linewidth=2, markersize=8, label='Output Dimension')
    ax6_twin.set_ylabel('Output Dimension', color='#4ECDC4')
    ax6_twin.tick_params(axis='y', labelcolor='#4ECDC4')
    
    ax6.set_title('Component Efficiency Analysis', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建详细的数据表格
    create_detailed_table(results)
    
    return results

def create_detailed_table(results):
    """
    创建详细的数据表格
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = [
        ['Component', 'Parameters', 'FLOPs', 'Params %', 'FLOPs %', 'Output Dim'],
        ['Feature Extraction', '0', f'{results["feature_flops"]:,}', '0.0%', 
         f'{results["feature_flops"]/results["total_flops"]*100:.1f}%', '11'],
        ['CNN Branch', f'{results["cnn_params"]:,}', f'{results["cnn_flops"]:,}', 
         f'{results["cnn_params"]/results["total_params"]*100:.1f}%', 
         f'{results["cnn_flops"]/results["total_flops"]*100:.1f}%', '32'],
        ['Transformer Branch', f'{results["transformer_params"]:,}', f'{results["transformer_flops"]:,}', 
         f'{results["transformer_params"]/results["total_params"]*100:.1f}%', 
         f'{results["transformer_flops"]/results["total_flops"]*100:.1f}%', '32'],
        ['Fusion Layer', f'{results["fusion_params"]:,}', f'{results["fusion_flops"]:,}', 
         f'{results["fusion_params"]/results["total_params"]*100:.1f}%', 
         f'{results["fusion_flops"]/results["total_flops"]*100:.1f}%', '11'],
        ['Total', f'{results["total_params"]:,}', f'{results["total_flops"]:,}', 
         '100.0%', '100.0%', '11']
    ]
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    table[(0, 0)].set_facecolor('#4ECDC4')
    table[(0, 1)].set_facecolor('#4ECDC4')
    table[(0, 2)].set_facecolor('#4ECDC4')
    table[(0, 3)].set_facecolor('#4ECDC4')
    table[(0, 4)].set_facecolor('#4ECDC4')
    table[(0, 5)].set_facecolor('#4ECDC4')
    
    for i in range(1, len(table_data)):
        if i == len(table_data) - 1:  # 最后一行（总计）
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#FFE4B5')
        elif i % 2 == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#F0F8FF')
    
    plt.title('AFECNN Model Complexity Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('model_complexity_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_architecture_diagram():
    """
    创建模型架构图
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 定义组件位置和尺寸
    components = {
        'Input': {'pos': (1, 5), 'size': (1.5, 1), 'color': '#FFE4B5'},
        'Feature_Extraction': {'pos': (3.5, 7), 'size': (2, 1.5), 'color': '#96CEB4'},
        'CNN_Branch': {'pos': (3.5, 3), 'size': (2, 2), 'color': '#FF6B6B'},
        'Transformer': {'pos': (6.5, 3), 'size': (2, 2), 'color': '#4ECDC4'},
        'Concatenate': {'pos': (10, 5), 'size': (1.5, 1), 'color': '#DDA0DD'},
        'Dense1': {'pos': (12.5, 5), 'size': (1.5, 1), 'color': '#45B7D1'},
        'Dense2': {'pos': (15, 5), 'size': (1.5, 1), 'color': '#F0A3FF'},
        'Output': {'pos': (17.5, 5), 'size': (1.5, 1), 'color': '#FFEAA7'}
    }
    
    # 绘制组件
    for name, info in components.items():
        rect = Rectangle(info['pos'], info['size'][0], info['size'][1], 
                        facecolor=info['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # 添加文本
        ax.text(info['pos'][0] + info['size'][0]/2, info['pos'][1] + info['size'][1]/2, 
               name.replace('_', '\n'), ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制连接线
    connections = [
        ((1.75, 5.5), (3.5, 7.5)),  # Input -> Feature_Extraction
        ((1.75, 5.5), (3.5, 4)),    # Input -> CNN_Branch
        ((5.5, 4), (6.5, 4)),       # CNN_Branch -> Transformer
        ((5.5, 7.5), (10, 5.5)),    # Feature_Extraction -> Concatenate
        ((8.5, 4), (10, 5.5)),      # Transformer -> Concatenate
        ((11.5, 5.5), (12.5, 5.5)), # Concatenate -> Dense1
        ((14, 5.5), (15, 5.5)),     # Dense1 -> Dense2
        ((16.5, 5.5), (17.5, 5.5))  # Dense2 -> Output
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # 添加详细信息
    info_text = [
        "Input: (2, 128)",
        "Feature Extraction: 11 features",
        "CNN: 3 Conv2D + BN + MaxPool",
        "Transformer: 3 blocks",
        "Fusion: Dense(256) + Dense(11)"
    ]
    
    for i, text in enumerate(info_text):
        ax.text(1, 1.5 - i*0.3, text, fontsize=10, ha='left')
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('AFECNN Architecture Diagram', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('afecnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("创建模型复杂度可视化分析...")
    results = create_complexity_visualizations()
    
    print("\n创建架构图...")
    create_architecture_diagram()
    
    print("\n可视化分析完成！生成的文件:")
    print("1. model_complexity_analysis.png - 综合复杂度分析图")
    print("2. model_complexity_table.png - 详细数据表格")
    print("3. afecnn_architecture.png - 模型架构图")
    
    # 输出关键数据
    print(f"\n关键数据摘要:")
    print(f"总参数量: {results['total_params']:,}")
    print(f"总FLOPs: {results['total_flops']:,}")
    print(f"模型大小: {results['total_params'] * 4 / 1024 / 1024:.2f} MB")
    
    # 保存结果到文件
    with open('complexity_summary.txt', 'w') as f:
        f.write("AFECNN Model Complexity Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Parameters: {results['total_params']:,}\n")
        f.write(f"Total FLOPs: {results['total_flops']:,}\n")
        f.write(f"Model Size: {results['total_params'] * 4 / 1024 / 1024:.2f} MB\n\n")
        f.write("Component Breakdown:\n")
        f.write(f"  Feature Extraction: {results['feature_flops']:,} FLOPs (0 params)\n")
        f.write(f"  CNN Branch: {results['cnn_flops']:,} FLOPs ({results['cnn_params']:,} params)\n")
        f.write(f"  Transformer Branch: {results['transformer_flops']:,} FLOPs ({results['transformer_params']:,} params)\n")
        f.write(f"  Fusion Layer: {results['fusion_flops']:,} FLOPs ({results['fusion_params']:,} params)\n")
    
    print("4. complexity_summary.txt - 文本摘要") 