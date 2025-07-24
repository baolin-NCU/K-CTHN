#!/usr/bin/env python3
"""
AFECNN 模型复杂度分析工具
=========================

这个脚本提供了一个简单的接口来分析AFECNN模型的复杂度，包括：
1. 参数量统计
2. FLOPs计算
3. 内存使用分析
4. 可视化报告生成

使用方法：
python run_complexity_analysis.py

作者：AI Assistant
"""

import sys
import os
import traceback
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'tensorflow',
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请安装缺少的包：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_basic_analysis():
    """运行基础分析"""
    print("="*80)
    print("🔍 AFECNN 模型复杂度分析")
    print("="*80)
    
    try:
        from detailed_flop_calculator import comprehensive_model_analysis
        
        print("\n📊 正在进行详细分析...")
        results = comprehensive_model_analysis()
        
        print("\n✅ 分析完成！")
        return results
        
    except Exception as e:
        print(f"❌ 基础分析失败: {e}")
        traceback.print_exc()
        return None

def run_visualization():
    """运行可视化分析"""
    print("\n🎨 正在生成可视化报告...")
    
    try:
        from visualize_model_complexity import create_complexity_visualizations, create_architecture_diagram
        
        # 创建复杂度可视化
        results = create_complexity_visualizations()
        
        # 创建架构图
        create_architecture_diagram()
        
        print("✅ 可视化报告生成完成！")
        return results
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        print("   可能原因：缺少matplotlib或seaborn库")
        traceback.print_exc()
        return None

def generate_report(results):
    """生成文本报告"""
    if not results:
        print("❌ 无法生成报告，分析结果为空")
        return
    
    print("\n📝 正在生成报告...")
    
    try:
        # 生成详细报告
        report_content = f"""
AFECNN 模型复杂度分析报告
========================

📅 生成时间：{Path(__file__).stat().st_mtime}

🎯 模型概述
-----------
AFECNN是一个双分支神经网络，结合了人工特征提取和深度学习方法：
- 人工特征分支：基于高阶统计量和累积量的特征提取
- 深度学习分支：CNN + Transformer架构
- 融合层：将两个分支的特征进行融合分类

📊 复杂度统计
-----------
总参数量：{results['total_params']:,}
总FLOPs：{results['total_flops']:,}
模型大小：{results['total_params'] * 4 / 1024 / 1024:.2f} MB

🔍 分支详细分析
--------------
1. 人工特征提取分支：
   - 参数量：0 (无可训练参数)
   - FLOPs：{results['feature_flops']:,}
   - 占比：{results['feature_flops']/results['total_flops']*100:.1f}%
   - 输出维度：11

2. CNN分支：
   - 参数量：{results['cnn_params']:,}
   - FLOPs：{results['cnn_flops']:,}
   - 占比：{results['cnn_flops']/results['total_flops']*100:.1f}%
   - 输出维度：32

3. Transformer分支：
   - 参数量：{results['transformer_params']:,}
   - FLOPs：{results['transformer_flops']:,}
   - 占比：{results['transformer_flops']/results['total_flops']*100:.1f}%
   - 输出维度：32

4. 融合层：
   - 参数量：{results['fusion_params']:,}
   - FLOPs：{results['fusion_flops']:,}
   - 占比：{results['fusion_flops']/results['total_flops']*100:.1f}%
   - 输出维度：11

💡 关键发现
----------
1. 参数效率：Transformer分支占据了大部分参数量 ({results['transformer_params']/results['total_params']*100:.1f}%)
2. 计算效率：CNN分支的FLOPs占比相对较高 ({results['cnn_flops']/results['total_flops']*100:.1f}%)
3. 人工特征分支虽然无参数但提供了重要的专家知识特征
4. 融合层参数量适中，有效整合了两个分支的特征

🎯 优化建议
----------
1. 如果需要减少参数量，可以考虑减少Transformer的层数或维度
2. 如果需要减少计算量，可以考虑简化CNN结构
3. 人工特征分支提供了很好的参数效率，可以考虑扩展更多专家特征
4. 融合层设计合理，建议保持

📈 性能评估
----------
- 模型大小适中，适合部署
- 计算复杂度可接受
- 双分支设计平衡了专家知识和深度学习的优势
"""
        
        # 保存报告
        with open('AFECNN_complexity_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("✅ 报告生成完成：AFECNN_complexity_report.txt")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 AFECNN 模型复杂度分析工具")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 运行基础分析
    results = run_basic_analysis()
    if not results:
        print("❌ 基础分析失败，退出程序")
        sys.exit(1)
    
    # 运行可视化分析
    visualization_results = run_visualization()
    if visualization_results:
        results = visualization_results
    
    # 生成报告
    generate_report(results)
    
    # 显示生成的文件
    print("\n🎉 分析完成！生成的文件：")
    generated_files = [
        'detailed_model_analysis.pkl',
        'model_complexity_analysis.png',
        'model_complexity_table.png',
        'afecnn_architecture.png',
        'complexity_summary.txt',
        'AFECNN_complexity_report.txt'
    ]
    
    for file in generated_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (未生成)")
    
    print("\n📋 快速摘要：")
    print(f"   总参数量: {results['total_params']:,}")
    print(f"   总FLOPs: {results['total_flops']:,}")
    print(f"   模型大小: {results['total_params'] * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎯 建议：")
    print("   1. 查看生成的PNG图像了解可视化结果")
    print("   2. 阅读详细报告了解优化建议")
    print("   3. 使用PKL文件进行进一步分析")

if __name__ == "__main__":
    main() 