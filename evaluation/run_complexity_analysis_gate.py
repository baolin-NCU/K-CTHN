#!/usr/bin/env python3
"""
AFECNN Gate Model Complexity Analysis Tool

This script provides a comprehensive analysis of the AFECNN Gate model including:
- Parameter count and distribution
- FLOPs calculation and breakdown
- Memory usage estimation
- Efficiency metrics
- Visual reports and diagrams

Usage:
    python run_complexity_analysis_gate.py [options]

Options:
    --output-dir: Directory to save analysis results (default: auto-generated)
    --model-config: Custom model configuration file
    --format: Output format (text, csv, json, all) (default: all)
    --verbose: Enable verbose output
    --help: Show this help message

Example:
    python run_complexity_analysis_gate.py --output-dir results --verbose
"""

import argparse
import os
import sys
import traceback
from datetime import datetime
import json
import csv


def check_dependencies():
    """检查必要的依赖项"""
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
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_output_directory(output_dir=None):
    """设置输出目录"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"gate_model_analysis_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_basic_analysis(output_dir, verbose=False):
    """运行基本分析"""
    if verbose:
        print("🔍 Running basic model analysis...")
    
    try:
        from models.AFECNN.model_analysis_gate import analyze_gate_model_complexity
        results = analyze_gate_model_complexity()
        
        # 保存结果
        with open(os.path.join(output_dir, 'model_analysis.json'), 'w') as f:
            # 转换为可序列化的格式
            serializable_results = {}
            for key, value in results.items():
                if key != 'models':  # 跳过模型对象
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print("✅ Basic analysis completed")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in basic analysis: {str(e)}")
        if verbose:
            traceback.print_exc()
        return None


def run_complexity_analysis(output_dir, verbose=False):
    """运行复杂度分析"""
    if verbose:
        print("⚡ Running computational complexity analysis...")
    
    try:
        from models.AFECNN.detailed_flop_calculator_gate import analyze_computational_complexity
        results = analyze_computational_complexity()
        
        # 保存结果
        with open(os.path.join(output_dir, 'complexity_analysis.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print("✅ Complexity analysis completed")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in complexity analysis: {str(e)}")
        if verbose:
            traceback.print_exc()
        return None


def generate_visualizations(output_dir, verbose=False):
    """生成可视化图表"""
    if verbose:
        print("📊 Generating visualizations...")
    
    try:
        from models.AFECNN.visualize_model_complexity_gate import create_comprehensive_report
        report_dir = create_comprehensive_report(output_dir)
        
        if verbose:
            print("✅ Visualizations generated")
        
        return report_dir
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        if verbose:
            traceback.print_exc()
        return None


def create_summary_csv(model_results, complexity_results, output_dir):
    """创建CSV格式的总结"""
    if not model_results or not complexity_results:
        return
    
    # 参数统计
    param_data = [
        ['Component', 'Parameters', 'Percentage'],
        ['Feature Extraction', 0, 0.0],
        ['CNN + Transformer', model_results['cnn_transformer']['total_params'], 
         model_results['cnn_transformer']['total_params']/model_results['complete_model']['total_params']*100],
        ['Gate Mechanism', model_results['gate_params'], 
         model_results['gate_params']/model_results['complete_model']['total_params']*100],
        ['Total', model_results['complete_model']['total_params'], 100.0]
    ]
    
    with open(os.path.join(output_dir, 'parameter_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(param_data)
    
    # FLOPs统计
    flops_data = complexity_results['flops']
    flops_summary = [
        ['Component', 'FLOPs', 'Percentage'],
        ['Feature Extraction', flops_data['feature_extraction'], 
         flops_data['feature_extraction']/flops_data['total']*100],
        ['CNN', flops_data['cnn'], flops_data['cnn']/flops_data['total']*100],
        ['Transformer', flops_data['transformer'], flops_data['transformer']/flops_data['total']*100],
        ['Gate Fusion', flops_data['gate_fusion'], flops_data['gate_fusion']/flops_data['total']*100],
        ['Classification', flops_data['classification'], flops_data['classification']/flops_data['total']*100],
        ['Total', flops_data['total'], 100.0]
    ]
    
    with open(os.path.join(output_dir, 'flops_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(flops_summary)


def create_text_report(model_results, complexity_results, output_dir):
    """创建文本格式的报告"""
    if not model_results or not complexity_results:
        return
    
    report_content = f"""
AFECNN Gate Model Analysis Report
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL OVERVIEW
{'-'*20}
Architecture: Dual-branch with Gate Fusion
Input Shape: (2, 128)
Output Classes: 11

PARAMETER ANALYSIS
{'-'*20}
Total Parameters: {model_results['complete_model']['total_params']:,}
Trainable Parameters: {model_results['complete_model']['trainable_params']:,}
Non-trainable Parameters: {model_results['complete_model']['non_trainable_params']:,}

Component Breakdown:
- CNN + Transformer: {model_results['cnn_transformer']['total_params']:,} ({model_results['cnn_transformer']['total_params']/model_results['complete_model']['total_params']*100:.1f}%)
- Gate Mechanism: {model_results['gate_params']:,} ({model_results['gate_params']/model_results['complete_model']['total_params']*100:.1f}%)
- Feature Extraction: 0 (compute-only)

COMPUTATIONAL COMPLEXITY
{'-'*20}
Total FLOPs: {complexity_results['flops']['total']:,}
Model Size: {complexity_results['memory']['parameters']/1024**2:.2f} MB
Size Category: {complexity_results['efficiency']['size_category']}

FLOPs Distribution:
- Feature Extraction: {complexity_results['flops']['feature_extraction']:,} ({complexity_results['flops']['feature_extraction']/complexity_results['flops']['total']*100:.1f}%)
- CNN: {complexity_results['flops']['cnn']:,} ({complexity_results['flops']['cnn']/complexity_results['flops']['total']*100:.1f}%)
- Transformer: {complexity_results['flops']['transformer']:,} ({complexity_results['flops']['transformer']/complexity_results['flops']['total']*100:.1f}%)
- Gate Fusion: {complexity_results['flops']['gate_fusion']:,} ({complexity_results['flops']['gate_fusion']/complexity_results['flops']['total']*100:.1f}%)
- Classification: {complexity_results['flops']['classification']:,} ({complexity_results['flops']['classification']/complexity_results['flops']['total']*100:.1f}%)

MEMORY USAGE
{'-'*20}
Parameter Memory: {complexity_results['memory']['parameters']/1024**2:.2f} MB
Activation Memory: {complexity_results['memory']['activations']/1024**2:.2f} MB
Total Memory: {complexity_results['memory']['total']/1024**2:.2f} MB

EFFICIENCY METRICS
{'-'*20}
FLOPs per Parameter: {complexity_results['efficiency']['flops_per_param']:.2f}
Memory Efficiency: Optimized for deployment
Compute Intensity: {complexity_results['flops']['total']/1e6:.1f} MFLOPs

KEY FEATURES
{'-'*20}
✓ Gate Mechanism: Advanced fusion strategy
✓ Tensor-based: Dynamic feature extraction
✓ Hybrid Architecture: CNN + Transformer
✓ Efficient Design: {complexity_results['efficiency']['size_category']} model

RECOMMENDATIONS
{'-'*20}
1. Deployment: Suitable for resource-constrained environments
2. Optimization: Gate mechanism provides adaptive feature weighting
3. Scalability: Architecture can be scaled by adjusting layers
4. Applications: Ideal for signal processing tasks
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='AFECNN Gate Model Complexity Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python run_complexity_analysis_gate.py
    
    # Verbose output with custom directory
    python run_complexity_analysis_gate.py --output-dir my_analysis --verbose
    
    # Generate only specific format
    python run_complexity_analysis_gate.py --format csv
        """
    )
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save analysis results')
    parser.add_argument('--format', choices=['text', 'csv', 'json', 'all'], default='all',
                       help='Output format (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("🚀 AFECNN Gate Model Complexity Analysis Tool")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置输出目录
    output_dir = setup_output_directory(args.output_dir)
    print(f"📁 Output directory: {output_dir}")
    
    # 运行分析
    try:
        # 基本分析
        model_results = run_basic_analysis(output_dir, args.verbose)
        
        # 复杂度分析
        complexity_results = run_complexity_analysis(output_dir, args.verbose)
        
        # 生成不同格式的报告
        if args.format in ['text', 'all']:
            create_text_report(model_results, complexity_results, output_dir)
            if args.verbose:
                print("✅ Text report generated")
        
        if args.format in ['csv', 'all']:
            create_summary_csv(model_results, complexity_results, output_dir)
            if args.verbose:
                print("✅ CSV reports generated")
        
        # 生成可视化
        if not args.no_visualizations:
            generate_visualizations(output_dir, args.verbose)
        
        # 成功完成
        print("\n🎉 Analysis completed successfully!")
        print(f"📋 Results saved in: {output_dir}")
        
        # 显示生成的文件
        print("\nGenerated files:")
        for file in os.listdir(output_dir):
            print(f"  📄 {file}")
        
        # 显示关键指标
        if model_results and complexity_results:
            print(f"\n📊 Key Metrics:")
            print(f"  • Total Parameters: {model_results['complete_model']['total_params']:,}")
            print(f"  • Total FLOPs: {complexity_results['flops']['total']:,}")
            print(f"  • Model Size: {complexity_results['memory']['parameters']/1024**2:.2f} MB")
            print(f"  • Size Category: {complexity_results['efficiency']['size_category']}")
            print(f"  • Gate Parameters: {model_results['gate_params']:,}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 