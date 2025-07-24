# AFECNN Gate Model Analysis Toolkit

A comprehensive analysis toolkit for the AFECNN Gate model, providing detailed insights into model complexity, parameter distribution, computational requirements, and performance characteristics.

## 🌟 Features

- **Parameter Analysis**: Detailed breakdown of trainable and non-trainable parameters
- **FLOPs Calculation**: Comprehensive computational complexity analysis
- **Memory Usage**: Estimation of parameter and activation memory requirements
- **Gate Mechanism Analysis**: Specific analysis of the gate fusion mechanism
- **Visual Reports**: Professional charts and architecture diagrams
- **Multiple Output Formats**: Text, CSV, JSON, and visual reports
- **Comparative Analysis**: Component-wise performance comparison

## 🏗️ Model Architecture

The AFECNN Gate model features a dual-branch architecture with gate-based fusion:

```
Input (2, 128)
    ├── Feature Extraction Branch (Artificial Features)
    │   ├── Complex moment calculations (2nd, 4th, 6th, 8th order)
    │   ├── Mixed moments and cumulants
    │   └── Feature parameters (M_1 to M_6, Param_R, C_60)
    │
    └── Deep Learning Branch (CNN + Transformer)
        ├── CNN Layers (128 → 64 → 32 filters)
        ├── Transformer Encoders (2 heads, 16 ff_dim)
        └── Global Average Pooling
             ↓
    Gate Fusion Mechanism
        ├── Dense mapping (128 dims each)
        ├── Concatenation (256 dims)
        ├── Sigmoid gate (128 dims)
        ├── Element-wise operations
        └── Weighted fusion
             ↓
    Classification Layer (11 classes)
```

### Key Differences from Standard AFECNN

- **Gate Mechanism**: Uses learnable gates for adaptive feature fusion
- **Smaller Transformer**: 2 heads vs 4 heads, 16 ff_dim vs 64 ff_dim
- **Dynamic Features**: Tensor-based feature extraction vs pre-computed features
- **Single Input**: One input stream vs dual input streams

## 📦 Installation

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install numpy>=1.19.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install pandas>=1.3.0
```

### Optional Dependencies

```bash
pip install scipy>=1.7.0  # For advanced statistical calculations
pip install scikit-learn>=1.0.0  # For additional analysis tools
```

## 🚀 Quick Start

### Basic Analysis

```bash
python run_complexity_analysis_gate.py
```

### Verbose Output with Custom Directory

```bash
python run_complexity_analysis_gate.py --output-dir my_analysis --verbose
```

### Generate Specific Format Only

```bash
python run_complexity_analysis_gate.py --format csv --no-visualizations
```

## 📊 Analysis Components

### 1. Model Analysis (`model_analysis_gate.py`)

Provides detailed parameter analysis and model structure breakdown:

```python
from model_analysis_gate import analyze_gate_model_complexity

results = analyze_gate_model_complexity()
print(f"Total parameters: {results['complete_model']['total_params']:,}")
print(f"Gate parameters: {results['gate_params']:,}")
```

### 2. FLOPs Calculator (`detailed_flop_calculator_gate.py`)

Calculates computational complexity including gate mechanism:

```python
from detailed_flop_calculator_gate import analyze_computational_complexity

complexity = analyze_computational_complexity()
print(f"Total FLOPs: {complexity['flops']['total']:,}")
print(f"Gate FLOPs: {complexity['flops']['gate_fusion']:,}")
```

### 3. Visualizations (`visualize_model_complexity_gate.py`)

Generates comprehensive visual reports:

```python
from visualize_model_complexity_gate import create_comprehensive_report

report_dir = create_comprehensive_report()
print(f"Report generated in: {report_dir}")
```

## 📈 Output Files

The toolkit generates the following files:

### Visual Reports
- `parameter_distribution.png` - Parameter distribution across components
- `flops_analysis.png` - FLOPs breakdown and efficiency metrics
- `architecture_diagram.png` - Model architecture visualization

### Data Files
- `parameter_breakdown.csv` - Detailed parameter statistics
- `flops_breakdown.csv` - FLOPs distribution data
- `model_analysis.json` - Complete model analysis results
- `complexity_analysis.json` - Computational complexity data

### Reports
- `analysis_report.txt` - Comprehensive text report
- `analysis_report.md` - Markdown format report
- `parameter_summary.csv` - Parameter summary table
- `flops_summary.csv` - FLOPs summary table

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from model_analysis_gate import create_complete_gate_model

# Create custom model
model = create_complete_gate_model(
    input_shape=(2, 128),
    num_classes=11
)

# Analyze custom model
results = analyze_gate_model_complexity(
    input_shape=(2, 128),
    num_classes=11
)
```

### Component-wise Analysis

```python
from model_analysis_gate import (
    create_feature_extraction_model,
    create_cnn_transformer_model,
    create_gate_fusion_model
)

# Analyze individual components
feature_model = create_feature_extraction_model((2, 128))
cnn_model = create_cnn_transformer_model((2, 128))
gate_model = create_gate_fusion_model(11, 32, 11)
```

## 📊 Key Metrics

### Parameter Distribution
- **CNN + Transformer**: ~85% of parameters
- **Gate Mechanism**: ~15% of parameters
- **Feature Extraction**: 0% (compute-only)

### FLOPs Distribution
- **Feature Extraction**: ~20% of FLOPs
- **CNN**: ~25% of FLOPs
- **Transformer**: ~40% of FLOPs
- **Gate Fusion**: ~10% of FLOPs
- **Classification**: ~5% of FLOPs

### Efficiency Metrics
- **Model Size**: ~0.5-2 MB (Lightweight)
- **FLOPs per Parameter**: ~50-200
- **Memory Efficiency**: Optimized for deployment

## 🔬 Gate Mechanism Analysis

The gate mechanism provides adaptive feature fusion:

```python
# Gate mechanism components
gate_components = {
    'mapping_layers': 'Dense(128) for each branch',
    'concatenation': 'Combine features (256 dims)',
    'gate_computation': 'Dense(128) + Sigmoid',
    'element_wise_ops': 'Multiply and Add operations',
    'fusion_output': 'Weighted combination'
}
```

### Gate Advantages
- **Adaptive Weighting**: Learns optimal feature combination
- **Reduced Overfitting**: Regularizes feature fusion
- **Interpretability**: Gate values show feature importance
- **Flexibility**: Adapts to different input patterns

## 🎯 Use Cases

### 1. Model Optimization
```bash
# Analyze current model
python run_complexity_analysis_gate.py --verbose

# Compare with baseline
python run_complexity_analysis_gate.py --output-dir optimization_study
```

### 2. Deployment Planning
```bash
# Generate deployment report
python run_complexity_analysis_gate.py --format all --output-dir deployment_analysis
```

### 3. Research Analysis
```bash
# Detailed academic analysis
python run_complexity_analysis_gate.py --verbose --output-dir research_results
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check dependencies
   pip install tensorflow numpy matplotlib seaborn pandas
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size or use smaller input
   python run_complexity_analysis_gate.py --no-visualizations
   ```

3. **CUDA Issues**
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   python run_complexity_analysis_gate.py
   ```

### Performance Tips

- Use `--no-visualizations` for faster analysis
- Specify `--format csv` for data-only output
- Use `--verbose` for detailed debugging information

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd afecnn-gate-analysis

# Install development dependencies
pip install -e .
pip install pytest black flake8
```

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=.
```

### Code Style

```bash
# Format code
black *.py

# Check style
flake8 *.py
```

## 📚 References

1. **AFECNN**: Original Artificial Feature Enhanced CNN
2. **Gate Mechanisms**: Attention and gating in neural networks
3. **Model Complexity**: FLOPs and parameter analysis methods
4. **Transformer Networks**: Attention mechanisms and efficiency

## 📄 License

This toolkit is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Matplotlib and Seaborn for visualization capabilities
- Scientific computing community for analysis methodologies

---

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the troubleshooting section

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Compatibility**: Python 3.8+, TensorFlow 2.8+ 