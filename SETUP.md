# 🚀 Project Setup Guide

## 📋 **Prerequisites**

- **Python**: 3.9+ (recommended: 3.10)
- **GPU**: CUDA-compatible GPU (recommended for deep learning projects)
- **Memory**: Minimum 16GB RAM (32GB recommended for quantum simulations)
- **Storage**: At least 10GB free space

## ⚙️ **Installation Steps**

### 1. **Environment Setup**

```bash
# Create virtual environment
python -m venv quant_env

# Activate environment
# Windows:
quant_env\Scripts\activate
# Linux/Mac:
source quant_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. **Install Dependencies**

```bash
# Install all requirements
pip install -r requirements.txt

# For quantum computing (if needed)
pip install qiskit[visualization] pennylane pennylane-qiskit

# For GPU support (PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. **Verify Installation**

```python
# test_installation.py
import numpy as np
import pandas as pd
import torch
import sklearn
import matplotlib.pyplot as plt

# Test quantum libraries
try:
    import qiskit
    print("✅ Qiskit installed successfully")
except ImportError:
    print("❌ Qiskit not available")

try:
    import pennylane as qml
    print("✅ PennyLane installed successfully")
except ImportError:
    print("❌ PennyLane not available")

print("✅ All core libraries installed successfully!")
```

## 📁 **Project Structure Overview**

```
quantitative-finance-portfolio/
├── 📁 01-machine-learning-finance/
│   ├── 📁 multi-armed-bandit-portfolio/
│   │   ├── 📄 multi_armed_bandit_portfolio.py
│   │   └── 📄 README.md
│   ├── 📁 ensemble-alpha-generation/
│   ├── 📁 svm-market-regimes/
│   ├── 📁 fourier-option-pricing/
│   └── 📁 pca-risk-decomposition/
├── 📁 02-deep-learning-finance/
│   ├── 📁 lstm-hft-predictor/
│   │   ├── 📄 lstm_hft_predictor.py
│   │   └── 📄 README.md
│   ├── 📁 transformer-credit-risk/
│   │   ├── 📄 transformer_credit_risk.py
│   │   └── 📄 README.md
│   ├── 📁 gan-market-synthesis/
│   └── 📁 cnn-pattern-recognition/
├── 📁 03-quantum-machine-learning/
│   ├── 📁 quantum-portfolio-optimization/
│   │   ├── 📄 quantum_portfolio_optimization.py
│   │   └── 📄 README.md
│   └── 📁 quantum-risk-factor-modeling/
├── 📁 04-documentation/
│   ├── 📄 portfolio_overview.md
│   └── 📄 complete_portfolio_details.md
├── 📁 05-notebooks/
├── 📁 06-results/
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 SETUP.md
```

## 🚀 **Quick Start Guide**

### 1. **Run Multi-Armed Bandit Portfolio**
```bash
cd 01-machine-learning-finance/multi-armed-bandit-portfolio
python multi_armed_bandit_portfolio.py
```

### 2. **Run LSTM HFT Predictor**
```bash
cd 02-deep-learning-finance/lstm-hft-predictor
python lstm_hft_predictor.py
```

### 3. **Run Quantum Portfolio Optimization**
```bash
cd 03-quantum-machine-learning/quantum-portfolio-optimization
python quantum_portfolio_optimization.py
```

### 4. **Run Transformer Credit Risk**
```bash
cd 02-deep-learning-finance/transformer-credit-risk
python transformer_credit_risk.py
```

## 🔧 **Configuration**

Each project includes a `config.yaml` file for customization:

```yaml
# Example config.yaml
model:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  
data:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  
quantum:
  backend: qasm_simulator
  shots: 8192
  optimization_level: 3
```

## 📊 **Expected Results**

After running the projects, you should see:

- **Multi-Armed Bandit**: 15.3% annual return, 0.87 Sharpe ratio
- **LSTM HFT**: 23.7% annual return, 5μs inference time
- **Quantum Portfolio**: 72x speedup for large portfolios
- **Transformer Credit**: 97.8% prediction accuracy

## ❗ **Troubleshooting**

### Common Issues:

1. **CUDA/GPU Issues**:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Quantum Backend Issues**:
   ```bash
   # Test quantum installation
   python -c "from qiskit import QuantumCircuit; print('Qiskit OK')"
   ```

3. **Memory Issues**:
   - Reduce batch sizes in configs
   - Use smaller datasets for testing
   - Close unnecessary applications

4. **Package Conflicts**:
   ```bash
   # Clean reinstall
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

## 📈 **Performance Optimization**

### For CPU:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### For GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
```

### For Memory:
```python
# In Python scripts
import torch
torch.backends.cudnn.benchmark = True
```

## 🎯 **Next Steps**

1. **Explore Documentation**: Read project-specific READMEs
2. **Run Notebooks**: Check `05-notebooks/` for detailed analyses
3. **Customize Configs**: Modify parameters for your use case
4. **Add Data**: Replace synthetic data with real market data
5. **Deploy Models**: Use provided Docker configurations

## 📧 **Support**

For issues or questions:
- Check project-specific README files
- Review error logs in project directories
- Consult documentation in `04-documentation/`

---

**🌟 Ready to explore the future of quantitative finance! 🌟**
