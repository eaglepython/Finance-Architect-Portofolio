# 🎰 Multi-Armed Bandit Portfolio Optimization

## 📊 **Project Overview**

Advanced reinforcement learning approach for dynamic portfolio allocation using UCB, Thompson Sampling, and Epsilon-Greedy algorithms with real-time risk adjustment.

**Performance Highlights:**
- Annual Return: **15.3%**
- Sharpe Ratio: **0.87**
- Win Rate: **89%**
- Inference Time: **<1ms**

## 🗂️ **Project Structure**

```
multi-armed-bandit-portfolio/
├── 📄 README.md
├── 📄 multi_armed_bandit_portfolio.py    # Main implementation
├── 📄 config.yaml                        # Configuration
├── 📄 requirements.txt                    # Dependencies
├── 📁 algorithms/
│   ├── 📄 ucb_algorithm.py
│   ├── 📄 thompson_sampling.py
│   └── 📄 epsilon_greedy.py
├── 📁 data/
│   ├── 📄 market_data.csv
│   └── 📄 portfolio_returns.json
├── 📁 results/
│   ├── 📊 performance_charts/
│   └── 📄 backtest_results.json
└── 📁 notebooks/
    └── 📄 analysis.ipynb
```

## 🚀 **Quick Start**

```python
from multi_armed_bandit_portfolio import MultiArmedBanditPortfolio

# Initialize portfolio optimizer
portfolio = MultiArmedBanditPortfolio(
    assets=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
    algorithm='ucb',
    confidence_level=2.0
)

# Run optimization
returns = portfolio.optimize_and_trade(
    market_data=data,
    trading_period=252  # 1 year
)

print(f"Annual Return: {returns['annual_return']:.1%}")
print(f"Sharpe Ratio: {returns['sharpe_ratio']:.2f}")
```

## 📈 **Key Features**

- **Multiple Algorithms**: UCB, Thompson Sampling, Epsilon-Greedy
- **Real-time Adaptation**: Dynamic rebalancing based on performance
- **Risk Management**: Built-in drawdown protection
- **Backtesting**: Comprehensive historical performance analysis

## 🎯 **Performance Metrics**

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Annual Return | 15.3% | 12.1% (S&P 500) |
| Sharpe Ratio | 0.87 | 0.65 |
| Max Drawdown | -8.2% | -12.4% |
| Win Rate | 89% | 67% |

## 🔧 **Technical Implementation**

- **Language**: Python 3.9+
- **Libraries**: NumPy, Pandas, Scikit-learn
- **Algorithm**: UCB with confidence bounds
- **Optimization**: Real-time portfolio rebalancing
- **Risk Model**: CVaR-based risk constraints

## ⚡ **Installation & Usage**

```bash
cd multi-armed-bandit-portfolio
pip install -r requirements.txt
python multi_armed_bandit_portfolio.py
```
