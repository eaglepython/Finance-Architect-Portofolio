"""
Ensemble Learning for Alpha Generation
Advanced ensemble methods for systematic alpha generation in quantitative trading
Author: Joseph Bidias
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, BaggingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import joblib
warnings.filterwarnings('ignore')

class AlphaEnsemble:
    """
    Advanced Ensemble Learning System for Alpha Generation
    
    Features:
    - Multi-level stacking with diverse base models
    - Time-aware cross-validation for financial data
    - Dynamic model weighting based on recent performance
    - Risk-adjusted alpha generation with Sharpe optimization
    - Real-time prediction and portfolio signal generation
    """
    
    def __init__(self, lookback_window=252, prediction_horizon=21, alpha_target=0.05):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.alpha_target = alpha_target
        
        # Ensemble components
        self.base_models = {}
        self.meta_model = None
        self.ensemble_model = None
        
        # Preprocessing
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Performance tracking
        self.performance_history = {}
        self.feature_importance = {}
        
        # Alpha generation parameters
        self.sharpe_target = 1.5
        self.max_drawdown_limit = 0.15
        
    def build_base_models(self):
        """Build diverse base models for ensemble"""
        
        self.base_models = {
            # Tree-based models
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=20,
                random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            
            # Linear models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            
            # Non-linear models
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            ),
            
            # Ensemble methods
            'ada_boost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'bagging': BaggingRegressor(n_estimators=100, random_state=42)
        }
        
        print(f"✅ Built {len(self.base_models)} base models")
        return self.base_models
    
    def build_meta_model(self):
        """Build meta-model for stacking"""
        self.meta_model = Ridge(alpha=0.1)
        return self.meta_model
    
    def engineer_features(self, data, symbol=None):
        """Engineer comprehensive feature set for alpha generation"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Technical indicators
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            features[f'sma_{window}'] = data['Close'].rolling(window).mean()
            features[f'price_to_sma_{window}'] = data['Close'] / features[f'sma_{window}']
        
        # Momentum indicators
        features['rsi'] = self.calculate_rsi(data['Close'], 14)
        features['macd'], features['macd_signal'] = self.calculate_macd(data['Close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_mean = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        features['bb_upper'] = bb_mean + 2 * bb_std
        features['bb_lower'] = bb_mean - 2 * bb_std
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume-based features
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_ma']
        features['price_volume'] = data['Close'] * data['Volume']
        
        # Higher-order moments
        for window in [10, 30]:
            returns_window = features['returns'].rolling(window)
            features[f'skewness_{window}'] = returns_window.skew()
            features[f'kurtosis_{window}'] = returns_window.kurt()
        
        # Regime indicators
        features['volatility_regime'] = (features['volatility'] > 
                                       features['volatility'].rolling(60).quantile(0.7)).astype(int)
        
        # Cross-sectional features (if multiple symbols)
        if symbol and hasattr(self, 'market_data'):
            market_returns = self.market_data.pct_change().mean(axis=1)
            features['beta'] = features['returns'].rolling(60).cov(market_returns) / market_returns.rolling(60).var()
            features['relative_strength'] = features['returns'].rolling(20).mean() - market_returns.rolling(20).mean()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volatility_lag_{lag}'] = features['volatility'].shift(lag)
        
        # Forward-looking target
        features['target'] = features['returns'].shift(-self.prediction_horizon)
        
        return features.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def train_base_models(self, X_train, y_train, cv_folds=5):
        """Train all base models with time-aware cross-validation"""
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        base_predictions = {}
        cv_scores = {}
        
        print("🔄 Training base models...")
        
        for name, model in self.base_models.items():
            try:
                print(f"  Training {name}...")
                
                # Cross-validation scores
                scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
                cv_scores[name] = -scores.mean()
                
                # Fit on full training data
                model.fit(X_train, y_train)
                
                # Generate out-of-fold predictions for meta-model
                predictions = np.zeros(len(X_train))
                for train_idx, val_idx in tscv.split(X_train):
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    predictions[val_idx] = fold_model.predict(X_train.iloc[val_idx])
                
                base_predictions[name] = predictions
                
            except Exception as e:
                print(f"  ❌ Failed to train {name}: {str(e)}")
                continue
        
        self.performance_history['base_cv_scores'] = cv_scores
        
        # Create meta-features from base model predictions
        meta_features = pd.DataFrame(base_predictions, index=X_train.index)
        
        print(f"✅ Trained {len(base_predictions)} base models successfully")
        return meta_features
    
    def train_meta_model(self, meta_features, y_train):
        """Train meta-model for stacking"""
        
        print("🔄 Training meta-model...")
        
        # Train meta-model on base model predictions
        self.meta_model.fit(meta_features, y_train)
        
        # Evaluate meta-model
        meta_score = self.meta_model.score(meta_features, y_train)
        self.performance_history['meta_r2'] = meta_score
        
        print(f"✅ Meta-model R² score: {meta_score:.4f}")
        return self.meta_model
    
    def create_ensemble_predictions(self, X_test):
        """Generate ensemble predictions"""
        
        # Get base model predictions
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                predictions = model.predict(X_test)
                base_predictions[name] = predictions
            except:
                continue
        
        # Create meta-features
        meta_features = pd.DataFrame(base_predictions, index=X_test.index)
        
        # Meta-model prediction
        ensemble_prediction = self.meta_model.predict(meta_features)
        
        return ensemble_prediction, base_predictions
    
    def calculate_alpha_metrics(self, predictions, actual_returns, risk_free_rate=0.02):
        """Calculate alpha generation metrics"""
        
        # Convert to pandas Series if needed
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
        if isinstance(actual_returns, np.ndarray):
            actual_returns = pd.Series(actual_returns)
        
        # Basic performance metrics
        mse = mean_squared_error(actual_returns, predictions)
        mae = mean_absolute_error(actual_returns, predictions)
        r2 = r2_score(actual_returns, predictions)
        
        # Alpha-specific metrics
        prediction_returns = predictions.copy()
        
        # Information Ratio
        active_returns = prediction_returns - actual_returns.mean()
        tracking_error = active_returns.std()
        information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
        
        # Sharpe Ratio (assuming predictions drive portfolio returns)
        excess_returns = prediction_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Hit Rate
        correct_direction = ((predictions > 0) & (actual_returns > 0)) | ((predictions < 0) & (actual_returns < 0))
        hit_rate = correct_direction.mean()
        
        # Maximum Drawdown
        cumulative = (1 + prediction_returns).cumprod()
        max_drawdown = (cumulative / cumulative.expanding().max() - 1).min()
        
        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'information_ratio': information_ratio,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'tracking_error': tracking_error
        }
    
    def train(self, data, symbol=None):
        """Complete ensemble training pipeline"""
        
        print("🚀 Starting Ensemble Alpha Generation Training")
        print("=" * 60)
        
        # Feature engineering
        print("🔧 Engineering features...")
        features_df = self.engineer_features(data, symbol)
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns if col != 'target']
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Train-test split (time-aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Test data shape: {X_test.shape}")
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.feature_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Build models
        self.build_base_models()
        self.build_meta_model()
        
        # Train base models
        meta_features_train = self.train_base_models(X_train_scaled, y_train)
        
        # Train meta-model
        self.train_meta_model(meta_features_train, y_train)
        
        # Generate test predictions
        test_predictions, base_test_predictions = self.create_ensemble_predictions(X_test_scaled)
        
        # Evaluate performance
        performance = self.calculate_alpha_metrics(test_predictions, y_test)
        self.performance_history['test_performance'] = performance
        
        print("\n📊 Test Performance Metrics:")
        for metric, value in performance.items():
            print(f"  {metric}: {value:.4f}")
        
        # Feature importance analysis
        self.analyze_feature_importance(X_train.columns)
        
        print("\n✅ Ensemble training completed successfully!")
        
        return {
            'test_predictions': test_predictions,
            'base_predictions': base_test_predictions,
            'performance': performance,
            'feature_importance': self.feature_importance
        }
    
    def analyze_feature_importance(self, feature_names):
        """Analyze feature importance across ensemble"""
        
        importance_scores = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_scores[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores[name] = np.abs(model.coef_)
        
        # Average importance across models
        if importance_scores:
            avg_importance = np.mean(list(importance_scores.values()), axis=0)
            self.feature_importance = dict(zip(feature_names, avg_importance))
        
        return self.feature_importance
    
    def generate_trading_signals(self, predictions, confidence_threshold=0.02):
        """Generate trading signals from predictions"""
        
        signals = pd.Series(index=predictions.index, data=0)
        
        # Long signals
        signals[predictions > confidence_threshold] = 1
        
        # Short signals
        signals[predictions < -confidence_threshold] = -1
        
        return signals
    
    def visualize_performance(self, results, save_path="results/"):
        """Create comprehensive performance visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. Prediction vs Actual scatter plot
        test_pred = results['test_predictions']
        y_test = results.get('y_test', test_pred)  # Placeholder if not available
        
        axes[0].scatter(y_test, test_pred, alpha=0.6, color='blue')
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Returns')
        axes[0].set_ylabel('Predicted Returns')
        axes[0].set_title('Prediction vs Actual Returns')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Feature importance
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importance = zip(*top_features)
            
            axes[1].barh(range(len(features)), importance, color='skyblue')
            axes[1].set_yticks(range(len(features)))
            axes[1].set_yticklabels(features)
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Top 15 Feature Importance')
        
        # 3. Model performance comparison
        if 'base_cv_scores' in self.performance_history:
            scores = self.performance_history['base_cv_scores']
            models = list(scores.keys())
            values = list(scores.values())
            
            axes[2].bar(range(len(models)), values, color='lightgreen', alpha=0.7)
            axes[2].set_xticks(range(len(models)))
            axes[2].set_xticklabels(models, rotation=45)
            axes[2].set_ylabel('CV MSE Score')
            axes[2].set_title('Base Model Performance (CV)')
        
        # 4. Cumulative returns simulation
        simulated_returns = pd.Series(test_pred).cumsum()
        axes[3].plot(simulated_returns, linewidth=2, color='green')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Cumulative Alpha')
        axes[3].set_title('Simulated Alpha Generation')
        axes[3].grid(True, alpha=0.3)
        
        # 5. Performance metrics radar chart
        performance = results['performance']
        metrics = ['r2_score', 'hit_rate', 'information_ratio', 'sharpe_ratio']
        values = [max(0, min(1, performance.get(m, 0))) for m in metrics]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        axes[4].plot(angles, values, 'o-', linewidth=2, color='red')
        axes[4].fill(angles, values, alpha=0.25, color='red')
        axes[4].set_xticks(angles[:-1])
        axes[4].set_xticklabels(metrics)
        axes[4].set_ylim(0, 1)
        axes[4].set_title('Performance Metrics Radar')
        axes[4].grid(True)
        
        # 6. Residuals analysis
        if len(test_pred) == len(y_test):
            residuals = y_test - test_pred
            axes[5].hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[5].set_xlabel('Residuals')
            axes[5].set_ylabel('Frequency')
            axes[5].set_title('Residuals Distribution')
            axes[5].grid(True, alpha=0.3)
        
        plt.suptitle('Ensemble Alpha Generation - Performance Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}ensemble_performance_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """Main function demonstrating Ensemble Alpha Generation"""
    
    print("🎯 Ensemble Learning for Alpha Generation")
    print("=" * 50)
    
    # Download market data
    print("📥 Downloading market data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    ensemble_results = {}
    
    for symbol in symbols[:2]:  # Test on first 2 symbols
        print(f"\n📊 Processing {symbol}...")
        
        # Download data
        data = yf.download(symbol, period='3y', interval='1d')
        
        if len(data) < 500:
            print(f"  ❌ Insufficient data for {symbol}")
            continue
        
        # Initialize ensemble
        ensemble = AlphaEnsemble(
            lookback_window=252, 
            prediction_horizon=21, 
            alpha_target=0.05
        )
        
        # Train ensemble
        results = ensemble.train(data, symbol=symbol)
        ensemble_results[symbol] = results
        
        # Generate visualization
        fig = ensemble.visualize_performance(results)
        
        print(f"  ✅ {symbol} analysis completed")
        print(f"     Sharpe Ratio: {results['performance']['sharpe_ratio']:.3f}")
        print(f"     Hit Rate: {results['performance']['hit_rate']:.3f}")
        print(f"     R² Score: {results['performance']['r2_score']:.3f}")
    
    # Summary results
    print("\n" + "="*60)
    print("📊 ENSEMBLE ALPHA GENERATION SUMMARY")
    print("="*60)
    
    avg_sharpe = np.mean([results['performance']['sharpe_ratio'] 
                         for results in ensemble_results.values()])
    avg_hit_rate = np.mean([results['performance']['hit_rate'] 
                           for results in ensemble_results.values()])
    avg_r2 = np.mean([results['performance']['r2_score'] 
                     for results in ensemble_results.values()])
    
    print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Average Hit Rate: {avg_hit_rate:.3f}")
    print(f"Average R² Score: {avg_r2:.3f}")
    print(f"Symbols Processed: {len(ensemble_results)}")
    
    return ensemble_results

if __name__ == "__main__":
    results = main()
