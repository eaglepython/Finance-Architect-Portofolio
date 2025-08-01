"""
CNN Technical Pattern Recognition System
Advanced convolutional neural network for automated chart pattern detection
Author: Joseph Bidias
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PatternCNN:
    """
    Convolutional Neural Network for Technical Pattern Recognition
    
    Detects 12 classic chart patterns:
    - Head & Shoulders
    - Inverse Head & Shoulders
    - Double Top
    - Double Bottom
    - Ascending Triangle
    - Descending Triangle
    - Symmetrical Triangle
    - Bull Flag
    - Bear Flag
    - Wedge Rising
    - Wedge Falling
    - Rectangle
    """
    
    def __init__(self, image_size=(64, 64), num_patterns=12):
        self.image_size = image_size
        self.num_patterns = num_patterns
        self.model = None
        self.pattern_names = [
            'Head_Shoulders', 'Inv_Head_Shoulders', 'Double_Top', 'Double_Bottom',
            'Ascending_Triangle', 'Descending_Triangle', 'Symmetrical_Triangle',
            'Bull_Flag', 'Bear_Flag', 'Wedge_Rising', 'Wedge_Falling', 'Rectangle'
        ]
        self.confidence_threshold = 0.75
        
    def build_model(self):
        """Build CNN architecture optimized for pattern recognition"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_patterns, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def price_to_image(self, prices, volume=None):
        """Convert price series to chart image for CNN input"""
        fig, ax = plt.subplots(figsize=(self.image_size[0]/10, self.image_size[1]/10), dpi=10)
        ax.plot(prices, linewidth=2, color='black')
        
        if volume is not None:
            ax2 = ax.twinx()
            ax2.bar(range(len(volume)), volume, alpha=0.3, color='blue')
            ax2.set_ylim(0, max(volume) * 3)
        
        ax.set_xlim(0, len(prices)-1)
        ax.set_ylim(min(prices) * 0.98, max(prices) * 1.02)
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert to grayscale
        gray = np.dot(buf, [0.2989, 0.5870, 0.1140])
        gray = np.resize(gray, self.image_size)
        gray = gray / 255.0
        
        plt.close(fig)
        return gray.reshape(*self.image_size, 1)
    
    def generate_synthetic_patterns(self, num_samples=1000):
        """Generate synthetic chart patterns for training"""
        patterns = []
        labels = []
        
        for pattern_idx, pattern_name in enumerate(self.pattern_names):
            for _ in range(num_samples // self.num_patterns):
                if pattern_name == 'Head_Shoulders':
                    pattern = self._generate_head_shoulders()
                elif pattern_name == 'Double_Top':
                    pattern = self._generate_double_top()
                elif pattern_name == 'Ascending_Triangle':
                    pattern = self._generate_ascending_triangle()
                elif pattern_name == 'Bull_Flag':
                    pattern = self._generate_bull_flag()
                else:
                    # Generic pattern generation
                    pattern = self._generate_generic_pattern(pattern_name)
                
                image = self.price_to_image(pattern)
                patterns.append(image)
                labels.append(pattern_idx)
        
        return np.array(patterns), tf.keras.utils.to_categorical(labels, self.num_patterns)
    
    def _generate_head_shoulders(self):
        """Generate Head & Shoulders pattern"""
        base_price = 100
        noise = np.random.normal(0, 0.5, 50)
        
        # Left shoulder
        left_shoulder = np.linspace(base_price, base_price + 5, 10) + noise[:10]
        
        # Head
        head_up = np.linspace(base_price + 5, base_price + 10, 10) + noise[10:20]
        head_down = np.linspace(base_price + 10, base_price + 5, 10) + noise[20:30]
        
        # Right shoulder
        right_shoulder = np.linspace(base_price + 5, base_price + 4, 10) + noise[30:40]
        
        # Breakdown
        breakdown = np.linspace(base_price + 4, base_price - 2, 10) + noise[40:50]
        
        return np.concatenate([left_shoulder, head_up, head_down, right_shoulder, breakdown])
    
    def _generate_double_top(self):
        """Generate Double Top pattern"""
        base_price = 100
        noise = np.random.normal(0, 0.3, 50)
        
        # First peak
        peak1_up = np.linspace(base_price, base_price + 8, 12) + noise[:12]
        peak1_down = np.linspace(base_price + 8, base_price + 2, 12) + noise[12:24]
        
        # Second peak
        peak2_up = np.linspace(base_price + 2, base_price + 8.2, 13) + noise[24:37]
        peak2_down = np.linspace(base_price + 8.2, base_price - 1, 13) + noise[37:50]
        
        return np.concatenate([peak1_up, peak1_down, peak2_up, peak2_down])
    
    def _generate_ascending_triangle(self):
        """Generate Ascending Triangle pattern"""
        base_price = 100
        resistance = base_price + 5
        
        # Ascending support line with horizontal resistance
        support_slope = np.linspace(base_price, base_price + 4, 50)
        price_action = []
        
        for i in range(50):
            if i % 8 < 4:  # Upward move towards resistance
                price = min(support_slope[i] + np.random.uniform(0, 2), resistance)
            else:  # Pullback to support
                price = support_slope[i] + np.random.uniform(-0.5, 0.5)
            price_action.append(price)
        
        return np.array(price_action)
    
    def _generate_bull_flag(self):
        """Generate Bull Flag pattern"""
        base_price = 100
        
        # Strong upward move (flagpole)
        flagpole = np.linspace(base_price, base_price + 8, 15)
        
        # Consolidation (flag)
        flag_high = base_price + 8
        flag_low = base_price + 6
        consolidation = []
        
        for i in range(25):
            price = flag_high - (i * 0.08) + np.random.uniform(-0.3, 0.3)
            consolidation.append(max(price, flag_low))
        
        # Breakout
        breakout = np.linspace(consolidation[-1], base_price + 12, 10)
        
        return np.concatenate([flagpole, consolidation, breakout])
    
    def _generate_generic_pattern(self, pattern_name):
        """Generate other patterns with basic shapes"""
        base_price = 100
        length = 50
        
        if 'Triangle' in pattern_name:
            return np.random.triangular(base_price - 2, base_price, base_price + 2, length)
        elif 'Rectangle' in pattern_name:
            high, low = base_price + 3, base_price - 1
            return np.random.uniform(low, high, length)
        else:
            # Random walk with trend
            trend = 1 if 'Bull' in pattern_name or 'Rising' in pattern_name else -1
            return np.cumsum(np.random.normal(trend * 0.1, 0.5, length)) + base_price
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the CNN model"""
        print("Generating synthetic training data...")
        X, y = self.generate_synthetic_patterns(num_samples=12000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7
        )
        
        # Train model
        print("Training CNN model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy, test_top_k = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Results:")
        print(f"Accuracy: {test_accuracy:.3f}")
        print(f"Top-3 Accuracy: {test_top_k:.3f}")
        
        return history, (X_test, y_test)
    
    def detect_patterns(self, price_data, confidence_threshold=None):
        """Detect patterns in price data"""
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        if len(price_data) < 50:
            return {"error": "Insufficient data points (minimum 50 required)"}
        
        # Convert to image
        image = self.price_to_image(price_data)
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)[0]
        confidence = np.max(predictions)
        pattern_idx = np.argmax(predictions)
        
        if confidence >= confidence_threshold:
            return {
                "pattern": self.pattern_names[pattern_idx],
                "confidence": float(confidence),
                "all_probabilities": {
                    name: float(prob) for name, prob in zip(self.pattern_names, predictions)
                }
            }
        else:
            return {
                "pattern": "No_Clear_Pattern",
                "confidence": float(confidence),
                "message": f"Confidence {confidence:.3f} below threshold {confidence_threshold}"
            }
    
    def analyze_real_data(self, symbol, period="1y", interval="1d"):
        """Analyze real market data for patterns"""
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if len(data) < 50:
                return {"error": f"Insufficient data for {symbol}"}
            
            # Analyze recent 50-day window
            recent_prices = data['Close'].tail(50).values
            result = self.detect_patterns(recent_prices)
            
            result['symbol'] = symbol
            result['analysis_period'] = f"Last 50 {interval} candles"
            result['data_range'] = f"{data.index[0].date()} to {data.index[-1].date()}"
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to analyze {symbol}: {str(e)}"}
    
    def visualize_pattern_detection(self, price_data, save_path=None):
        """Visualize pattern detection results"""
        result = self.detect_patterns(price_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot price chart
        ax1.plot(price_data, linewidth=2, color='blue', alpha=0.8)
        ax1.set_title(f'Price Chart - Detected: {result.get("pattern", "Unknown")} '
                     f'(Confidence: {result.get("confidence", 0):.3f})', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot probability distribution
        if 'all_probabilities' in result:
            probs = result['all_probabilities']
            patterns = list(probs.keys())
            probabilities = list(probs.values())
            
            colors = ['red' if p == result.get('pattern') else 'lightblue' for p in patterns]
            bars = ax2.bar(patterns, probabilities, color=colors, alpha=0.7)
            ax2.set_title('Pattern Probability Distribution', fontsize=14)
            ax2.set_ylabel('Probability', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add confidence threshold line
            ax2.axhline(y=self.confidence_threshold, color='red', linestyle='--', 
                       label=f'Confidence Threshold ({self.confidence_threshold})')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, result

def main():
    """Main function to demonstrate CNN Pattern Recognition"""
    print("🎯 CNN Technical Pattern Recognition System")
    print("=" * 50)
    
    # Initialize CNN
    cnn = PatternCNN()
    cnn.build_model()
    
    print("Model Architecture:")
    cnn.model.summary()
    
    # Train model
    print("\n📊 Training Phase:")
    history, test_data = cnn.train(epochs=50, batch_size=64)
    
    # Test on real data
    print("\n🔍 Testing on Real Market Data:")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    results = []
    for symbol in symbols:
        result = cnn.analyze_real_data(symbol, period="6mo", interval="1d")
        results.append(result)
        
        if 'error' not in result:
            print(f"\n{symbol}:")
            print(f"  Pattern: {result['pattern']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"\n{symbol}: {result['error']}")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    print(f"Pattern Types Detected: {cnn.num_patterns}")
    print(f"Confidence Threshold: {cnn.confidence_threshold}")
    print(f"Image Resolution: {cnn.image_size}")
    
    return cnn, results

if __name__ == "__main__":
    cnn_model, analysis_results = main()
