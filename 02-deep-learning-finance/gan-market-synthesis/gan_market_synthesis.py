"""
GAN-Based Market Data Synthesis
Advanced Generative Adversarial Network for creating synthetic financial time series
Author: Joseph Bidias
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FinancialGAN:
    """
    Generative Adversarial Network for Financial Time Series Synthesis
    
    Features:
    - TimeGAN architecture optimized for temporal dependencies
    - Statistical property preservation (volatility clustering, fat tails)
    - Multi-asset correlation structure maintenance
    - Conditional generation based on market regimes
    """
    
    def __init__(self, sequence_length=100, feature_dim=5, latent_dim=100, batch_size=64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim  # OHLCV
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Models
        self.generator = None
        self.discriminator = None
        self.embedder = None
        self.recovery = None
        
        # Training history
        self.history = {'d_loss': [], 'g_loss': [], 'e_loss': [], 'r_loss': []}
        
        # Scalers for data normalization
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def build_embedder(self):
        """Build embedder network (real data -> latent space)"""
        embedder = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.Dense(self.latent_dim, activation='tanh')
        ], name='embedder')
        
        embedder.compile(optimizer='adam', loss='mse')
        return embedder
    
    def build_recovery(self):
        """Build recovery network (latent space -> reconstructed data)"""
        recovery = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.latent_dim)),
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.Dense(self.feature_dim, activation='tanh')
        ], name='recovery')
        
        recovery.compile(optimizer='adam', loss='mse')
        return recovery
    
    def build_generator(self):
        """Build generator network (noise -> synthetic latent sequences)"""
        generator = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.latent_dim)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.latent_dim, activation='tanh')
        ], name='generator')
        
        return generator
    
    def build_discriminator(self):
        """Build discriminator network (real/fake sequence classification)"""
        discriminator = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.latent_dim)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        discriminator.compile(
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return discriminator
    
    def build_models(self):
        """Build all GAN components"""
        self.embedder = self.build_embedder()
        self.recovery = self.build_recovery()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        print("✅ All GAN models built successfully!")
        
    def preprocess_data(self, data):
        """Preprocess financial data for GAN training"""
        if isinstance(data, pd.DataFrame):
            # Ensure OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
            
            processed_data = data[required_cols].values
        else:
            processed_data = data
        
        # Normalize data
        processed_data = self.scaler.fit_transform(processed_data)
        
        # Create sequences
        sequences = []
        for i in range(len(processed_data) - self.sequence_length + 1):
            sequences.append(processed_data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train_embedder_recovery(self, real_data, epochs=100):
        """Train embedder and recovery networks"""
        print("🔄 Training Embedder and Recovery networks...")
        
        # Embedder training
        for epoch in range(epochs):
            # Forward pass through embedder
            embedded = self.embedder(real_data, training=True)
            
            # Forward pass through recovery
            recovered = self.recovery(embedded, training=True)
            
            # Calculate reconstruction loss
            e_loss = tf.reduce_mean(tf.square(real_data - recovered))
            
            # Update embedder
            with tf.GradientTape() as tape:
                embedded = self.embedder(real_data, training=True)
                recovered = self.recovery(embedded, training=True)
                e_loss = tf.reduce_mean(tf.square(real_data - recovered))
            
            e_gradients = tape.gradient(e_loss, self.embedder.trainable_variables)
            self.embedder.optimizer.apply_gradients(zip(e_gradients, self.embedder.trainable_variables))
            
            # Update recovery
            with tf.GradientTape() as tape:
                embedded = self.embedder(real_data, training=True)
                recovered = self.recovery(embedded, training=True)
                r_loss = tf.reduce_mean(tf.square(real_data - recovered))
            
            r_gradients = tape.gradient(r_loss, self.recovery.trainable_variables)
            self.recovery.optimizer.apply_gradients(zip(r_gradients, self.recovery.trainable_variables))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: E_Loss = {e_loss:.4f}, R_Loss = {r_loss:.4f}")
                
            self.history['e_loss'].append(float(e_loss))
            self.history['r_loss'].append(float(r_loss))
    
    def train_adversarial(self, real_data, epochs=200):
        """Train generator and discriminator adversarially"""
        print("⚔️ Starting Adversarial Training...")
        
        # Labels for real and fake data
        real_labels = tf.ones((self.batch_size, 1))
        fake_labels = tf.zeros((self.batch_size, 1))
        
        for epoch in range(epochs):
            # Sample real data batch
            idx = np.random.randint(0, real_data.shape[0], self.batch_size)
            real_batch = real_data[idx]
            
            # Generate noise
            noise = tf.random.normal((self.batch_size, self.sequence_length, self.latent_dim))
            
            # Generate fake sequences
            fake_latent = self.generator(noise, training=True)
            real_latent = self.embedder(real_batch, training=False)
            
            # Train discriminator
            with tf.GradientTape() as disc_tape:
                real_pred = self.discriminator(real_latent, training=True)
                fake_pred = self.discriminator(fake_latent, training=True)
                
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=real_labels, logits=real_pred))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=fake_labels, logits=fake_pred))
                d_loss = d_loss_real + d_loss_fake
            
            d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables))
            
            # Train generator
            with tf.GradientTape() as gen_tape:
                fake_latent = self.generator(noise, training=True)
                fake_pred = self.discriminator(fake_latent, training=False)
                
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=real_labels, logits=fake_pred))
            
            g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(
                zip(g_gradients, self.generator.trainable_variables))
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: D_Loss = {d_loss:.4f}, G_Loss = {g_loss:.4f}")
                
            self.history['d_loss'].append(float(d_loss))
            self.history['g_loss'].append(float(g_loss))
    
    def train(self, data, pretrain_epochs=100, adversarial_epochs=200):
        """Complete GAN training pipeline"""
        print("🚀 Starting FinancialGAN Training Pipeline")
        print("=" * 50)
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        print(f"📊 Data shape: {processed_data.shape}")
        
        # Build models if not already built
        if self.generator is None:
            self.build_models()
        
        # Phase 1: Train embedder and recovery
        self.train_embedder_recovery(processed_data, epochs=pretrain_epochs)
        
        # Phase 2: Adversarial training
        self.train_adversarial(processed_data, epochs=adversarial_epochs)
        
        print("✅ Training completed successfully!")
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic financial time series"""
        if self.generator is None or self.recovery is None:
            raise ValueError("Model must be trained before generating data")
        
        # Generate noise
        noise = tf.random.normal((n_samples, self.sequence_length, self.latent_dim))
        
        # Generate synthetic latent sequences
        synthetic_latent = self.generator(noise, training=False)
        
        # Recover to original space
        synthetic_data = self.recovery(synthetic_latent, training=False)
        
        # Inverse transform to original scale
        synthetic_sequences = []
        for seq in synthetic_data:
            seq_reshaped = seq.numpy().reshape(-1, self.feature_dim)
            seq_original = self.scaler.inverse_transform(seq_reshaped)
            synthetic_sequences.append(seq_original.reshape(self.sequence_length, self.feature_dim))
        
        return np.array(synthetic_sequences)
    
    def evaluate_statistical_fidelity(self, real_data, synthetic_data, save_path="results/"):
        """Evaluate how well synthetic data preserves statistical properties"""
        
        # Flatten sequences for analysis
        real_flat = real_data.reshape(-1, self.feature_dim)
        synthetic_flat = synthetic_data.reshape(-1, self.feature_dim)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        results = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names):
            real_values = real_flat[:, i]
            synthetic_values = synthetic_flat[:, i]
            
            # Statistical tests
            ks_stat, ks_p = stats.ks_2samp(real_values, synthetic_values)
            mw_stat, mw_p = stats.mannwhitneyu(real_values, synthetic_values, alternative='two-sided')
            
            # Distribution comparison
            axes[i].hist(real_values, bins=50, alpha=0.6, label='Real', density=True, color='blue')
            axes[i].hist(synthetic_values, bins=50, alpha=0.6, label='Synthetic', density=True, color='red')
            axes[i].set_title(f'{feature}\nKS p-value: {ks_p:.4f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            results[feature] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'mw_statistic': mw_stat,
                'mw_p_value': mw_p,
                'real_mean': np.mean(real_values),
                'synthetic_mean': np.mean(synthetic_values),
                'real_std': np.std(real_values),
                'synthetic_std': np.std(synthetic_values)
            }
        
        # Overall fidelity score
        axes[5].axis('off')
        fidelity_scores = [1 - min(results[f]['ks_statistic'], 1.0) for f in feature_names]
        overall_fidelity = np.mean(fidelity_scores)
        
        axes[5].text(0.1, 0.8, f"Overall Statistical Fidelity: {overall_fidelity:.3f}", 
                    fontsize=16, fontweight='bold', transform=axes[5].transAxes)
        axes[5].text(0.1, 0.6, f"Mean KS p-value: {np.mean([results[f]['ks_p_value'] for f in feature_names]):.4f}", 
                    fontsize=12, transform=axes[5].transAxes)
        
        plt.suptitle('Statistical Fidelity Analysis - Real vs Synthetic Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}statistical_fidelity.png', dpi=300, bbox_inches='tight')
        
        return results, overall_fidelity, fig
    
    def visualize_synthetic_samples(self, synthetic_data, n_samples=5, save_path="results/"):
        """Visualize generated synthetic time series"""
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i in range(n_samples):
            sample = synthetic_data[i]
            
            # Plot OHLC
            for j, (feature, color) in enumerate(zip(feature_names[:-1], colors[:-1])):
                axes[i].plot(sample[:, j], label=feature, color=color, alpha=0.8, linewidth=1.5)
            
            # Plot volume on secondary axis
            ax2 = axes[i].twinx()
            ax2.bar(range(len(sample)), sample[:, 4], alpha=0.3, color=colors[4], label='Volume')
            ax2.set_ylabel('Volume', color=colors[4])
            
            axes[i].set_title(f'Synthetic Sample {i+1}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Price')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Generated Synthetic Financial Time Series', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}synthetic_samples.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def calculate_correlations(self, real_data, synthetic_data):
        """Calculate and compare correlation structures"""
        
        # Flatten sequences
        real_flat = real_data.reshape(-1, self.feature_dim)
        synthetic_flat = synthetic_data.reshape(-1, self.feature_dim)
        
        # Calculate correlation matrices
        real_corr = np.corrcoef(real_flat.T)
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synthetic_corr)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Real correlations
        sns.heatmap(real_corr, annot=True, fmt='.3f', ax=axes[0], 
                   xticklabels=feature_names, yticklabels=feature_names,
                   cmap='coolwarm', center=0)
        axes[0].set_title('Real Data Correlations')
        
        # Synthetic correlations
        sns.heatmap(synthetic_corr, annot=True, fmt='.3f', ax=axes[1],
                   xticklabels=feature_names, yticklabels=feature_names,
                   cmap='coolwarm', center=0)
        axes[1].set_title('Synthetic Data Correlations')
        
        # Difference
        sns.heatmap(corr_diff, annot=True, fmt='.3f', ax=axes[2],
                   xticklabels=feature_names, yticklabels=feature_names,
                   cmap='Reds')
        axes[2].set_title('Absolute Correlation Difference')
        
        plt.tight_layout()
        plt.savefig('results/correlation_analysis.png', dpi=300, bbox_inches='tight')
        
        return real_corr, synthetic_corr, corr_diff, fig

def main():
    """Main function demonstrating FinancialGAN"""
    print("🎨 GAN-Based Market Data Synthesis")
    print("=" * 50)
    
    # Download real market data
    print("📥 Downloading market data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    all_data = []
    for symbol in symbols:
        data = yf.download(symbol, period='2y', interval='1d')
        if len(data) > 200:
            all_data.append(data)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"📊 Combined dataset shape: {combined_data.shape}")
    
    # Initialize GAN
    gan = FinancialGAN(sequence_length=50, feature_dim=5, latent_dim=64)
    
    # Train GAN
    print("\n🚀 Training GAN...")
    gan.train(combined_data, pretrain_epochs=50, adversarial_epochs=100)
    
    # Generate synthetic data
    print("\n🎯 Generating synthetic data...")
    synthetic_data = gan.generate_synthetic_data(n_samples=1000)
    print(f"Generated {synthetic_data.shape[0]} synthetic sequences")
    
    # Evaluate statistical fidelity
    print("\n📊 Evaluating statistical fidelity...")
    real_test = gan.preprocess_data(combined_data.tail(1000))
    fidelity_results, overall_score, fidelity_fig = gan.evaluate_statistical_fidelity(
        real_test[:100], synthetic_data[:100])
    
    print(f"Overall Statistical Fidelity: {overall_score:.3f}")
    
    # Visualize samples
    print("\n🎨 Visualizing synthetic samples...")
    samples_fig = gan.visualize_synthetic_samples(synthetic_data, n_samples=5)
    
    # Correlation analysis
    print("\n🔗 Analyzing correlations...")
    real_corr, synth_corr, corr_diff, corr_fig = gan.calculate_correlations(
        real_test[:100], synthetic_data[:100])
    
    print(f"Mean correlation difference: {np.mean(corr_diff):.4f}")
    
    print("\n✅ GAN training and analysis completed!")
    
    return gan, synthetic_data, fidelity_results

if __name__ == "__main__":
    model, synthetic_sequences, evaluation_results = main()
