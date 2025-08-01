"""
CNN Pattern Recognition - Results Visualization and Analysis
Generates comprehensive performance reports and visual analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_performance_report(cnn_model, test_data, save_path="results/"):
    """Generate comprehensive performance analysis"""
    
    X_test, y_test = test_data
    
    # Predictions
    y_pred = cnn_model.model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification Report
    report = classification_report(
        y_true_classes, y_pred_classes, 
        target_names=cnn_model.pattern_names,
        output_dict=True
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix Heatmap
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cnn_model.pattern_names,
                yticklabels=cnn_model.pattern_names)
    plt.title('Confusion Matrix - Pattern Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Pattern')
    plt.xlabel('Predicted Pattern')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Per-Pattern Accuracy
    plt.subplot(2, 3, 2)
    pattern_accuracy = [report[pattern]['f1-score'] for pattern in cnn_model.pattern_names]
    bars = plt.bar(range(len(cnn_model.pattern_names)), pattern_accuracy, 
                   color='skyblue', alpha=0.8)
    plt.title('F1-Score by Pattern Type', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xlabel('Pattern Type')
    plt.xticks(range(len(cnn_model.pattern_names)), cnn_model.pattern_names, rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Precision vs Recall
    plt.subplot(2, 3, 3)
    precision = [report[pattern]['precision'] for pattern in cnn_model.pattern_names]
    recall = [report[pattern]['recall'] for pattern in cnn_model.pattern_names]
    
    plt.scatter(precision, recall, s=100, alpha=0.7, c=range(len(precision)), cmap='viridis')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall by Pattern', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add pattern labels
    for i, pattern in enumerate(cnn_model.pattern_names):
        plt.annotate(pattern.replace('_', ' '), (precision[i], recall[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Confidence Distribution
    plt.subplot(2, 3, 4)
    confidence_scores = np.max(y_pred, axis=1)
    plt.hist(confidence_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(cnn_model.confidence_threshold, color='red', linestyle='--', 
                label=f'Threshold ({cnn_model.confidence_threshold})')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 5. Top-K Accuracy Analysis
    plt.subplot(2, 3, 5)
    k_values = range(1, 6)
    top_k_accuracies = []
    
    for k in k_values:
        top_k_pred = np.argsort(y_pred, axis=1)[:, -k:]
        top_k_acc = np.mean([y_true_classes[i] in top_k_pred[i] for i in range(len(y_true_classes))])
        top_k_accuracies.append(top_k_acc)
    
    plt.plot(k_values, top_k_accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Top-K')
    plt.ylabel('Accuracy')
    plt.title('Top-K Accuracy Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # 6. Performance Metrics Summary
    plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        report['accuracy'],
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score']
    ]
    
    colors = ['gold', 'lightgreen', 'lightblue', 'plum']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)
    plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}cnn_performance_analysis.png', dpi=300, bbox_inches='tight')
    
    return report, cm, fig

def create_pattern_examples(cnn_model, save_path="results/"):
    """Generate visual examples of each pattern type"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, pattern_name in enumerate(cnn_model.pattern_names):
        # Generate example pattern
        if pattern_name == 'Head_Shoulders':
            example = cnn_model._generate_head_shoulders()
        elif pattern_name == 'Double_Top':
            example = cnn_model._generate_double_top()
        elif pattern_name == 'Ascending_Triangle':
            example = cnn_model._generate_ascending_triangle()
        elif pattern_name == 'Bull_Flag':
            example = cnn_model._generate_bull_flag()
        else:
            example = cnn_model._generate_generic_pattern(pattern_name)
        
        # Plot example
        axes[i].plot(example, linewidth=2, color='blue', alpha=0.8)
        axes[i].set_title(pattern_name.replace('_', ' '), fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Price')
        axes[i].set_xlabel('Time')
    
    plt.suptitle('CNN Pattern Recognition - Training Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}pattern_examples.png', dpi=300, bbox_inches='tight')
    
    return fig

def analyze_market_patterns(cnn_model, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'], 
                          save_path="results/"):
    """Analyze real market data and create summary report"""
    
    results = []
    successful_analyses = 0
    
    print("🔍 Analyzing Real Market Data...")
    
    for symbol in symbols:
        result = cnn_model.analyze_real_data(symbol, period="6mo", interval="1d")
        results.append(result)
        
        if 'error' not in result:
            successful_analyses += 1
            print(f"✅ {symbol}: {result['pattern']} (Confidence: {result['confidence']:.3f})")
        else:
            print(f"❌ {symbol}: {result['error']}")
    
    # Create summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pattern distribution
    pattern_counts = {}
    confidences = []
    
    for result in results:
        if 'error' not in result:
            pattern = result['pattern']
            confidence = result['confidence']
            
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            confidences.append(confidence)
    
    # Plot pattern distribution
    if pattern_counts:
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        
        ax1.pie(counts, labels=patterns, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Detected Pattern Distribution in Real Market Data', 
                     fontsize=14, fontweight='bold')
    
    # Plot confidence distribution
    if confidences:
        ax2.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(cnn_model.confidence_threshold, color='red', linestyle='--',
                   label=f'Threshold ({cnn_model.confidence_threshold})')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}market_analysis_summary.png', dpi=300, bbox_inches='tight')
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            'Symbol': r.get('symbol', 'Unknown'),
            'Pattern': r.get('pattern', 'Error'),
            'Confidence': r.get('confidence', 0.0),
            'Status': 'Success' if 'error' not in r else 'Error'
        } for r in results
    ])
    
    results_df.to_csv(f'{save_path}market_analysis_results.csv', index=False)
    
    print(f"\n📊 Analysis Summary:")
    print(f"Successful Analyses: {successful_analyses}/{len(symbols)}")
    print(f"Average Confidence: {np.mean(confidences):.3f}" if confidences else "No valid results")
    print(f"Results saved to: {save_path}")
    
    return results, fig

def generate_comprehensive_report():
    """Generate complete analysis report"""
    
    print("🚀 Generating Comprehensive CNN Pattern Recognition Report")
    print("=" * 60)
    
    # Initialize and train model
    from cnn_pattern_recognition import PatternCNN
    
    cnn = PatternCNN()
    cnn.build_model()
    
    # Train model
    print("\n📊 Training Model...")
    history, test_data = cnn.train(epochs=30, batch_size=64)
    
    # Generate performance report
    print("\n📈 Generating Performance Analysis...")
    report, cm, perf_fig = generate_performance_report(cnn, test_data)
    
    # Create pattern examples
    print("\n🎯 Creating Pattern Examples...")
    examples_fig = create_pattern_examples(cnn)
    
    # Analyze market data
    print("\n🔍 Analyzing Real Market Data...")
    market_results, market_fig = analyze_market_patterns(cnn)
    
    # Performance Summary
    print("\n" + "="*60)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {report['accuracy']:.3f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.3f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.3f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    print(f"Pattern Types: {len(cnn.pattern_names)}")
    print(f"Confidence Threshold: {cnn.confidence_threshold}")
    print(f"Model Parameters: {cnn.model.count_params():,}")
    
    # Top performing patterns
    pattern_f1_scores = [(pattern, report[pattern]['f1-score']) 
                        for pattern in cnn.pattern_names]
    pattern_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 5 Best Detected Patterns:")
    for i, (pattern, score) in enumerate(pattern_f1_scores[:5], 1):
        print(f"{i}. {pattern.replace('_', ' ')}: {score:.3f}")
    
    return cnn, report, market_results

if __name__ == "__main__":
    model, performance_report, market_analysis = generate_comprehensive_report()
