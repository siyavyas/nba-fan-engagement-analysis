"""
NBA Fan Engagement Analysis - Model Training & Evaluation
Trains classification models to predict attendance tiers
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/models'
RANDOM_STATE = 42

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_processed_data():
    
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv')
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv')
    X_train_scaled = pd.read_csv(f'{DATA_DIR}/X_train_scaled.csv')
    X_test_scaled = pd.read_csv(f'{DATA_DIR}/X_test_scaled.csv')
    
    y_train = pd.read_csv(f'{DATA_DIR}/y_train_classification.csv')['tier']
    y_test = pd.read_csv(f'{DATA_DIR}/y_test_classification.csv')['tier']
    
    train_metadata = pd.read_csv(f'{DATA_DIR}/train_metadata.csv')
    test_metadata = pd.read_csv(f'{DATA_DIR}/test_metadata.csv')
    
    # Load config
    with open(f'{DATA_DIR}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"Data loaded successfully")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Classes: {sorted(y_train.unique())}")
    
    return (X_train, X_test, X_train_scaled, X_test_scaled, 
            y_train, y_test, train_metadata, test_metadata, config)


def establish_baseline(y_train, y_test):
    
    # Strategy 1: Always predict the most common class
    most_common_class = y_train.mode()[0]
    y_pred_baseline = np.full(len(y_test), most_common_class)
    
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')
    
    print("\nBaseline Strategy: Always predict most common class")
    print(f"  Most common class in training: {most_common_class} (Medium)")
    print(f"  Baseline Accuracy: {baseline_accuracy:.3f}")
    print(f"  Baseline F1-Score: {baseline_f1:.3f}")
    
    print("\nOur models must beat this baseline to be useful!")
    
    return baseline_accuracy, baseline_f1


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Uses scaled features
    """
    print("\nModel configuration:")
    print("  max_iter=1000, random_state=42")
    print("  Using scaled features (required for logistic regression)")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nTraining complete")
    print(f"  Training Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Test F1-Score: {test_f1:.3f}")
    
    return model, y_pred_test, test_acc, test_f1


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Uses unscaled features
    """
    
    print("\nModel configuration:")
    print("  n_estimators=100")
    print("  max_depth=10")
    print("  min_samples_split=5")
    print("  random_state=42")
    print("  Using unscaled features (tree-based models don't need scaling)")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nTraining complete")
    print(f"  Training Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Test F1-Score: {test_f1:.3f}")
    
    # Check for overfitting
    if train_acc - test_acc > 0.15:
        print(f"\nWarning: Possible overfitting detected")
        print(f"  Train-test gap: {train_acc - test_acc:.3f}")
    
    return model, y_pred_test, test_acc, test_f1


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Uses unscaled features
    """
    
    print("\nModel configuration:")
    print("  n_estimators=100")
    print("  max_depth=5")
    print("  learning_rate=0.1")
    print("  random_state=42")
    print("  Using unscaled features")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nTraining complete")
    print(f"  Training Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Test F1-Score: {test_f1:.3f}")
    
    # Check for overfitting
    if train_acc - test_acc > 0.15:
        print(f"\nWarning: Possible overfitting detected")
        print(f"  Train-test gap: {train_acc - test_acc:.3f}")
    
    return model, y_pred_test, test_acc, test_f1


def compare_models(results):
    """
    Compare all models and identify the best performer
    """
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('test_f1', ascending=False)
    
    print("\nModel Performance Summary:")
    print(f"{'Model':<25} {'Accuracy':>12} {'F1-Score':>12} {'Improvement':>15}")
    
    baseline_f1 = results['Baseline']['test_f1']
    
    for model_name, metrics in comparison_df.iterrows():
        acc = metrics['test_accuracy']
        f1 = metrics['test_f1']
        improvement = ((f1 - baseline_f1) / baseline_f1) * 100
        
        print(f"{model_name:<25} {acc:>12.3f} {f1:>12.3f} {improvement:>14.1f}%")
    
    # Identify best model
    best_model = comparison_df.index[0]
    best_f1 = comparison_df.iloc[0]['test_f1']
    
    print(f"\nBest Model: {best_model}")
    print(f"   F1-Score: {best_f1:.3f}")
    print(f"   Improvement over baseline: {((best_f1 - baseline_f1) / baseline_f1) * 100:.1f}%")
    
    return best_model, comparison_df


def detailed_evaluation(model, model_name, X_test, y_test, feature_names):
    """
    Perform detailed evaluation of the best model
    """
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")

    report = classification_report(
        y_test, y_pred, 
        target_names=['Low', 'Medium', 'High'],
        digits=3
    )
    print(report)
    
    # Per-class metrics
    print("\nPer-Class Analysis:")

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    tier_names = ['Low', 'Medium', 'High']
    for i, tier in enumerate(tier_names):
        print(f"\n{tier} Attendance:")
        print(f"  Precision: {precision[i]:.3f} (when we predict {tier}, we're right {precision[i]*100:.1f}% of time)")
        print(f"  Recall:    {recall[i]:.3f} (we catch {recall[i]*100:.1f}% of actual {tier} games)")
        print(f"  F1-Score:  {f1[i]:.3f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n\nConfusion Matrix Analysis:")

    print("Actual →  Low  Med  High")
    for i, tier in enumerate(tier_names):
        row_str = f"{tier:>6}    "
        for j in range(3):
            row_str += f"{cm[i,j]:>4} "
        print(row_str)
    
    # Most common misclassifications
    print("\nMost Common Errors:")
    total_errors = len(y_test) - accuracy_score(y_test, y_pred, normalize=False)
    if total_errors > 0:
        for i in range(3):
            for j in range(3):
                if i != j and cm[i, j] > 0:
                    print(f"  Predicted {tier_names[j]} when actually {tier_names[i]}: {cm[i,j]} times")
    
    return y_pred, cm


def feature_importance_analysis(model, model_name, feature_names):
    """
    Analyze feature importance for tree-based models
    """
    
    if not hasattr(model, 'feature_importances_'):
        print(f"\n {model_name} does not support feature importance")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':>15}")
    
    for idx, row in feature_imp_df.head(10).iterrows():
        print(f"{feature_imp_df.index.get_loc(idx)+1:<6} {row['feature']:<30} {row['importance']:>15.4f}")
    
    return feature_imp_df


def error_analysis(model, model_name, X_test, y_test, test_metadata):
    """
    Analyze which games the model gets wrong
    """
    
    y_pred = model.predict(X_test)
    
    # Find misclassified examples
    errors = y_test != y_pred
    n_errors = errors.sum()
    
    print(f"\nTotal misclassifications: {n_errors} / {len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")
    
    if n_errors == 0:
        print("Perfect predictions on test set!")
        return
    
    # Analyze error patterns
    error_df = test_metadata[errors].copy()
    error_df['actual'] = y_test[errors].values
    error_df['predicted'] = y_pred[errors]
    error_df['Date'] = pd.to_datetime(error_df['Date'])
    
    tier_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    error_df['actual_tier'] = error_df['actual'].map(tier_names)
    error_df['predicted_tier'] = error_df['predicted'].map(tier_names)
    
    print("\nMisclassified Games:")

    print(f"{'Date':<12} {'Opponent':<8} {'Actual':<10} {'Predicted':<10}")
    
    for _, row in error_df.head(10).iterrows():
        print(f"{row['Date'].date()} {row['Opponent']:<8} {row['actual_tier']:<10} {row['predicted_tier']:<10}")
    
    # Error patterns
    print("\n\nError Patterns:")
    
    # By opponent
    if 'Opponent' in error_df.columns:
        opp_errors = error_df['Opponent'].value_counts().head(5)
        if len(opp_errors) > 0:
            print("\nOpponents with most errors:")
            for opp, count in opp_errors.items():
                print(f"  {opp}: {count} errors")
    
    return error_df


def create_visualizations(models, results, best_model, feature_imp_df, 
                         y_test, best_predictions, confusion_mat):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Figure 1: Model Comparison
    print("  Creating: model_comparison.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[m]['test_accuracy'] for m in model_names]
    f1_scores = [results[m]['test_f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#00A693')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='#FFA500')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Feature Importance 
    if feature_imp_df is not None:
        print("  Creating: feature_importance.png")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        top_features = feature_imp_df.head(10)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                       color='#00A693', alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top 10 Feature Importances - {best_model}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Confusion Matrix
    print("  Creating: confusion_matrix.png")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    tier_names = ['Low', 'Medium', 'High']
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=tier_names, yticklabels=tier_names,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=1, linecolor='black')
    
    ax.set_ylabel('Actual Tier', fontsize=12)
    ax.set_xlabel('Predicted Tier', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAll visualizations saved to results/models/")


def save_model_artifacts(models, results, best_model, feature_imp_df, config):
    # Save best model
    with open(f'{RESULTS_DIR}/best_model.pkl', 'wb') as f:
        pickle.dump(models[best_model], f)
    print(f"Saved: best_model.pkl ({best_model})")
    
    # Save all models
    with open(f'{RESULTS_DIR}/all_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"Saved: all_models.pkl")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'{RESULTS_DIR}/model_results.csv')
    print(f"Saved: model_results.csv")
    
    # Save feature importance
    if feature_imp_df is not None:
        feature_imp_df.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)
        print(f"Saved: feature_importance.csv")
    
    print(f"\nAll artifacts saved to: {RESULTS_DIR}/")


def main():
    print("NBA FAN ENGAGEMENT ANALYSIS - MODEL TRAINING")
    print()
    
    # Load data
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, train_metadata, test_metadata, config) = load_processed_data()
    
    feature_names = X_train.columns.tolist()
    
    # Step 1: Baseline
    baseline_acc, baseline_f1 = establish_baseline(y_train, y_test)
    
    # Track all results
    results = {
        'Baseline': {
            'test_accuracy': baseline_acc,
            'test_f1': baseline_f1
        }
    }
    
    models = {}
    
    # Step 2: Logistic Regression
    lr_model, lr_pred, lr_acc, lr_f1 = train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = {
        'test_accuracy': lr_acc,
        'test_f1': lr_f1
    }
    
    # Step 3: Random Forest
    rf_model, rf_pred, rf_acc, rf_f1 = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'test_accuracy': rf_acc,
        'test_f1': rf_f1
    }
    
    # Step 4: XGBoost
    xgb_model, xgb_pred, xgb_acc, xgb_f1 = train_xgboost(
        X_train, X_test, y_train, y_test
    )
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'test_accuracy': xgb_acc,
        'test_f1': xgb_f1
    }
    
    # Step 5: Compare models
    best_model_name, comparison_df = compare_models(results)
    best_model = models[best_model_name]
    
    # Determine which X_test to use for best model
    if best_model_name == 'Logistic Regression':
        X_test_best = X_test_scaled
    else:
        X_test_best = X_test
    
    # Step 6-9: Detailed analysis of best model
    best_pred, confusion_mat = detailed_evaluation(
        best_model, best_model_name, X_test_best, y_test, feature_names
    )
    
    feature_imp_df = feature_importance_analysis(
        best_model, best_model_name, feature_names
    )
    
    error_df = error_analysis(
        best_model, best_model_name, X_test_best, y_test, test_metadata
    )
    
    # generate_business_insights(
    #    best_model, best_model_name, feature_imp_df, y_test, best_pred, config
    #)
    
    # Step 10-11: Save outputs
    create_visualizations(
        models, results, best_model_name, feature_imp_df,
        y_test, best_pred, confusion_mat
    )
    
    save_model_artifacts(
        models, results, best_model_name, feature_imp_df, config
    )
    
    # Final summary
    print("MODEL TRAINING COMPLETE!")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"   Test Accuracy: {results[best_model_name]['test_accuracy']:.3f}")
    print(f"   Test F1-Score: {results[best_model_name]['test_f1']:.3f}")
    
    baseline_f1 = results['Baseline']['test_f1']
    improvement = ((results[best_model_name]['test_f1'] - baseline_f1) / baseline_f1) * 100
    print(f"   Improvement over baseline: {improvement:.1f}%")
    
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print(f"  • Trained models (.pkl)")
    print(f"  • Performance metrics (.csv)")
    print(f"  • Feature importance (.csv)")
    print(f"  • Visualizations (.png)")
    

if __name__ == "__main__":
    main()