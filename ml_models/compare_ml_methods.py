#!/usr/bin/env python3
"""
Compare Multiple ML Methods on Biblical Allusion Detection
Tests: Logistic Regression, SVM, Naive Bayes, XGBoost, RandomForest
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import time

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available - skipping")

# Load data
print("Loading data...")
train = pd.read_csv('C:/Users/felixo2/Desktop/2026/ALL/ml_data/train.csv')
val = pd.read_csv('C:/Users/felixo2/Desktop/2026/ALL/ml_data/validation.csv')
test = pd.read_csv('C:/Users/felixo2/Desktop/2026/ALL/ml_data/test.csv')

X_train, y_train = train['Text'], train['label']
X_val, y_val = val['Text'], val['label']
X_test, y_test = test['Text'], test['label']

# TF-IDF Vectorization
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM (Linear)': SVC(kernel='linear', class_weight='balanced', random_state=42),
    'Naive Bayes': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, 
                                          class_weight='balanced', random_state=42, n_jobs=-1)
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                     random_state=42, n_jobs=-1)

# Train and evaluate
results = []

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Model: {name}")
    print(f"{'='*80}")
    
    # Train
    print("Training...")
    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Predict on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val_tfidf)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    # Predict on test set
    print("Evaluating on test set...")
    y_test_pred = model.predict(X_test_tfidf)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Train Time (s)': train_time,
        'Val F1': val_f1,
        'Val Precision': val_precision,
        'Val Recall': val_recall,
        'Test F1': test_f1,
        'Test Precision': test_precision,
        'Test Recall': test_recall
    })
    
    print(f"\nValidation Results:")
    print(f"  F1-Score:  {val_f1:.4f} ({val_f1*100:.2f}%)")
    print(f"  Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
    print(f"  Recall:    {val_recall:.4f} ({val_recall*100:.2f}%)")
    
    print(f"\nTest Results:")
    print(f"  F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")

# Summary table
print("\n" + "="*80)
print("SUMMARY: MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test F1', ascending=False)

print("\nValidation Set Performance:")
print(results_df[['Model', 'Val F1', 'Val Precision', 'Val Recall']].to_string(index=False))

print("\nTest Set Performance:")
print(results_df[['Model', 'Test F1', 'Test Precision', 'Test Recall']].to_string(index=False))

print("\nTraining Time:")
print(results_df[['Model', 'Train Time (s)']].to_string(index=False))

# Save results
results_df.to_csv('C:/Users/felixo2/Desktop/2026/ALL/ml_models/method_comparison_results.csv', index=False)
print("\nResults saved to: ml_models/method_comparison_results.csv")

# Best model
best_model = results_df.iloc[0]
print(f"\nBEST MODEL: {best_model['Model']}")
print(f"   Test F1-Score: {best_model['Test F1']:.4f} ({best_model['Test F1']*100:.2f}%)")
print(f"   Training Time: {best_model['Train Time (s)']:.2f} seconds")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
