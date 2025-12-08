# ML Data Split Summary

## Overview
All annotated data has been merged and split for machine learning training with gold standards separated to prevent data leakage.

## Files Created

### 1. Complete Merged Dataset
**File**: `all_annotations_merged.csv`
- **Total samples**: 1,411
- **Annotators**: Batemmy (1,000), JM (201), Temitayo (210)
- **Includes**: All annotations with duplicates (multiple annotators per passage)
- **Use**: Reference dataset with all annotation perspectives

### 2. Training Set
**File**: `train.csv`
- **Samples**: 649 (69.9% of main dataset)
- **Positive class**: 540 (83.2%)
- **Negative class**: 109 (16.8%)
- **Class ratio**: 4.95:1
- **Use**: Model training

### 3. Validation Set
**File**: `validation.csv`
- **Samples**: 139 (15.0% of main dataset)
- **Positive class**: 116 (83.5%)
- **Negative class**: 23 (16.5%)
- **Class ratio**: 5.04:1
- **Use**: Hyperparameter tuning and model selection

### 4. Test Set
**File**: `test.csv`
- **Samples**: 140 (15.1% of main dataset)
- **Positive class**: 116 (82.9%)
- **Negative class**: 24 (17.1%)
- **Class ratio**: 4.83:1
- **Use**: Final model evaluation

### 5. Gold Standard 1 (Perfect Agreement)
**File**: `gold_standard_perfect.csv`
- **Samples**: 180 passages
- **Annotators**: Batemmy & JM
- **Agreement**: Cohen's Kappa = 1.0 (perfect)
- **Positive class**: 146 (81.1%)
- **Negative class**: 34 (18.9%)
- **Class ratio**: 4.29:1
- **Use**: High-quality benchmark for primary model evaluation

### 6. Gold Standard 2 (All 3 Annotators)
**File**: `gold_standard_all3.csv`
- **Samples**: 107 passages
- **Annotators**: Batemmy, JM, Temitayo
- **Agreement**: Fleiss' Kappa = 0.0614 (slight)
- **Positive class**: 91 (85.0%)
- **Negative class**: 16 (15.0%)
- **Class ratio**: 5.69:1
- **Use**: Robustness testing with diverse annotation perspectives

## Data Split Strategy

### Main Dataset Split (70-15-15)
1. **Remove gold standards first** to prevent data leakage
2. **Main dataset**: 928 unique passages (after removing 180 gold standard passages)
3. **Stratified split**: Maintains class ratio across all splits
4. **Random seed**: 42 (for reproducibility)

### Gold Standards (Separate)
- **Gold Standard 1**: 180 passages with perfect agreement (Batemmy & JM)
- **Gold Standard 2**: 107 passages with all 3 annotators
- **No overlap** with train/val/test sets

## Class Distribution Summary

| Dataset | Total | Positive | Negative | Ratio |
|---------|-------|----------|----------|-------|
| **Training** | 649 | 540 (83.2%) | 109 (16.8%) | 4.95:1 |
| **Validation** | 139 | 116 (83.5%) | 23 (16.5%) | 5.04:1 |
| **Test** | 140 | 116 (82.9%) | 24 (17.1%) | 4.83:1 |
| **Gold 1 (Perfect)** | 180 | 146 (81.1%) | 34 (18.9%) | 4.29:1 |
| **Gold 2 (All 3)** | 107 | 91 (85.0%) | 16 (15.0%) | 5.69:1 |
| **TOTAL** | 1,215* | 1,009 | 206 | 4.90:1 |

*Note: Total is 1,215 unique passages (not 1,411) because duplicates were removed after identifying gold standards

## Key Features

### Data Integrity
✅ **No data leakage**: Gold standards completely separated from train/val/test
✅ **Stratified splits**: Class ratios maintained across all splits
✅ **Reproducible**: Random seed = 42 for consistent splits
✅ **Unique passages**: Duplicates removed (kept first occurrence)

### Evaluation Strategy
1. **Development**: Use validation set for hyperparameter tuning
2. **Primary evaluation**: Test on Gold Standard 1 (180 perfect agreement passages)
3. **Robustness testing**: Test on Gold Standard 2 (107 all-annotator passages)
4. **Final evaluation**: Test on standard test set (140 passages)

### Class Balance
- **Consistent ratios**: ~4.5-5:1 across all splits
- **Good for ML**: Sufficient minority class samples (16-24 per split)
- **Stratification**: Ensures representative samples in each split

## Usage Instructions

### Loading Data
```python
import pandas as pd

# Load training data
train = pd.read_csv('ml_data/train.csv')
val = pd.read_csv('ml_data/validation.csv')
test = pd.read_csv('ml_data/test.csv')

# Load gold standards
gold_perfect = pd.read_csv('ml_data/gold_standard_perfect.csv')
gold_all3 = pd.read_csv('ml_data/gold_standard_all3.csv')

# Extract features and labels
X_train = train['Text']
y_train = train['label']
```

### Evaluation Workflow
```python
# 1. Train on training set
model.fit(X_train, y_train)

# 2. Tune on validation set
best_params = tune_hyperparameters(model, X_val, y_val)

# 3. Evaluate on test set
test_score = model.score(X_test, y_test)

# 4. Benchmark on gold standards
gold1_score = model.score(X_gold1, y_gold1)  # High-quality benchmark
gold2_score = model.score(X_gold2, y_gold2)  # Robustness test
```

## Important Notes

### Gold Standard Separation
⚠️ **CRITICAL**: Gold standard passages are **NOT** in train/val/test sets
- This prevents data leakage and ensures unbiased evaluation
- Gold standards serve as independent benchmarks
- Use gold standards ONLY for final evaluation, never for training

### Duplicate Handling
- Original dataset: 1,411 annotations (with duplicates from multiple annotators)
- Unique passages: 1,215 (after removing duplicates)
- Strategy: Kept first occurrence when multiple annotators annotated same passage

### Class Imbalance
- Ratio: ~4.5-5:1 (positive:negative)
- Recommendation: Use `class_weight='balanced'` in models
- Sufficient minority samples: 16-34 negatives per split

## ML Training Results ✅

**Model**: RandomForestClassifier with TF-IDF features
**Status**: Training complete

### Performance Summary

| Dataset | F1-Score | Precision | Recall | Cohen's Kappa |
|---------|----------|-----------|--------|---------------|
| Validation | 97.82% | 99.12% | 96.55% | 0.8762 |
| Test | 96.92% | 99.10% | 94.83% | 0.8374 |
| Gold Std 1 | 95.92% | 95.27% | 96.58% | 0.7774 |
| Gold Std 2 | 95.56% | 96.63% | 94.51% | 0.7204 |

**Key Achievement**: F1-Score > 95% on all test sets

See `ml_models/TRAINING_RESULTS.md` for complete details.

## File Locations

All files saved to: `C:\Users\felixo2\Desktop\2026\ALL\ml_data\`

```
ml_data/
├── all_annotations_merged.csv      (1,411 samples - complete dataset)
├── train.csv                        (649 samples - 70%)
├── validation.csv                   (139 samples - 15%)
├── test.csv                         (140 samples - 15%)
├── gold_standard_perfect.csv        (180 samples - Kappa=1.0)
├── gold_standard_all3.csv           (107 samples - All 3 annotators)
└── DATA_SPLIT_SUMMARY.md            (This file)
```

---

**Status**: ML Training Complete ✅ | Model Deployed 🚀
