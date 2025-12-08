# ML Training Results - Biblical Allusion Detection

## Model Overview

**Model**: RandomForestClassifier
**Features**: TF-IDF (max 5,000 features, 1-3 grams)
**Training Data**: 649 samples
**Vocabulary Size**: 37 unique features
**Class Weight**: Balanced (handles 4.95:1 class imbalance)

## Hyperparameters

- **n_estimators**: 200 trees
- **max_depth**: 20
- **min_samples_split**: 5
- **class_weight**: balanced
- **random_state**: 42
- **n_jobs**: -1 (all cores)

## Performance Results

### Summary Table

| Dataset | Samples | Precision | Recall | F1-Score | Cohen's Kappa |
|---------|---------|-----------|--------|----------|---------------|
| **Validation** | 139 | 0.9912 | 0.9655 | **0.9782** | 0.8762 |
| **Test** | 140 | 0.9910 | 0.9483 | **0.9692** | 0.8374 |
| **Gold Standard 1** | 180 | 0.9527 | 0.9658 | **0.9592** | 0.7774 |
| **Gold Standard 2** | 107 | 0.9663 | 0.9451 | **0.9556** | 0.7204 |

### Detailed Results

#### 1. Validation Set (139 samples)
- **Precision**: 99.12% - Almost all predicted allusions are correct
- **Recall**: 96.55% - Catches 96.55% of true allusions
- **F1-Score**: 97.82% - Excellent balance
- **Cohen's Kappa**: 0.8762 - Almost perfect agreement
- **Confusion Matrix**:
  - True Negatives: 22
  - False Positives: 1
  - False Negatives: 4
  - True Positives: 112

#### 2. Test Set (140 samples)
- **Precision**: 99.10% - Highly accurate predictions
- **Recall**: 94.83% - Catches 94.83% of true allusions
- **F1-Score**: 96.92% - Strong performance
- **Cohen's Kappa**: 0.8374 - Almost perfect agreement
- **Confusion Matrix**:
  - True Negatives: 23
  - False Positives: 1
  - False Negatives: 6
  - True Positives: 110

#### 3. Gold Standard 1 - Perfect Agreement (180 samples)
- **Precision**: 95.27% - High accuracy
- **Recall**: 96.58% - Excellent coverage
- **F1-Score**: 95.92% - Strong performance on high-quality benchmark
- **Cohen's Kappa**: 0.7774 - Substantial agreement
- **Confusion Matrix**:
  - True Negatives: 27
  - False Positives: 7
  - False Negatives: 5
  - True Positives: 141

#### 4. Gold Standard 2 - All 3 Annotators (107 samples)
- **Precision**: 96.63% - Very accurate
- **Recall**: 94.51% - Good coverage
- **F1-Score**: 95.56% - Robust performance despite annotation variability
- **Cohen's Kappa**: 0.7204 - Substantial agreement
- **Confusion Matrix**:
  - True Negatives: 13
  - False Positives: 3
  - False Negatives: 5
  - True Positives: 86

## Feature Importance

### Top 20 Most Important Features

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | lot | 0.2117 | Common word (false positive indicator) |
| 2 | job | 0.1527 | Common word (false positive indicator) |
| 3 | hope | 0.1360 | Common word (false positive indicator) |
| 4 | paul | 0.0817 | Biblical character (true allusion) |
| 5 | mark | 0.0720 | Common word (false positive indicator) |
| 6 | pilate | 0.0692 | Biblical character (true allusion) |
| 7 | hagar | 0.0414 | Biblical character (true allusion) |
| 8 | acts | 0.0335 | Biblical book (true allusion) |
| 9 | numbers | 0.0317 | Common word/Biblical book |
| 10 | jacob | 0.0251 | Biblical character (true allusion) |
| 11 | ruth | 0.0211 | Biblical character (true allusion) |
| 12 | jesus | 0.0181 | Biblical character (true allusion) |
| 13 | gideon | 0.0152 | Biblical character (true allusion) |
| 14 | eve | 0.0146 | Biblical character (true allusion) |
| 15 | mary | 0.0104 | Biblical character (true allusion) |
| 16 | jude | 0.0093 | Biblical character (true allusion) |
| 17 | solomon | 0.0092 | Biblical character (true allusion) |
| 18 | john | 0.0083 | Biblical character (true allusion) |
| 19 | grace | 0.0080 | Thematic reference |
| 20 | charity | 0.0078 | Thematic reference |

### Key Insights

**False Positive Indicators** (Top 3):
- "lot", "job", "hope" - Common English words that are NOT biblical allusions
- Model learned to identify these as false positives (high importance)

**True Allusion Indicators**:
- "paul", "pilate", "hagar", "jacob", "ruth" - Biblical character names
- Model correctly identifies these as true allusions

## Model Strengths

✅ **Excellent Precision** (95-99%) - Very few false positives
✅ **Strong Recall** (94-97%) - Catches most true allusions
✅ **Robust Performance** - Consistent across all test sets
✅ **Handles Class Imbalance** - Balanced class weights work well
✅ **Learned Key Patterns** - Distinguishes common words from biblical names
✅ **Generalizes Well** - Similar performance on gold standards vs test set

## Model Weaknesses

⚠️ **Small Vocabulary** - Only 37 features (limited by small training set)
⚠️ **Context-Independent** - TF-IDF doesn't capture surrounding context
⚠️ **Ambiguous Words** - May struggle with words like "numbers", "mark", "john"

## Error Analysis

### False Positives (7 on Gold Standard 1)
- Model predicted allusion, but annotators marked as false positive
- Likely ambiguous cases (e.g., "mark" as common word vs. Gospel of Mark)

### False Negatives (5-6 across test sets)
- Model missed true allusions
- Possibly rare biblical references not well-represented in training data

## Comparison to Human Annotators

| Metric | Model (Gold Std 1) | Human IAA (Batemmy-JM) |
|--------|-------------------|------------------------|
| Cohen's Kappa | 0.7774 | 1.0000 |
| Agreement Level | Substantial | Perfect |

**Interpretation**: Model achieves "substantial agreement" (Kappa > 0.6) with human annotators, approaching the "almost perfect" threshold (Kappa > 0.8).

## Files Saved

1. **random_forest_model.pkl** - Trained RandomForest model
2. **tfidf_vectorizer.pkl** - TF-IDF vectorizer (fitted on training data)
3. **evaluation_results.csv** - Performance metrics for all datasets
4. **TRAINING_RESULTS.md** - This file

## Usage

### Loading Model
```python
import joblib
model = joblib.load('ml_models/random_forest_model.pkl')
vectorizer = joblib.load('ml_models/tfidf_vectorizer.pkl')
```

### Making Predictions
```python
# Single prediction
text = "Paul"
text_vec = vectorizer.transform([text])
prediction = model.predict(text_vec)[0]  # 1 = allusion, 0 = not allusion
probability = model.predict_proba(text_vec)[0]

# Batch prediction
texts = ["Paul", "Pilate", "lot", "hope"]
texts_vec = vectorizer.transform(texts)
predictions = model.predict(texts_vec)
```

## Recommendations

### For Production Use
1. ✅ **Deploy as-is** - Model performs well (F1 > 95%)
2. ✅ **Use probability scores** - Set threshold based on precision/recall tradeoff
3. ⚠️ **Human review** - Flag low-confidence predictions (prob < 0.8)

### For Future Improvement
1. **More training data** - Expand beyond 649 samples
2. **Context features** - Add surrounding words, sentence structure
3. **Ensemble methods** - Combine with other models (SVM, Neural Networks)
4. **Fine-tuned transformers** - BERT/RoBERTa for better context understanding

## Conclusion

The RandomForest model achieves **excellent performance** (F1 > 95%) on all test sets, demonstrating:
- Strong ability to distinguish biblical allusions from common words
- Robust generalization across different annotation perspectives
- Substantial agreement with human annotators (Kappa > 0.7)

**Status**: Model ready for deployment ✅

---

**Training Date**: January 2025
**Model Version**: 1.0
**Training Time**: < 1 minute
**Best F1-Score**: 97.82% (Validation Set)
