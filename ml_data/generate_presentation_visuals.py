import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Data
datasets = ['Training', 'Validation', 'Test', 'Gold Std 1', 'Gold Std 2']
samples = [649, 139, 140, 180, 107]
positive = [540, 116, 116, 146, 91]
negative = [109, 23, 24, 34, 16]
f1_scores = [None, 97.82, 96.92, 95.92, 95.56]

# 1. Dataset Size Comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(datasets, samples, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Dataset Size Distribution', fontsize=14, fontweight='bold', pad=20)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_dataset_sizes.png', bbox_inches='tight')
plt.close()

# 2. Class Distribution Stacked Bar
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
p1 = ax.bar(x, positive, label='Positive', color='#27AE60')
p2 = ax.bar(x, negative, bottom=positive, label='Negative', color='#E74C3C')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution Across Datasets', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=11)
for i, (pos, neg) in enumerate(zip(positive, negative)):
    ax.text(i, pos/2, f'{pos}', ha='center', va='center', fontweight='bold', color='white')
    ax.text(i, pos + neg/2, f'{neg}', ha='center', va='center', fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_class_distribution.png', bbox_inches='tight')
plt.close()

# 3. Class Ratio Comparison
ratios = [4.95, 5.04, 4.83, 4.29, 5.69]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(datasets, ratios, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax.set_xlabel('Positive:Negative Ratio', fontsize=12, fontweight='bold')
ax.set_title('Class Imbalance Ratio by Dataset', fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=4.90, color='red', linestyle='--', linewidth=2, label='Overall Avg (4.90:1)')
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    ax.text(ratio + 0.1, bar.get_y() + bar.get_height()/2, f'{ratio:.2f}:1', va='center', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_class_ratios.png', bbox_inches='tight')
plt.close()

# 4. Model Performance Comparison
test_datasets = ['Validation', 'Test', 'Gold Std 1', 'Gold Std 2']
f1_test = [97.82, 96.92, 95.92, 95.56]
precision = [99.12, 99.10, 95.27, 96.63]
recall = [96.55, 94.83, 96.58, 94.51]

x = np.arange(len(test_datasets))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, f1_test, width, label='F1-Score', color='#3498DB')
ax.bar(x, precision, width, label='Precision', color='#2ECC71')
ax.bar(x + width, recall, width, label='Recall', color='#F39C12')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Across Test Sets', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(test_datasets, rotation=15, ha='right')
ax.legend(loc='lower left', fontsize=11)
ax.set_ylim([90, 100])
for i in range(len(test_datasets)):
    ax.text(i - width, f1_test[i] + 0.3, f'{f1_test[i]:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i, precision[i] + 0.3, f'{precision[i]:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width, recall[i] + 0.3, f'{recall[i]:.1f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_model_performance.png', bbox_inches='tight')
plt.close()

# 5. Data Split Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))
sizes = [649, 139, 140, 287]
labels = ['Training\n(649 samples)', 'Validation\n(139 samples)', 'Test\n(140 samples)', 'Gold Standards\n(287 samples)']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
explode = (0.05, 0.05, 0.05, 0.1)
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('Data Split Distribution (1,215 Unique Passages)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_data_split_pie.png', bbox_inches='tight')
plt.close()

# 6. Annotator Contribution
annotators = ['Batemmy', 'JM', 'Temitayo']
contributions = [1000, 201, 210]
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(annotators, contributions, color=['#E74C3C', '#3498DB', '#F39C12'])
ax.set_ylabel('Number of Annotations', fontsize=12, fontweight='bold')
ax.set_title('Annotator Contributions', fontsize=14, fontweight='bold', pad=20)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_annotator_contributions.png', bbox_inches='tight')
plt.close()

# 7. Cohen's Kappa Comparison
kappa_datasets = ['Validation', 'Test', 'Gold Std 1', 'Gold Std 2']
kappa_scores = [0.8762, 0.8374, 0.7774, 0.7204]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(kappa_datasets, kappa_scores, color=['#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax.set_xlabel("Cohen's Kappa", fontsize=12, fontweight='bold')
ax.set_title("Inter-Rater Agreement (Cohen's Kappa)", fontsize=14, fontweight='bold', pad=20)
ax.set_xlim([0, 1])
for i, (bar, kappa) in enumerate(zip(bars, kappa_scores)):
    ax.text(kappa + 0.02, bar.get_y() + bar.get_height()/2, f'{kappa:.4f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/felixo2/Desktop/2026/ALL/ml_data/chart_kappa_scores.png', bbox_inches='tight')
plt.close()

print("All 7 presentation charts generated successfully!")
print("\nGenerated files:")
print("1. chart_dataset_sizes.png - Dataset size comparison")
print("2. chart_class_distribution.png - Class distribution stacked bars")
print("3. chart_class_ratios.png - Class imbalance ratios")
print("4. chart_model_performance.png - Model performance metrics")
print("5. chart_data_split_pie.png - Data split pie chart")
print("6. chart_annotator_contributions.png - Annotator contributions")
print("7. chart_kappa_scores.png - Cohen's Kappa comparison")
