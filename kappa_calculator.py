#!/usr/bin/env python3
"""
Cohen's Kappa Calculator for Inter-Rater Reliability
Usage: python kappa_calculator.py annotator1.csv annotator2.csv
"""

import csv
import sys
from collections import defaultdict

def calculate_cohens_kappa(annotations1, annotations2):
    """Calculate Cohen's kappa for two sets of annotations"""
    
    # Get common passages
    common_passages = set(annotations1.keys()) & set(annotations2.keys())
    
    if len(common_passages) < 10:
        print(f"Warning: Only {len(common_passages)} overlapping passages found")
    
    # Count agreements and disagreements
    agreements = 0
    total = len(common_passages)
    
    # Count category frequencies for expected agreement
    cat1_counts = defaultdict(int)
    cat2_counts = defaultdict(int)
    
    for passage_id in common_passages:
        cat1 = annotations1[passage_id]
        cat2 = annotations2[passage_id]
        
        cat1_counts[cat1] += 1
        cat2_counts[cat2] += 1
        
        if cat1 == cat2:
            agreements += 1
    
    # Observed agreement
    po = agreements / total if total > 0 else 0
    
    # Expected agreement by chance
    pe = 0
    all_categories = set(cat1_counts.keys()) | set(cat2_counts.keys())
    
    for category in all_categories:
        p1 = cat1_counts[category] / total
        p2 = cat2_counts[category] / total
        pe += p1 * p2
    
    # Cohen's kappa
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1
    
    return kappa, po, pe, total, agreements

def load_annotations(filename):
    """Load annotations from CSV file"""
    annotations = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            passage_id = f"{row['Novel']}_{row['Passage_ID']}"
            allusion_type = row['Allusion_Type']
            annotations[passage_id] = allusion_type
    
    return annotations

def main():
    if len(sys.argv) != 3:
        print("Usage: python kappa_calculator.py annotator1.csv annotator2.csv")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    try:
        annotations1 = load_annotations(file1)
        annotations2 = load_annotations(file2)
        
        print(f"Loaded {len(annotations1)} annotations from {file1}")
        print(f"Loaded {len(annotations2)} annotations from {file2}")
        
        kappa, po, pe, total, agreements = calculate_cohens_kappa(annotations1, annotations2)
        
        print(f"\n📊 Inter-Rater Reliability Results")
        print(f"=" * 40)
        print(f"Common passages: {total}")
        print(f"Agreements: {agreements}")
        print(f"Observed agreement (Po): {po:.3f}")
        print(f"Expected agreement (Pe): {pe:.3f}")
        print(f"Cohen's Kappa (κ): {kappa:.3f}")
        
        # Interpretation
        if kappa >= 0.81:
            interpretation = "Almost perfect agreement"
        elif kappa >= 0.61:
            interpretation = "Substantial agreement"
        elif kappa >= 0.41:
            interpretation = "Moderate agreement"
        elif kappa >= 0.21:
            interpretation = "Fair agreement"
        else:
            interpretation = "Poor agreement"
        
        print(f"Interpretation: {interpretation}")
        
        if kappa >= 0.70:
            print("✅ Target reliability (κ > 0.70) achieved!")
        else:
            print("❌ Below target reliability. Consider additional training.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    main()