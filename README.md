# Morrison Biblical Allusion Annotation Project

A digital humanities project for automated detection and human validation of biblical allusions in Toni Morrison's novels.

## 🎯 Project Overview

This project combines computational text analysis with human expertise to create a high-quality dataset of biblical allusions in Morrison's complete works. The dataset will be used to train machine learning models for automated literary analysis.

**Principal Investigator:** Felix Oke, PhD Student, UIUC School of Information Sciences  
**Advisor:** Ryan Cordell, Associate Professor, UIUC iSchool

## 📚 Corpus

All 8 Toni Morrison novels:
- The Bluest Eye (1970)
- Sula (1973) 
- Song of Solomon (1977)
- Tar Baby (1981)
- Beloved (1987)
- Jazz (1992)
- Paradise (1998)
- A Mercy (2008)

## 🔧 Tools & Methodology

### 1. Automated Detection
- **biblical_allusion_detector.py**: Rule-based detection using biblical text patterns, character names, and verse references
- **Output**: CSV/JSON files with detected allusions, confidence scores, and context

### 2. Human Validation
- **morrison-biblical-annotation-tool.html**: Web-based annotation interface
- **annotation-guide.html**: Comprehensive guidelines for annotators
- **Target**: 200+ validated annotations per annotator

### 3. Quality Assurance
- **kappa_calculator.py**: Inter-rater reliability testing (target: κ > 0.70)
- Expert validation by faculty with biblical literature expertise

## 🚀 Quick Start

### For Annotators

1. **Access the tool**: Open `morrison-biblical-annotation-tool.html` in your browser
2. **Read the guide**: Review `annotation-guide.html` for detailed instructions
3. **Load data**: Upload CSV/JSON files from the detector
4. **Annotate**: Validate and correct automated detections
5. **Export**: Save your annotations as CSV for submission

### For Researchers

1. **Run detection**:
   ```bash
   python biblical_allusion_detector.py
   ```

2. **Calculate reliability**:
   ```bash
   python kappa_calculator.py annotator1.csv annotator2.csv
   ```

## 📁 Repository Structure

```
morrison-biblical-annotation/
├── README.md                              # This file
├── biblical_allusion_detector.py          # Automated detection script
├── morrison-biblical-annotation-tool.html # Web annotation interface
├── annotation-guide.html                  # Annotator guidelines
├── annotator-deployment-guide.html        # Project management guide
├── kappa_calculator.py                    # Inter-rater reliability calculator
├── data/
│   ├── detector-results/                  # Automated detection outputs
│   │   ├── beloved_allusions.csv
│   │   ├── song_of_solomon_allusions.csv
│   │   └── morrison_allusions_combined.csv
│   └── annotations/                       # Human-validated annotations
│       ├── annotator1_annotations.csv
│       ├── annotator2_annotations.csv
│       └── final_validated_dataset.csv
└── docs/
    ├── methodology.md                     # Technical documentation
    └── results.md                         # Project outcomes
```

## 👥 Team & Roles

- **Primary Annotators** (2): Graduate students in Literature/DH
- **Expert Validator** (1): Faculty member with biblical literature expertise  
- **Quality Controller** (1): Graduate student for reliability testing

## 📊 Success Metrics

- **Quantity**: 1,000+ validated annotations across all novels
- **Quality**: Inter-rater agreement κ > 0.70
- **Coverage**: All 8 Morrison novels represented
- **Balance**: Mix of allusion types and confidence levels

## 🔬 Allusion Categories

1. **Direct Quote**: Exact or near-exact biblical text
2. **Paraphrase**: Biblical content in Morrison's words
3. **Character Reference**: Biblical names or figures
4. **Thematic Reference**: Biblical themes or concepts
5. **Structural Echo**: Biblical narrative patterns
6. **No Allusion**: False positive detection

## 📈 Timeline

- **Week 1**: Setup and annotator training
- **Weeks 2-4**: Primary annotation work
- **Week 5**: Validation and quality control
- **Week 6**: Final dataset preparation

## 🎓 Academic Context

This project supports Felix Oke's PhD dissertation research on computational approaches to biblical allusion detection in African American literature. The work builds on:

- Ted Underwood's research on digital humanities curriculum
- Ryan Cordell's BookLab methodologies
- UIUC's computational humanities initiatives

## 📄 Citation

If you use this dataset or methodology, please cite:

```
Oke, Felix. "Biblical Allusion Detection in Toni Morrison's Novels: 
A Digital Humanities Approach." University of Illinois at Urbana-Champaign, 2026.
```

## 📞 Contact

- **Felix Oke**: [email] - Project lead
- **Ryan Cordell**: [email] - Faculty advisor

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- University of Illinois School of Information Sciences
- UIUC University Library and Rare Book & Manuscript Library
- National Center for Supercomputing Applications (NCSA)
- Digital Humanities community at UIUC