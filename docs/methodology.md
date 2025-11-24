# Methodology: Automated Detection of Biblical Allusions in Toni Morrison's Novels

**Author:** Felix Oke, PhD Student  
**Institution:** School of Information Sciences, University of Illinois at Urbana-Champaign  
**Advisor:** Ryan Cordell

## Introduction and Motivation

Allusion as a form of reference has been studied as an important interpretive device in literature. It's often challenging to systematically identify and analyze allusions at large scale due to their varied forms of representation without relying on computational methods (Bamman & Crane, 2011). While computational approaches have been successfully applied to classical texts (Coffee et al., 2013; Forstall et al., 2015), African American literature remains understudied in this domain.

This project aims to develop text mining methods for automated detection of biblical allusions in Toni Morrison's complete corpus of eight novels. These novels represent different periods and themes in Morrison's work, from *The Bluest Eye* (1970) with its foundational exploration of beauty and identity, to *A Mercy* (2008) with its historical examination of early American slavery. Using datasets from HathiTrust Digital Library, we combine rule-based techniques with named entity recognition and information retrieval approaches to identify biblical allusions with higher accuracy.

With human annotation in the loop, we assess model performance to create a foundation resource for allusion detection in African American literature. This research contributes to studies on intertextuality more broadly (Kristeva, 1980) and to computational interpretation of literary texts specifically (Manjavacas et al., 2019).

## Research Questions

1. What types and frequencies of biblical allusions appear across Morrison's eight novels?
2. How can we systematically annotate biblical allusions to create reliable training data for classification models?
3. How do explicit vs. implicit allusions distribute across Morrison's literary career?
4. What computational approaches best capture the range of allusive strategies in contemporary African American literature?

## Related Work

Computational allusion detection has evolved from simple string matching to sophisticated machine learning approaches (Manjavacas et al., 2019). Early work by Bamman and Crane (2011) established foundational principles for textual allusion logic, while Coffee et al. (2013) demonstrated machine learning applications to classical literature. Recent advances include enhanced n-gram matching (Forstall et al., 2015) and transformation-aware text reuse detection (Moritz et al., 2016).

However, existing approaches primarily target classical texts with well-documented intertextual relationships. Modern literary works, particularly those by marginalized authors, present different challenges including copyright restrictions, varied allusive strategies, and limited computational resources. Our work extends these methodologies to contemporary African American literature while addressing practical constraints of working with copyrighted materials.

## Dataset

### Primary Corpus
We obtained Morrison's eight novels from HathiTrust Digital Library's Extended Features API (Downie et al., 2014), which provides computational access while respecting copyright restrictions:

- *The Bluest Eye* (1970) – HathiTrust ID: uc1.32106018657251
- *Sula* (1973) – HathiTrust ID: uc1.32106019072633  
- *Song of Solomon* (1977) – HathiTrust ID: mdp.39015032749130
- *Tar Baby* (1981) – HathiTrust ID: uc1.32106005767956
- *Beloved* (1987) – HathiTrust ID: mdp.49015003142743
- *Jazz* (1992) – HathiTrust ID: ien.35556029664190
- *Paradise* (1998) – HathiTrust ID: mdp.39015066087613
- *A Mercy* (2008) – HathiTrust ID: mdp.39076002787351

### Biblical Reference Corpus
Our biblical reference corpus consists of the King James Version Bible, segmented into verses and indexed with book, chapter, and verse identifiers. This public-domain text serves as the source for identifying biblical allusions in Morrison's novels.

### Allusion Classification System
Biblical allusions are classified into six types:
1. **Direct Quote**: Exact or near-exact biblical text
2. **Paraphrase**: Biblical content in Morrison's words
3. **Character Reference**: Biblical names or figures
4. **Thematic Reference**: Biblical themes or concepts
5. **Structural Echo**: Biblical narrative patterns
6. **No Biblical Reference**: False positive detection

Six functional categories capture the literary purpose:
1. **Characterization**: Developing character traits or backgrounds
2. **Thematic Development**: Advancing central themes
3. **Narrative Structure**: Organizing plot elements
4. **Cultural Commentary**: Critiquing social or religious norms
5. **Ironic Contrast**: Creating tension through juxtaposition
6. **Spiritual Dimension**: Exploring religious or metaphysical themes

## Computational Detection Pipeline

### Phase 1: Rule-Based Detection
- **Exact String Matching**: Direct quotations from biblical text
- **Named Entity Recognition**: Biblical names using enhanced spaCy NER with custom biblical dictionaries
- **Pattern Matching**: Common biblical phrases and citation formats (e.g., "Romans 9:25", "First Corinthians")
- **Keyword Detection**: Thematic concepts (redemption, sacrifice, exodus)

### Phase 2: Information Retrieval Methods
- **N-gram Overlap Analysis**: 3-5 gram comparison between novel segments and biblical verses
- **TF-IDF Similarity**: Lexical overlap detection using term frequency-inverse document frequency
- **Fuzzy String Matching**: Paraphrases and variations using edit distance algorithms
- **Semantic Similarity**: Pre-trained word embeddings for conceptual relationships

### Phase 3: Ensemble Classification
- **Weighted Combination**: Multiple detection methods with optimized weights
- **Confidence Scoring**: Evidence aggregation from multiple sources
- **Threshold Optimization**: Using annotated ground truth for parameter tuning

## Data Preprocessing

### Text Cleaning
- Remove non-textual elements (page numbers, headers, OCR artifacts)
- Verify OCR quality and correct systematic errors
- Preserve original punctuation and capitalization for proper noun identification

### Segmentation and Normalization
- Segment Bible into verses with book/chapter/verse indexing
- Create dual versions: normalized (lowercased) and original case
- Tokenization preserving sentence and paragraph boundaries

### Annotation Protocol
Manual labeling of biblical allusions using custom web-based annotation tool:
- **Tool**: `morrison-biblical-annotation-tool.html`
- **Guidelines**: Comprehensive annotation guide with examples
- **Quality Control**: Inter-annotator reliability testing (target: κ > 0.70)
- **Target**: 200+ annotations per annotator across all novels

## Human-in-the-Loop Validation

### Annotation Workflow
1. **Automated Detection**: Generate candidate allusions using computational pipeline
2. **Human Validation**: Annotators review and correct automated detections
3. **Expert Review**: Faculty validation of complex or ambiguous cases
4. **Quality Assurance**: Inter-rater reliability testing and error analysis

### Annotation Team
- **Primary Annotators** (2): Graduate students in Literature/Digital Humanities
- **Expert Validator** (1): Faculty member with biblical literature expertise
- **Quality Controller** (1): Graduate student for reliability testing

## Evaluation Metrics

### Performance Measures
- **Precision, Recall, F1-score**: For each method and allusion type
- **Inter-annotator Agreement**: Cohen's kappa coefficient (κ > 0.70 target)
- **Temporal Analysis**: Detection accuracy across Morrison's career
- **Comparative Analysis**: Baseline vs. advanced methods

### Error Analysis
- **False Positive Analysis**: Why non-allusions were detected (common biblical vocabulary, coincidental matches)
- **False Negative Analysis**: Understanding missed allusions (implicit references, transformed language)
- **Method Comparison**: Which approaches perform best for different allusion types
- **Iterative Improvement**: Using error patterns to refine detection algorithms

## Technical Implementation

### Tools and Libraries
- **Python 3.8+**: Primary programming language
- **spaCy**: Named entity recognition and text processing
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **pandas/numpy**: Data manipulation and analysis
- **Custom Web Tool**: HTML/JavaScript annotation interface

### Data Management
- **Version Control**: Git repository for code and documentation
- **Data Storage**: CSV/JSON formats for interoperability
- **Backup Strategy**: GitHub repository with regular commits
- **Privacy**: Private repository during development, public upon publication

## Expected Outcomes

### Deliverables
1. **Annotated Dataset**: 1,000+ validated biblical allusions across Morrison's novels
2. **Detection System**: Open-source Python toolkit for biblical allusion detection
3. **Evaluation Framework**: Metrics and benchmarks for allusion detection systems
4. **Scholarly Publication**: Methodology and findings for digital humanities community

### Contributions
- **Methodological**: Computational approaches for modern literary allusion detection
- **Empirical**: Systematic analysis of biblical allusions in Morrison's complete works
- **Technical**: Open-source tools for literary text analysis
- **Pedagogical**: Resources for digital humanities education and research

## Timeline

- **Weeks 1-2**: Data acquisition and preprocessing
- **Weeks 3-4**: Computational pipeline development and testing
- **Weeks 5-8**: Human annotation and validation
- **Weeks 9-10**: Model evaluation and refinement
- **Weeks 11-12**: Documentation and publication preparation

## Bibliography

- Bamman, David, and Gregory Crane. "The Logic and Discovery of Textual Allusion." *Proceedings of the 2011 Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities*, 2011, pp. 1-9.

- Coffee, Neil, et al. "Modelling the Interpretation of Literary Allusion with Machine Learning Techniques." *Digital Scholarship in the Humanities*, vol. 28, no. 4, 2013, pp. 692-712.

- Downie, J. Stephen, et al. "The HathiTrust Research Center: Using Large-Scale Analytics to Support Digital Humanities Scholarship." *Digital Humanities 2014*, 2014.

- Forstall, Christopher, et al. "Modeling the Scholars: Detecting Intertextuality through Enhanced Word-Level N-Gram Matching." *Digital Scholarship in the Humanities*, vol. 30, no. 4, 2015, pp. 503-515.

- Kristeva, Julia. *Desire in Language: A Semiotic Approach to Literature and Art*. Columbia University Press, 1980.

- Manjavacas, Enrique, Brian Long, and Mike Kestemont. "On the Feasibility of Automated Detection of Allusive Text Reuse." *Proceedings of the 3rd Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature*, Minneapolis, USA, 2019, pp. 104-114.

- Moritz, Maria, et al. "Transformation-Aware Text Reuse Detection." *Digital Humanities 2016*, 2016.

## HathiTrust Catalog Records

- [The Bluest Eye](https://catalog.hathitrust.org/Record/005967881)
- [Sula](https://catalog.hathitrust.org/Record/004737103)
- [Song of Solomon](https://catalog.hathitrust.org/Record/002473454)
- [Tar Baby](https://catalog.hathitrust.org/Record/000169417)
- [Beloved](https://catalog.hathitrust.org/Record/000870183)
- [Jazz](https://catalog.hathitrust.org/Record/008325775)
- [Paradise](https://catalog.hathitrust.org/Record/003959081)
- [A Mercy](https://catalog.hathitrust.org/Record/005898814)