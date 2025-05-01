# DL-Marriage-Prediction

**Authors**: Mingrui Chen, Yanmi Yu, Yixiao Zhang, Yicen Ye  
**Institution**: Brown University, CSCI 1470 Deep Learning  
**Date**: April 30, 2025


## Project Overview

This study analyzes how demographic traits influence partner selection by predicting ideal match attributes from individual profiles. Using the TabTransformer architecture on NYT wedding data, we model relationships between features like age, education, and occupation through multi-label classification.

Marriage patterns reveal societal structures and biases. Our work extends sociological research while enabling practical applications—from bias auditing to improving recommendation systems—and advances Transformer methods for social data analysis.

The model takes inputs such as:

- Gender
- Age group
- Education level(Undergraduate VS graduate)
- School category
- Field of occupation
- level of the job

to predict similar attributes for a potential partner.

## Data

### Source
The dataset is derived from **9,160 New York Times wedding announcements** (2013–2023).
   - Used NYT Article Search API
   - Search parameters:
     - Keywords: "wedding", "vow", "marriage"
     - Metadata subsection: "Fashion & Style"
     - Date range: January 2013 - December 2023

### Dataset Characteristics
| Feature Type       | Categories/Values | Processing Method |
|--------------------|-------------------|-------------------|
| **Demographics**   | Gender, Age       | GPT 3.5 batch Direct extraction |
| **Education**      | 5 tiers           | Keyword matching  |
| **Occupation**     | 25 fields         | GPT-40 mini clustering  |
| **Relationships**  | Meeting context   | eyword matching  |


## Model Architecture

Our **ImprovedTabTransformer** architecture combines feature embeddings with transformer-based processing:

Key components:
- **Feature Embeddings**: Each categorical feature embedded into shared space with LayerNorm
- **Transformer Core**:
  - 4-layer encoder with 8 attention heads
  - 256-dimensional feedforward networks
- **Prediction Heads**: Task-specific MLPs with GELU activation

## Example Usage

```python
# Sample input for prediction
sample_row = [
    'Female',                           # partner_gender
    '30-34',                            # partner_age_bin  
    'Ivy League',                       # partner_school_category
    'S4',                               # partner_level_id
    'Business and Financial Occupations', # partner_field
    '1'                                 # partner_is_graduate
]

# Get prediction
predictions = predict_partner(sample_row, model, label_encoders)

# Display results
print("\n=== Predicted Partner Profile ===")
print(f"Gender:         {predictions['target_gender']}")
print(f"Age Group:      {predictions['target_age_bin']}")
print(f"School Category: {predictions['target_school_category']}") 
print(f"Education Level: {predictions['target_level_id']}")
print(f"Field of Study:  {predictions['target_field']}")


## Requirements

Python 3.x
PyTorch
Pandas
NumPy
Jupyter Notebook

## Acknowledgements
[1] Raw dataset collected by Dr. Zhenchao Qian, Dr. Guixing Wei and Yanmi Yu(Brown University).
[2] Huang, Xin, et al. "TabTransformer: Tabular Data Modeling Using Contextual Embeddings." arXiv, 11 Dec. 2020, https://arxiv.org/abs/2012.06678
