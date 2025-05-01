# DL-Marriage-Prediction

A deep learning project that predicts characteristics of potential marriage partners based on personal attributes using data from wedding announcements.

## Project Overview

This project uses deep learning techniques to predict the characteristics of a person's potential marriage partner based on their own demographic and educational information. The model takes inputs such as:

- Gender
- Age group
- Education level
- School category
- Field of study

to predict similar attributes for a potential partner.

## Data

The dataset is derived from wedding announcements, primarily from The New York Times. It includes information about:

- Partner demographics (gender, age)
- Educational background (school category, education level)
- Professional information (field of study/occupation)
- How couples first met

## Project Structure
src/
├── EDA.ipynb # Exploratory Data Analysis
├── preprocess_data.ipynb # Data preprocessing
├── 0_archive_model_dev.ipynb # Archived model development
├── 1_model_dev.ipynb # Model development
├── 2_model_dev.ipynb
├── 3_model_dev.ipynb
└── evaluation_report.txt # Performance metrics


## Model

The project implements a neural network model that predicts multiple attributes:

1. Gender
2. Age group
3. School category (Ivy League, Top 30 Public, etc.)
4. Education level
5. Field of study/occupation

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



Model Performance

Prediction Target	Accuracy
Gender	~84%
Age group	~63%
School category	~51%
Education level	~46%
Field of study	~35%
Detailed evaluation metrics can be found in src/evaluation_report.txt.

Requirements

Python 3.x
PyTorch
Pandas
NumPy
Jupyter Notebook