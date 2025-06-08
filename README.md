# heart-disease-ml
A Machine Learning Portfolio for Coursework. Detecting Heart Disease using Tabular and Image Datasets
## Project Title: Heart Disease


## Dataset sources 
### Dataset 1: Indicators of Heart Disease 2020 (CDC)
  - Original Source: US Centers for Disease Control and Prevention (CDC)
    - Link: https://www.cdc.gov/brfss/annual_data/annual_2020.html
  - Retrieved from: Kaggle
    - Link: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data
  - License: Creative Commons (CC0) Public Domain

#### Sample Data
| AgeCategory | Sex   | BMI | Smoking | PhysicalActivity | HeartDisease |
|-------------|-------|-----|---------|------------------|--------------|
| 60-64       | Male  | 26  | No      | Yes             | No           |
| 18-24       | Female| 22  | Yes     | No              | Yes          |



### Dataset 2:
  - Original Source: Mendeley Data
    - Link: https://data.mendeley.com/datasets/gwbz3fsgp8/2
  - License: Creative Commons Attribution 4.0

#### Sample Data
  - ECG for abnormal heartbeat (HB)
  - ECG for normal heartbeat


## Data Preparation Pipeline 

- Data Loading - Access and load the datasets into a usable format.
- Data Cleaning - Handle missing values and standardize formats.
- Feature Engineering - Encoding categorical variables and normalizing numerical values.
- Feature Selection - Based on correlation with the target variable, HeartDisease.
To run the pipeline, execute the script data_preparation.py in the scripts folder.
- Image Script:
   - for image processing run script, image script that takes preprocessed images from their file and create training and testing numpy files with 2D arrays 

## Requirement Description
R2 - Data Analysis and Exploration
- Objective: To explore and analyze datasets, identifying relevant correlations and patterns.
- Location: R2 Data Analysis Notebook
- Model Outputs: Various health indicators predicting HeartDisease
- Model Inputs: 41 features reduced to 10, 20, and 41 selected features datasets.


 The models using Dataset 1 are predicting whether a person has heart disease using demographic and health markers from a survey.
 The models using Dataset 2 are predicting whether the ECG image is of someone with an abonormal heartbeat, having a heart attack, had a heart attack previously and a normal "healthy" individual.


## Repo Structure
The main folders in this file include:
  - data - *for datasets*
  - notebooks  - *Jupyter notebooks for analysis/modelling*
  - scripts - *Python scripts for preprocessing/ML*
  - documentation - *project documents and weekly updates, organized per week*


## Group Members
  - Ilham Ghori
  - Juhanah Madhiyyah
  - Kanana Muchiri
  - Raiqah Shameer
  - Sherif Fares
