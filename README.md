# Stroke Prediction & Market Basket Analysis

This project presents an end-to-end data mining solution developed in Python, featuring a Streamlit dashboard that combines clinical risk assessment with transactional pattern recognition.

## Project Overview

This terminal project addresses two distinct data mining tasks:

**1.Stroke Prediction**: Building a classifier pipeline to detect stroke risk using the Healthcare Stroke Dataset.


**2.Market Basket Analysis**: Uncovering purchasing patterns in the Groceries dataset using association rule mining.



## 1. Stroke Prediction Pipeline

Given the extreme class imbalance (only 4.9% of patients in the dataset had a stroke), this project prioritizes **Recall** over Accuracy to ensure high-risk cases are flagged.


**Models Trained**: Decision Tree, RIPPER (Rule-based), SVM (RBF Kernel), and Gaussian Naive Bayes.

 
**Key Findings**: The **Decision Tree and SVM** models achieved the best recall of 0.76. While RIPPER achieved high accuracy (0.949), it failed to flag stroke cases, highlighting the danger of using accuracy as a metric on imbalanced data.


 
**Recommendation**: The **Decision Tree** is the preferred model, balancing high recall with full clinical interpretability.



## 2. Market Basket Analysis

This module evaluates the performance of association rule mining algorithms on the Groceries dataset.


**Algorithms**: Apriori vs. FP-Growth.


 
**Key Findings**: Both algorithms produced identical frequent itemsets (122) and association rules (6).



**Performance**: Apriori proved slightly faster on this specific dataset due to its sparsity and short transaction length.



**Strongest Rule**: `{root vegetables, whole milk} → {other vegetables}` with a lift of 2.45.



## Frontend Dashboard

The project includes an interactive **Streamlit dashboard**:

 
**Stroke Prediction**: Interactive forms to input patient data and visualize model performance (Confusion Matrices, Precision/Recall).


 
**Market Basket Analysis**: Adjustable sliders for support and confidence thresholds, side-by-side performance comparison, and downloadable rule tables.



## How to Run

1. **Dependencies**: Install the required libraries:
```bash
pip install -r requirements.txt

```


2. **Launch Application**: Run the dashboard:
```bash
streamlit run app.py

```



## Project Structure


`/data/`: Contains `groceries.xlsx` and `healthcare-dataset-stroke-data.xlsx`.


 
`/src/`: Analysis notebooks and core mining scripts.



`app.py`: The main Streamlit dashboard application.



