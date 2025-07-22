# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using various classification algorithms and advanced data preprocessing techniques.

## Project Overview

This project implements a robust fraud detection system that identifies potentially fraudulent credit card transactions. The system addresses the challenge of highly imbalanced datasets common in fraud detection scenarios and employs multiple machine learning models to achieve optimal performance.

### Key Features
- **Data Preprocessing**: Outlier detection and removal using IQR method
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Feature Scaling**: StandardScaler for numerical features
- **Multiple Algorithms**: Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
- **Comprehensive Evaluation**: ROC curves, Precision-Recall curves, Confusion matrices

## Dataset

The project uses the **Credit Card Fraud Detection Dataset** (`creditcard.csv`), which contains:
- **284,807 transactions** (original dataset before preprocessing)
- **30 numerical features** (V1-V28 are PCA-transformed, plus Time and Amount)
- **Binary target variable** (Class: 0 = Legitimate, 1 = Fraudulent)
- **Highly imbalanced** (~0.17% fraudulent transactions)

### Dataset Characteristics
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction amount
- **V1-V28**: Principal Component Analysis (PCA) transformed features
- **Class**: Target variable (0: Normal, 1: Fraud)

## 🛠️ Technologies Used

```python
# Core Libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
scipy>=1.7.0

# Optional
jupyter>=1.0.0  # for notebook version
```

## Project Structure

```
credit-card-fraud-detection/
│__ Credit-Card-Fraud-analysis # Main implementation file 
├── Credit-Card-Fraud.py              # Main Python script
├── creditcard.csv             # Dataset (download separately)
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
   - Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the project root directory

## Usage

### Running the Complete Pipeline
```bash
python ml_project.py
```

### Key Components

#### 1. Data Preprocessing
```python
# Outlier removal using IQR method
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
# Remove outliers beyond 1.5 * IQR
```

#### 2. Handling Class Imbalance
```python
# Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

#### 3. Model Training
The project implements two main algorithms:
- **Logistic Regression** with hyperparameter tuning
- **Random Forest** with randomized search optimization

## Model Performance

### Logistic Regression
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Parameters Optimized**: Regularization strength (C), penalty type (L1/L2)
- **Scoring Metric**: F1-score (optimal for imbalanced datasets)

### Random Forest
- **Hyperparameter Tuning**: RandomizedSearchCV for efficiency
- **Parameters Optimized**: n_estimators, max_depth, min_samples_split, etc.
- **Class Balancing**: Built-in class_weight='balanced'

### Evaluation Metrics
- **ROC-AUC Score**: Area under ROC curve
- **Precision-Recall Curve**: Critical for imbalanced datasets
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Precision, Recall, F1-score per class

## Visualizations

The project generates comprehensive visualizations:
1. **Transaction Amount Distribution**: Boxplot showing outliers
2. **Class Distribution**: Count plot of legitimate vs fraudulent transactions
3. **Correlation Heatmap**: Feature correlation analysis
4. **ROC Curves**: Model performance comparison
5. **Precision-Recall Curves**: Detailed performance for minority class
6. **Confusion Matrices**: Classification accuracy breakdown

## Key Results

### Dataset Statistics (Post-preprocessing)
- **Class Distribution**: ~99.83% Legitimate, ~0.17% Fraudulent
- **Data Quality**: No missing values, outliers removed using IQR method

### Model Insights
- **SMOTE Effectiveness**: Balanced training data for better minority class learning
- **Feature Scaling**: Critical for algorithms like Logistic Regression
- **Cross-validation**: Prevents overfitting and ensures generalization

## Customization

### Adjusting SMOTE Parameters
```python
smote = SMOTE(
    sampling_strategy='auto',  # Can specify ratios
    k_neighbors=5,             # Number of neighbors
    random_state=42
)
```

### Modifying Hyperparameter Search
```python
# Logistic Regression parameters
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

## Requirements

Create a `requirements.txt` file:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
scipy>=1.7.0
```

## Acknowledgments

- **Dataset Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Research Paper**: The dataset was collected and analysed during a research collaboration of Worldline and the Machine Learning Group
- **Libraries**: Thanks to scikit-learn, imbalanced-learn, and other open-source contributors

## Contact

For questions or suggestions, please open an issue or contact [Parthaggarwal481@gmail.com].

---

**Note**: This project is for educational and research purposes. In production environments, additional security measures, real-time processing capabilities, and regulatory compliance would be necessary.
