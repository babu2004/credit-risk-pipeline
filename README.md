# Credit Risk Scoring Project

## Overview
This project focuses on building and evaluating machine learning models to predict credit risk. The goal is to identify customers who are likely to default on their loans based on various demographic and financial features. This notebook explores Decision Trees, Random Forests, and XGBoost models, including parameter tuning and performance comparison.

## Dataset
The dataset used in this project is `CreditScoring.csv`. It contains information about loan applicants, including their status (ok/default), seniority, home ownership, age, marital status, records, job type, expenses, income, assets, debt, loan amount, and price.

## Data Preprocessing and Preparation
1.  **Loading Data**: The `CreditScoring.csv` file is loaded into a pandas DataFrame.
2.  **Column Renaming**: Column names are standardized to lowercase and spaces are replaced with underscores.
3.  **Categorical Variable Re-encoding**: Numerical categorical features (status, home, marital, records, job) are mapped to more descriptive string values.
4.  **Handling Missing Values**: Specific numerical values (e.g., 99999999) representing missing data in 'income', 'assets', and 'debt' columns are replaced with `np.nan`.
5.  **Filtering Unknown Status**: Rows with 'unknown' status are removed from the dataset.
6.  **Train/Validation/Test Split**: The dataset is split into training, validation, and test sets using `train_test_split` with a 60/20/20 ratio (full train -> train/val -> test).
7.  **Target Variable Preparation**: The 'status' column is converted into a binary target variable (`y_train`, `y_val`, `y_test`), where 'default' is 1 and 'ok' is 0.
8.  **Feature Vectorization**: Categorical features are converted into numerical representations using `DictVectorizer`.

## Models and Tuning
### 1. Decision Tree Classifier
*   **Initial Training**: A Decision Tree Classifier is trained with `max_depth=3`.
*   **Hyperparameter Tuning**: The `max_depth` and `min_samples_leaf` parameters are tuned to find the optimal configuration based on AUC scores on the validation set. A heatmap is used to visualize the performance across different parameter combinations.
*   **Final Model**: The best performing Decision Tree uses `max_depth=6` and `min_samples_leaf=20`.

### 2. Random Forest Classifier
*   **Initial Training**: A Random Forest Classifier is trained with `n_estimators=10`.
*   **Hyperparameter Tuning**: `n_estimators`, `max_depth`, and `min_samples_leaf` are tuned to optimize performance.
    *   `n_estimators`: Explored from 10 to 200. Performance stabilizes around 130-190 estimators.
    *   `max_depth`: Explored depths of 5, 10, and 15, with `max_depth=10` showing superior performance.
    *   `min_samples_leaf`: Explored values 1, 3, 5, 10, 15 with `min_samples_leaf=3` giving good results.
*   **Final Model**: The best Random Forest model uses `n_estimators=200`, `max_depth=10`, and `min_samples_leaf=3`.

### 3. XGBoost Classifier
*   **Data Preparation**: Data is converted into `DMatrix` format, optimized for XGBoost.
*   **Initial Training**: An XGBoost model is trained with `eta=0.3`, `max_depth=6`, and `min_child_weight=1`.
*   **Hyperparameter Tuning**: `eta`, `max_depth`, and `min_child_weight` are tuned.
    *   `eta`: Explored values like 0.3, 0.1, 0.05. `eta=0.1` showed good balance between convergence and performance.
    *   `max_depth`: Explored values like 3, 4, 6. `max_depth=3` appeared to be optimal.
    *   `min_child_weight`: Explored values. `min_child_weight=1` was kept as the default due to no significant improvement with other values.
*   **Final Model**: The optimized XGBoost model uses `eta=0.1`, `max_depth=3`, and `min_child_weight=1` with 175 boosting rounds.

## Results and Model Comparison
The Area Under the Receiver Operating Characteristic Curve (AUC) is used as the primary evaluation metric.

| Model             | Validation AUC |
| :---------------- | :------------- |
| Decision Tree     | 0.771          |
| Random Forest     | 0.823          |
| XGBoost           | 0.831          |

**Conclusion**: The XGBoost model achieved the highest AUC score on the validation set, indicating it is the best-performing model for this credit risk scoring task among the three evaluated models. The Random Forest model also performed very well, closely following XGBoost.
