# student-performance-prediction

A machine learning web application that predicts whether a student is likely to **Pass or Fail** based on academic, demographic, and family-related factors.

The project uses the **UCI Student Performance Dataset** and provides an interactive **Streamlit** interface where users can enter student information, select a machine learning model, and compare predictions.

## Features

* Interactive Streamlit web interface
* Student performance prediction: **Pass / Fail**
* Four machine learning models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost
* Option to use one model or compare all models
* Pass probability for each prediction
* Model performance comparison using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Displays the best-performing model based on test accuracy
* Uses the original UCI dataset coding for categorical features
* Dataset is downloaded automatically from the UCI repository

## Dataset

This project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

The application uses the **Math course dataset (`student-mat.csv`)**.

### Target

The target variable is created from the final grade `G3`:

* `G3 >= 10` → **Pass (1)**
* `G3 < 10` → **Fail (0)**

### Features

The model uses student-related features such as:

* School
* Sex
* Age
* Home address
* Family size
* Parents' cohabitation status
* Mother's education
* Father's education
* Parents' jobs
* Reason for choosing school
* Guardian
* Travel time
* Study time
* Past class failures
* Educational support
* Extra-curricular activities
* Internet access
* Family relationship quality
* Free time
* Going out
* Health
* Absences
* Romantic relationship

### Excluded Features

`G1`, `G2`, and `G3` are not used as input features.

* `G1` = first period grade
* `G2` = second period grade
* `G3` = final grade

`G3` is only used to create the **Pass/Fail target**.

`Dalc` and `Walc` are also excluded from the model.

## Machine Learning Models

### Logistic Regression

A classification model used as a simple and interpretable baseline.

### Decision Tree

A tree-based model that makes predictions through a sequence of decision rules.

### Random Forest

An ensemble model that combines multiple decision trees to improve prediction performance.

### XGBoost

A gradient boosting model that builds an ensemble of decision trees sequentially to improve classification performance.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* Streamlit


### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Requirements

The required Python libraries are:

```text
streamlit
pandas
scikit-learn
xgboost
```

## How It Works

```text
UCI Student Performance Dataset
            ↓
      Data Preparation
            ↓
      Create Pass Target
            ↓
      Remove G1, G2, G3
            ↓
       Encode Features
            ↓
       Train/Test Split
            ↓
 ┌──────────┬──────────┬──────────┬──────────┐
 ↓          ↓          ↓          ↓
Logistic   Decision   Random    XGBoost
Regression  Tree      Forest
 ↓          ↓          ↓          ↓
 └──────────┴──────────┴──────────┴──────────┘
            ↓
      Model Prediction
            ↓
     Performance Metrics
            ↓
      Streamlit Results
```

## Model Evaluation

The models are evaluated using:

| Metric    | Description                                      |
| --------- | ------------------------------------------------ |
| Accuracy  | Percentage of correct predictions                |
| Precision | How many predicted passes were actually passes   |
| Recall    | How many actual passes were correctly identified |
| F1 Score  | Balance between precision and recall             |


