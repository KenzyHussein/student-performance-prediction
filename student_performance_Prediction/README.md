# Student Performance Prediction - Streamlit

This project predicts whether a student is likely to pass based on the UCI Student Performance dataset.

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The app can run one model at a time or all four models together and compare their predictions and test metrics.

## Important dataset choices

- G1, G2 and G3 are not used as input features.
- G3 is used to create the target:
  - G3 >= 10 -> Pass
  - G3 < 10 -> Fail
- Dalc and Walc are excluded.
- Romantic Relationship is kept as a feature.
- Mother's Education, Father's Education and Parents' Cohabitation use the original UCI codes.

## Run locally

Open PowerShell in this folder:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app downloads `student-mat.csv` automatically from the UCI repository, so you do not need to place the CSV in the project folder.

## Models

XGBoost is implemented with `XGBClassifier`.
