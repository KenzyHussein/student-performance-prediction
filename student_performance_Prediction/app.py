import streamlit as st
import pandas as pd
from io import BytesIO
from urllib.request import urlopen
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title("Student Performance Prediction")
st.write(
    "Enter the student's information, choose one or all machine learning models, "
    "and compare their predictions and performance."
)

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    with urlopen(url) as response:
        zip_bytes = BytesIO(response.read())

    with zipfile.ZipFile(zip_bytes) as z:
        with z.open("student-mat.csv") as f:
            return pd.read_csv(f, sep=";")

@st.cache_resource
def train_models():
    df = load_data().copy()

    # G3 is the final grade. Pass = 1 when G3 >= 10, otherwise 0.
    df["Pass"] = (df["G3"] >= 10).astype(int)

    # G1, G2 and G3 are grades and are not used as prediction inputs.
    # Dalc and Walc are excluded as requested.
    # Romantic relationship is kept as a feature.
    columns_to_drop = ["G1", "G2", "G3", "Pass", "Dalc", "Walc"]

    X = df.drop(columns=columns_to_drop)
    y = df["Pass"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        }

    return trained_models, metrics, X.columns.tolist()

trained_models, metrics, feature_columns = train_models()

st.sidebar.header("Model Selection")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "All Models"
]

selected_model = st.sidebar.selectbox(
    "Choose model",
    model_options
)

st.header("Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    school = st.selectbox("School", ["GP - Gabriel Pereira", "MS - Mousinho da Silveira"])
    sex = st.selectbox("Sex", ["F - Female", "M - Male"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home Address", ["U - Urban", "R - Rural"])
    famsize = st.selectbox("Family Size", ["LE3 - Less than or equal to 3", "GT3 - Greater than 3"])

with col2:
    pstatus = st.selectbox(
        "Parents' Cohabitation",
        ["T - Living together", "A - Living apart"]
    )

    medu = st.selectbox(
        "Mother's Education",
        [
            "0 - None",
            "1 - Primary education (4th grade)",
            "2 - 5th to 9th grade",
            "3 - Secondary education",
            "4 - Higher education"
        ]
    )

    fedu = st.selectbox(
        "Father's Education",
        [
            "0 - None",
            "1 - Primary education (4th grade)",
            "2 - 5th to 9th grade",
            "3 - Secondary education",
            "4 - Higher education"
        ]
    )

    mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
    fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])

with col3:
    reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    traveltime = st.selectbox(
        "Travel Time",
        [
            "1 - <15 min",
            "2 - 15 to 30 min",
            "3 - 30 min to 1 hour",
            "4 - >1 hour"
        ]
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        [
            "1 - <2 hours",
            "2 - 2 to 5 hours",
            "3 - 5 to 10 hours",
            "4 - >10 hours"
        ]
    )

    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4])

st.header("Academic and Family Factors")

col1, col2, col3 = st.columns(3)

with col1:
    schoolsup = st.selectbox("Extra Educational Support", ["no", "yes"])
    famsup = st.selectbox("Family Educational Support", ["no", "yes"])
    paid = st.selectbox("Extra Paid Classes", ["no", "yes"])
    activities = st.selectbox("Extra-curricular Activities", ["no", "yes"])
    nursery = st.selectbox("Attended Nursery School", ["no", "yes"])

with col2:
    higher = st.selectbox("Wants Higher Education", ["no", "yes"])
    internet = st.selectbox("Internet Access at Home", ["no", "yes"])
    famrel = st.slider("Family Relationship Quality", 1, 5, 3)
    freetime = st.slider("Free Time After School", 1, 5, 3)
    goout = st.slider("Going Out With Friends", 1, 5, 3)

with col3:
    health = st.slider("Current Health Status", 1, 5, 3)
    absences = st.number_input("School Absences", min_value=0, max_value=93, value=0)
    romantic = st.selectbox("Romantic Relationship", ["no", "yes"])

def extract_code(value):
    return value.split(" - ")[0]

def create_input():
    student = {
        "school": extract_code(school),
        "sex": extract_code(sex),
        "age": age,
        "address": extract_code(address),
        "famsize": extract_code(famsize),
        "Pstatus": extract_code(pstatus),
        "Medu": int(extract_code(medu)),
        "Fedu": int(extract_code(fedu)),
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(extract_code(traveltime)),
        "studytime": int(extract_code(studytime)),
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "health": health,
        "absences": absences
    }

    student_df = pd.DataFrame([student])
    student_df = pd.get_dummies(student_df, drop_first=True)
    student_df = student_df.reindex(columns=feature_columns, fill_value=0)

    return student_df

if st.button("Predict Performance"):
    student_input = create_input()

    if selected_model == "All Models":
        st.header("Model Predictions")

        results = []

        for name, model in trained_models.items():
            prediction = int(model.predict(student_input)[0])
            probability = float(model.predict_proba(student_input)[0][1])

            results.append({
                "Model": name,
                "Prediction": "Pass" if prediction == 1 else "Fail",
                "Pass Probability": f"{probability * 100:.2f}%"
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        st.header("Model Performance Comparison")

        metric_df = pd.DataFrame(metrics).T
        st.dataframe(
            metric_df.style.format("{:.4f}"),
            use_container_width=True
        )

        best_model = metric_df["Accuracy"].idxmax()
        st.write(
            f"Best model based on test accuracy: **{best_model}** "
            f"({metric_df.loc[best_model, 'Accuracy']:.2%})"
        )

        predictions = results_df["Prediction"].tolist()

        if len(set(predictions)) == 1:
            st.success(f"All models agree: {predictions[0]}.")
        else:
            st.warning("The models do not all agree on the prediction.")

    else:
        model = trained_models[selected_model]

        prediction = int(model.predict(student_input)[0])
        probability = float(model.predict_proba(student_input)[0][1])

        st.header(f"{selected_model} Prediction")

        if prediction == 1:
            st.success("Predicted Result: Pass")
        else:
            st.error("Predicted Result: Fail")

        st.write(f"Pass Probability: **{probability * 100:.2f}%**")

        st.header("Model Performance")
        metric_values = metrics[selected_model]

        metric_df = pd.DataFrame(
            [metric_values],
            index=[selected_model]
        )

        st.dataframe(
            metric_df.style.format("{:.4f}"),
            use_container_width=True
        )
