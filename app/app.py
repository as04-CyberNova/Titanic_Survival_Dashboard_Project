import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("Titanic Survival Dashboard")

df = pd.read_csv("data\titanic.csv")
df = df.replace("?", pd.NA)

st.sidebar.header("Filters")

if "Sex" in df.columns:
    gender_filter = st.sidebar.multiselect("Select Gender", df["Sex"].dropna().unique())
    if gender_filter:
        df = df[df["Sex"].isin(gender_filter)]

if "Pclass" in df.columns:
    class_filter = st.sidebar.multiselect("Select Passenger Class", df["Pclass"].dropna().unique())
    if class_filter:
        df = df[df["Pclass"].isin(class_filter)]

if "Age" in df.columns:
    age_filter = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1])]

if "Survived" in df.columns:
    survival_filter = st.sidebar.multiselect("Select Survival Status", df["Survived"].dropna().unique())
    if survival_filter:
        df = df[df["Survived"].isin(survival_filter)]

if "Embarked" in df.columns:
    embarked_filter = st.sidebar.multiselect("Select Embarkation Port", df["Embarked"].dropna().unique())
    if embarked_filter:
        df = df[df["Embarked"].isin(embarked_filter)]

st.subheader("Summary Metrics")
col1, col2, col3 = st.columns(3)
if "Survived" in df.columns:
    survival_rate = df["Survived"].mean() * 100
    col1.metric("Survival Rate", f"{survival_rate:.2f}%")

if "Age" in df.columns:
    col2.metric("Average Age", round(df["Age"].mean(), 2))

col3.metric("Total Passengers", len(df))

st.subheader("Raw Data Preview")
st.write(df.head())

st.subheader("Survival Counts")
if "Survived" in df.columns:
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Survival by Gender")
if "Sex" in df.columns and "Survived" in df.columns:
    fig, ax = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Survival by Passenger Class")
if "Pclass" in df.columns and "Survived" in df.columns:
    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Age Distribution")
if "Age" in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=["number"])
if not numeric_df.empty:
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.subheader("Logistic Regression Model: Predict Survival")
ml_df = df.copy()
categorical_cols = ml_df.select_dtypes(include=["object"]).columns.tolist()
ml_df = pd.get_dummies(ml_df, columns=categorical_cols, drop_first=True)
ml_df = ml_df.dropna(subset=["Age", "Fare", "Survived"])

X = ml_df.drop("Survived", axis=1)
y = ml_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")

st.write(cm)
st.write("Classification Report")
st.text(classification_report(y_test, y_pred))
