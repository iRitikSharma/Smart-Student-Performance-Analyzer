# ==============================
# Student Success Prediction
# Using Logistic Regression
# ==============================

# ----- Import Required Libraries -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ----- Load Dataset -----
data = pd.read_excel("student_success_data.xlsx")
df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())


# ----- Check for Missing Values -----
print("\nMissing values in each column:")
print(df.isnull().sum())


# ----- Encode Categorical Columns -----
# LabelEncoder is used when we have only two categories (Yes / No)
le = LabelEncoder()

df['Internet'] = le.fit_transform(df['Internet'])   # Yes → 1, No → 0
df['Passed'] = le.fit_transform(df['Passed'])       # Pass → 1, Fail → 0


# ----- Feature Scaling -----
features = ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])


# ----- Split Features and Target -----
X = df_scaled[features]      # Independent variables
y = df_scaled['Passed']      # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----- Train Logistic Regression Model -----
model = LogisticRegression()
model.fit(X_train, y_train)


# ----- Model Evaluation -----
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ----- Confusion Matrix Visualization -----
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# ----- User Input Prediction -----
print("\n------ Predict Your Result ------")

try:
    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance (%): "))
    past_score = float(input("Enter Past Score: "))
    sleep_hours = float(input("Enter Sleep Hours: "))

    user_input_df = pd.DataFrame([{
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hours
    }])

    # Scale user input using the same scaler
    user_input_scaled = scaler.transform(user_input_df)

    prediction = model.predict(user_input_scaled)[0]
    result = "Pass" if prediction == 1 else "Fail"

    print(f"\nPrediction Based on Input: {result}")

except Exception as e:
    print("An error occurred:", e)


# =====================================================
# Notes:
# -----------------------------------------------------
# 1. Machine learning models do not understand strings,
#    they only work with numeric data.
#
# 2. LabelEncoder is best when there are only two
#    categories (Yes / No, Pass / Fail).
#
# 3. For multiple categories (like cities), use
#    pd.get_dummies().
#
# 4. Feature scaling is important because ML models
#    are sensitive to differences in data scale.
#
# =====================================================


# ----- Machine Learning Workflow -----
# Step 1: Load and understand the data
# Step 2: Preprocess data (cleaning, encoding)
# Step 3: Feature scaling (StandardScaler / MinMaxScaler)
# Step 4: Train the model
# Step 5: Evaluate performance
# Step 6: Make predictions
