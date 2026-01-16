# 🎓 Smart Student Performance Analyzer

An end-to-end **Machine Learning project** that predicts whether a student will **Pass or Fail** based on academic and lifestyle factors. This project demonstrates the complete ML workflow — from data preprocessing to model evaluation and user-based prediction — using **Python, Pandas, Scikit-learn, and Data Visualization**.

---

## 📌 Project Overview

Educational institutions often struggle to identify students who may be at academic risk. This project aims to solve that problem by analyzing student data and predicting academic outcomes using **Logistic Regression**.

The system takes into account:

* 📚 Study Hours
* 🏫 Attendance Percentage
* 📝 Past Academic Score
* 😴 Sleep Hours

Using these inputs, the model predicts whether a student is likely to **Pass or Fail**.

---

## 🚀 Key Features

✅ Clean and structured ML pipeline
✅ Binary classification using Logistic Regression
✅ Feature scaling using StandardScaler
✅ Label encoding for categorical data
✅ Performance evaluation with classification report
✅ Confusion matrix visualization using Seaborn
✅ Interactive user input for real-time prediction

---

## 🧠 Machine Learning Workflow

1. **Data Loading** – Load Excel-based student dataset
2. **Data Understanding** – Inspect structure and missing values
3. **Data Preprocessing**

   * Encode categorical values (Yes/No → 1/0)
   * Scale numerical features
4. **Feature Selection** – Select important academic attributes
5. **Train-Test Split** – 80% training, 20% testing
6. **Model Training** – Logistic Regression
7. **Model Evaluation**

   * Classification Report
   * Confusion Matrix
8. **Prediction** – Predict results based on user input

---

## 🗂️ Dataset Description

The dataset is stored in an Excel file:
📄 **student_success_data.xlsx**

### Columns Used:

| Column Name | Description                     |
| ----------- | ------------------------------- |
| StudyHours  | Number of hours studied per day |
| Attendance  | Attendance percentage           |
| PastScore   | Previous academic performance   |
| SleepHours  | Average sleep hours per day     |
| Internet    | Internet availability (Yes/No)  |
| Passed      | Final result (Pass/Fail)        |

---

## 🛠️ Technologies Used

* **Python** 🐍
* **Pandas & NumPy** – Data handling
* **Scikit-learn** – Machine learning
* **Matplotlib & Seaborn** – Data visualization
* **Excel (.xlsx)** – Dataset storage

---

## 📊 Model Used

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for **binary classification**. It estimates the probability of a binary outcome using a logistic (sigmoid) function.

**Why Logistic Regression?**

* Simple and interpretable
* Efficient for small to medium datasets
* Ideal for binary outcomes (Pass / Fail)

---

## 📈 Model Evaluation

The model performance is evaluated using:

* **Classification Report**

  * Precision
  * Recall
  * F1-score
  * Accuracy

* **Confusion Matrix**

  * Visual representation of correct vs incorrect predictions

---

## 🧪 Sample Prediction Flow

```text
Enter Study Hours: 6
Enter Attendance (%): 85
Enter Past Score: 78
Enter Sleep Hours: 7

Prediction Based on Input: PASS
```

---

## ▶️ How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/your-username/student-success-prediction.git
```

2. **Navigate to the project folder**

```bash
cd student-success-prediction
```

3. **Install required dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

4. **Run the Python script**

```bash
python student_success_prediction.py
```

---

## 📌 Project Structure

```text
student-success-prediction/
│
├── student_success_data.xlsx
├── student_success_prediction.py
├── README.md
└── requirements.txt
```

---

## 💡 Learning Outcomes

By working on this project, you will learn:

* How to preprocess real-world data
* When and why to use Label Encoding
* Importance of feature scaling
* Implementing Logistic Regression
* Evaluating classification models
* Visualizing ML results

---

## 🔮 Future Enhancements

* Add more features (family income, mental health, etc.)
* Try advanced models (Random Forest, XGBoost)
* Convert into a **Flask/Django web application**
* Deploy model using **REST API**
* Add model persistence using Pickle

---

## 👨‍💻 Author

**Ritik Sharma**
Python Developer | Machine Learning Enthusiast

---

## ⭐ Final Note

This project is ideal for:

* Machine Learning beginners
* Academic mini-projects
* Resume & portfolio showcase
* Interview demonstrations

If you like this project, don’t forget to ⭐ the repository!

Happy Learning 🚀
