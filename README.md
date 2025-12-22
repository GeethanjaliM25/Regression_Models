Healthcare Stroke Prediction – Regression Models Demo

A comprehensive machine learning project that demonstrates and compares multiple regression techniques on a real-world healthcare stroke dataset, including detailed preprocessing, evaluation metrics, and visualizations.

📌 Project Overview

This project applies various regression algorithms to predict a continuous health indicator (avg_glucose_level) from patient medical data.
It is designed to demonstrate all major regression models in a single workflow, making it ideal for academic demos, lab evaluations, and interviews.

🚀 Regression Models Implemented

The following regression techniques are implemented and compared:

Linear Regression

Polynomial Regression (Degree 2)

Ridge Regression

Lasso Regression

Elastic Net Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regression (SVR)

Each model is evaluated using:

R² Score

Mean Squared Error (MSE)

📂 Dataset Information

Dataset Name: Healthcare Stroke Dataset

Source: Kaggle

Type: Real-world medical dataset

Target Variable: avg_glucose_level

Dataset Features Include:

Age

Gender

Hypertension

Heart Disease

BMI

Smoking Status

Work Type

Residence Type

Average Glucose Level

🛠️ Technologies Used

Python 3.13

Pandas & NumPy – Data manipulation

Matplotlib & Seaborn – Data visualization

Scikit-learn – Machine Learning models & preprocessing

VS Code – Development environment

📁 Project Structure
regression_types/
│
├── stroke_regression_demo.py        # Main Python script
├── healthcare-dataset-stroke-data.csv
├── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/healthcare-stroke-regression.git
cd healthcare-stroke-regression

2️⃣ Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn

▶️ How to Run the Project

Make sure the CSV file is in the same directory as the Python script.

python stroke_regression_demo.py

📊 Output & Visualizations

After execution, the project produces:

✅ Console Output

Dataset shape and column details

Selected target variable

Model-wise R² score and MSE

📈 Graphical Output

Correlation heatmap

Regression model comparison bar chart

Actual vs Predicted scatter plots for each model

All visualizations open automatically using Matplotlib.

🧪 Preprocessing Steps

Removal of ID columns

Handling missing values

One-hot encoding of categorical variables

Feature scaling using StandardScaler

Train-test split (80% training, 20% testing)

🎯 Key Learning Outcomes

Understanding differences between regression models

Handling real-world healthcare data

Model evaluation using statistical metrics

Visual comparison of regression performance

Writing reusable, modular ML code

📌 Use Cases

Academic mini project

Machine Learning lab demonstration

Regression concepts explanation

Healthcare data analysis

Resume and GitHub portfolio project

🔮 Future Enhancements

Hyperparameter tuning using GridSearchCV

Model persistence using Pickle

Web interface using Flask or Streamlit

Feature importance visualization

Time-series health prediction

👩‍💻 Author

Geethanjali M
B.E. Student – Computer Science & Engineering
Machine Learning & Data Science Enthusiast
