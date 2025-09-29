# Task5
Python Data Analytics
# Task 5: Exploratory Data Analysis (EDA) on the Titanic Dataset

## Project Goal
This project fulfills Task 5 of the Data Analyst Internship program, which involves performing Exploratory Data Analysis (EDA) on the Titanic passenger data (`train.csv`). The primary objective is to visually and statistically explore the dataset to extract meaningful insights, identify patterns, and determine the key factors influencing passenger survival.

## Tools Used
* **Python 3**
* **Pandas:** Data manipulation and cleaning.
* **NumPy:** Numerical operations.
* **Matplotlib & Seaborn:** Data visualization.

## Analysis Steps
1.  **Data Preprocessing:** Handled missing values (imputed 'Age' and 'Embarked', dropped 'Cabin'), and dropped irrelevant features ('Name', 'Ticket', 'PassengerId').
2.  **Univariate Analysis:** Explored the distribution of individual features (`Survived`, `Pclass`, `Age`, `Fare`, etc.).
3.  **Bivariate/Multivariate Analysis:** Examined the relationship between survival and key features (`Sex`, `Pclass`, `Age`, `Fare`) using bar plots, KDE plots, and a **Correlation Heatmap**.
4.  **Key Findings:** Summarized the observations, confirming that **Gender** and **Passenger Class** were the dominant predictors of survival.

## Files in this Repository
* `train.csv`: The raw dataset used for the analysis.
* `eda_titanic.py`: The complete Python script containing the EDA code.
* `findings_summary.pdf` (Placeholder): A document summarizing the analytical findings and insights.
