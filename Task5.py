import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
sns.set_style("whitegrid")

# 1. Load the Dataset
try:
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# ====================================================================
# A. DATA OVERVIEW AND CLEANING (Hints a)
# ====================================================================

print("\n--- A. Data Overview (df.info()) ---")
df.info()

print("\n--- B. Descriptive Statistics (df.describe()) ---")
print(df.describe())

print("\n--- C. Missing Values Check ---")
print(df.isnull().sum().sort_values(ascending=False))

# Handling Missing Values
# 1. 'Cabin' has too many missing values (77%), so we drop it.
df.drop('Cabin', axis=1, inplace=True)

# 2. 'Age' has missing values (20%). Impute with the median.
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# 3. 'Embarked' has 2 missing values. Impute with the mode (most frequent port).
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# Drop irrelevant columns for EDA/Correlation
df_eda = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Encode categorical features for correlation (Sex, Embarked, Pclass)
df_encoded = df_eda.copy()
df_encoded['Sex_encoded'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
# Keep original 'Embarked' and 'Pclass' for visualization purposes
df_encoded_viz = df_encoded.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['Embarked'], prefix='Embarked', drop_first=True)


print("\n--- D. Value Counts (Example for Pclass & Sex) ---")
print("Pclass counts:\n", df_eda['Pclass'].value_counts())
print("\nSex counts:\n", df_eda['Sex'].value_counts())


# ====================================================================
# B. UNIVARIATE ANALYSIS (Hints d)
# ====================================================================

plt.figure(figsize=(12, 10))

# 1. Survival Distribution (Target Variable)
plt.subplot(3, 2, 1)
sns.countplot(x='Survived', data=df_eda)
plt.title('1. Survival Count (0=No, 1=Yes)')
plt.ylabel('Number of Passengers')

# 2. Passenger Class Distribution
plt.subplot(3, 2, 2)
sns.countplot(x='Pclass', data=df_eda)
plt.title('2. Passenger Class Distribution (1st, 2nd, 3rd)')
plt.ylabel('Number of Passengers')

# 3. Age Distribution (Histogram)
plt.subplot(3, 2, 3)
sns.histplot(df_eda['Age'], bins=30, kde=True)
plt.title('3. Age Distribution')

# 4. Fare Distribution (Boxplot to show spread and outliers)
plt.subplot(3, 2, 4)
sns.boxplot(x=df_eda['Fare'])
plt.title('4. Fare Distribution (with Outliers)')

# 5. SibSp Distribution
plt.subplot(3, 2, 5)
sns.countplot(x='SibSp', data=df_eda)
plt.title('5. Siblings/Spouses Aboard')

# 6. Parch Distribution
plt.subplot(3, 2, 6)
sns.countplot(x='Parch', data=df_eda)
plt.title('6. Parents/Children Aboard')

plt.tight_layout()
plt.show() # Placeholder for Plot 1

# ====================================================================
# C. BIVARIATE ANALYSIS - SURVIVAL FACTORS (Hints c, d)
# ====================================================================

plt.figure(figsize=(15, 12))

# 7. Survival Rate by Gender
plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=df_eda)
plt.title('7. Survival Rate by Gender')

# 8. Survival Rate by Passenger Class
plt.subplot(2, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=df_eda)
plt.title('8. Survival Rate by Passenger Class')

# 9. Age Distribution (Survived vs. Not Survived)
plt.subplot(2, 2, 3)
sns.kdeplot(df_eda[df_eda['Survived'] == 1]['Age'], fill=True, label='Survived')
sns.kdeplot(df_eda[df_eda['Survived'] == 0]['Age'], fill=True, label='Not Survived')
plt.title('9. Age Density Distribution by Survival')
plt.legend()

# 10. Survival Rate by Embarkation Port
plt.subplot(2, 2, 4)
sns.barplot(x='Embarked', y='Survived', data=df_eda)
plt.title('10. Survival Rate by Embarkation Port')

plt.tight_layout()
plt.show() # Placeholder for Plot 2

# ====================================================================
# D. MULTIVARIATE/CORRELATION ANALYSIS (Hints b)
# ====================================================================

# Select numerical columns for correlation matrix
corr_data = df_encoded[['Survived', 'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]

print("\n--- E. Correlation Matrix ---")
corr_matrix = corr_data.corr()
print(corr_matrix['Survived'].sort_values(ascending=False))

# 11. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('11. Correlation Matrix Heatmap')
plt.show() # Placeholder for Plot 3

# 12. Pairplot (using a small, key subset of features) (Hints b)
# Pclass is converted to a string to treat it as a category for the pairplot hue
df_pair = df_encoded_viz.copy() # Use df_encoded_viz which retains original Pclass
df_pair['Pclass'] = df_pair['Pclass'].astype(str)
sns.pairplot(df_pair[['Survived', 'Pclass', 'Age', 'Fare', 'Sex_encoded']], hue='Survived', palette='viridis', diag_kind='kde')
plt.suptitle('12. Pair Plot of Key Features by Survival', y=1.02)
plt.show() # Placeholder for Plot 4

print("\nAnalysis Complete. Check generated plots and statistical outputs for findings summary.")
