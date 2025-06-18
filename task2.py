### Load Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')
df.head()

### Data Cleaning

# Summary of dataset
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Drop columns with too many missing values
df.drop(columns=['deck'], inplace=True)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop rows with remaining missing values
df.dropna(inplace=True)

# Convert 'sex', 'embarked', 'class' to categorical
categorical_cols = ['sex', 'embarked', 'class', 'who', 'adult_male', 'alone']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Final check
df.info()
### Exploratory Data Analysis (EDA)

# Survival count
sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.show()

# Age distribution
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Survival by Sex
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Sex')
plt.show()

# Survival by Class
sns.countplot(x='class', hue='survived', data=df)
plt.title('Survival by Class')
plt.show()

# Age vs Fare
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title('Age vs Fare (colored by Survival)')
plt.show()

# Encode categorical variables for correlation
df_encoded = pd.get_dummies(df[['sex', 'class', 'embarked', 'who', 'alone']], drop_first=True)
df_numeric = pd.concat([df[['survived', 'age', 'fare', 'pclass', 'sibsp', 'parch']], df_encoded], axis=1)

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()