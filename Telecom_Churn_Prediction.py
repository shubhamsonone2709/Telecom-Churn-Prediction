import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns

# Load the dataset
f = pd.read_csv("Churn.csv")

# Data preprocessing and manipulation
f = f.drop(columns=["customerID", "tenure", "TotalCharges"])
f = f.dropna()

label_encoder = LabelEncoder()
f = f.apply(label_encoder.fit_transform)

# def group_tenure(tenure):
#     if tenure >= 0 and tenure <= 12:
#         return '0-12 Month'
#     elif tenure > 12 and tenure <= 24:
#         return '12-24 Month'
#     elif tenure > 24 and tenure <= 48:
#         return '24-48 Month'
#     elif tenure > 48 and tenure <= 60:
#         return '48-60 Month'
#     elif tenure > 60:
#         return '> 60 Month'

# f['tenure_group'] = f['tenure'].apply(group_tenure)
# f = pd.get_dummies(f, columns=['tenure_group'], drop_first=True)

# Resampling using SMOTE
X = f.drop(columns=["Churn"])
y = f["Churn"]
oversampler = SMOTE()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=2017)

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Results for {name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Visualization
accuracy_values = [model.score(X_test, y_test) for model in models.values()]
accuracy_df = pd.DataFrame({
    "method": list(models.keys()),
    "accuracy": accuracy_values
})

sns.barplot(x="method", y="accuracy", data=accuracy_df)
plt.title("Accuracy Comparison of Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()
