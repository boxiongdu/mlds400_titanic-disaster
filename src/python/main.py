import pandas as pd
import sklearn
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"
train = pd.read_csv(data_dir / "train.csv")

print("\nloaded data:")
print(train.head())

print("\nColumn types and non-null counts:")
train.info()

print("\nSummary statistics:")
print(train.describe())

print("\nMissing values per column:")
print(train.isna().sum())

# Select a small set of features for the model
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
prepared = train[features + ["Survived"]].copy()

print(f"\nSelected model features: {features}")
print("Excluded identifiers and text fields from this initial model.")

# Fill missing numeric values using training-data medians
age_median = prepared["Age"].median()
fare_median = prepared["Fare"].median()

prepared["Age"] = prepared["Age"].fillna(age_median)
prepared["Fare"] = prepared["Fare"].fillna(fare_median)

print(f"\nFilled missing Age values with median: {age_median:.2f}")
print(f"Filled missing Fare values with median: {fare_median:.2f}")

# Convert sex to a numeric feature
prepared["Sex"] = prepared["Sex"].map({"male": 0, "female": 1})
print("\nEncoded Sex: male = 0, female = 1.")

# Treat passenger class as categorical
prepared = pd.get_dummies(
    prepared,
    columns=["Pclass"],
    drop_first=True,
    dtype=int,
)
print("Encoded Pclass as dummy variables, with class 1 as the baseline.")

print("\nMissing values after preprocessing:")
print(prepared.isna().sum())

print("\nPrepared dataset:")
print(prepared.head())

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Question 15: Build and fit a logistic regression model
X_train = prepared.drop(columns="Survived")
y_train = prepared["Survived"]

# Scaling helps logistic regression converge
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=38),
)

model.fit(X_train, y_train)
print("\n15. Trained logistic regression with standardized features.")

# Question 16: Measure training accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

print(f"\n16. Training accuracy: {train_accuracy:.2%}")
print("This measures performance on the same data used to fit the model.")

# Question 17: Load and prepare the test dataset
test = pd.read_csv(data_dir / "test.csv")
print(f"\n17. Loaded test dataset: {test.shape[0]} passengers.")

X_test = test[features].copy()

# Use the medians calculated from TRAINING data
X_test["Age"] = X_test["Age"].fillna(age_median)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)

print(f"Filled missing test Age values with training median: {age_median:.2f}")
print(f"Filled missing test Fare values with training median: {fare_median:.2f}")

X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
print("Encoded test Sex: male = 0, female = 1.")

# Fix the categories so class 1 is always the baseline
X_test["Pclass"] = pd.Categorical(
    X_test["Pclass"],
    categories=[1, 2, 3],
)

X_test = pd.get_dummies(
    X_test,
    columns=["Pclass"],
    drop_first=True,
    dtype=int,
)

# Match the training feature names and column order
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
print("Encoded passenger class and aligned test columns with training columns.")

# predict() uses the scaler fitted on training data
test_predictions = model.predict(X_test)

results = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "PredictedSurvived": test_predictions,
})

print("\nFirst 10 test predictions (0 = died, 1 = survived):")
print(results.head(10).to_string(index=False))
print(f"Generated predictions for all {len(results)} test passengers.")

# Question 18: Report whether test accuracy can be calculated
if "Survived" in test.columns and test["Survived"].notna().all():
    test_accuracy = accuracy_score(test["Survived"], test_predictions)
    print(f"\n18. Test accuracy: {test_accuracy:.2%}")
else:
    print("\n18. Test accuracy cannot be calculated:")
    print("The provided test.csv does not contain complete true survival labels.")

# Question 19: Progress, preprocessing, and results are printed above
print("\nAnalysis completed.")