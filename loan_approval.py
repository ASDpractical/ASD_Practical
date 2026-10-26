
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    "Applicant Income": [
        60000, 25000, 80000, 20000, 90000,
        30000, 75000, 18000, 65000, 95000,
        22000, 85000, 40000, 15000, 70000,
        28000, 82000, 12000, 68000, 35000
    ],
    "Credit Score": [
        780, 580, 750, 520, 800, 610, 740, 490,
        720, 810, 550, 770, 650, 450, 730, 590,
        790, 430, 710, 620
    ],
    "Loan Amount": [
        200000, 300000, 250000, 350000, 180000,
        280000, 220000, 320000, 200000, 150000,
        330000, 190000, 250000, 360000, 210000,
        310000, 180000, 340000, 230000, 270000
    ],
    "Approval": [
        "Approved", "Rejected", "Approved", "Rejected",
        "Approved", "Rejected", "Approved", "Rejected",
        "Approved", "Approved", "Rejected", "Approved",
        "Approved", "Rejected", "Approved", "Rejected",
        "Approved", "Rejected", "Approved", "Rejected"
    ]
}

df = pd.DataFrame(data)
df.to_csv("loan approval.csv", index=False)

# Load dataset
df = pd.read_csv("loan approval.csv")
print("Dataset:\n", df)

# Separate features and target
X = df[["Applicant Income", "Credit Score", "Loan Amount"]]
y = df["Approval"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# MLflow experiment
mlflow.set_experiment("Loan_Approval")

# Compare two max_depth values
for depth in [2, 4]:
    with mlflow.start_run(run_name=f"depth_{depth}"):
        model = DecisionTreeClassifier(
            max_depth=depth, random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("\nMax Depth:", depth)
        print("Accuracy:", round(accuracy, 4))

print("\nLoan approval experiments completed.")

# py -0p
# py -3.12 --version
# py -3.12 -m pip install pandas scikit-learn mlflow
# py -3.12 -c "import pandas, sklearn, mlflow; print('All libraries installed successfully')"
# cd C:\College\ASD
# py -3.12 loan_approval.py
# py -3.12 -m mlflow ui
# http://127.0.0.1:5000