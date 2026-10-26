
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create dataset
data = {
    "Years of Experience": [
        8, 2, 10, 1, 7, 3, 12, 2, 9, 1,
        6, 4, 11, 2, 8, 3, 10, 1, 7, 5
    ],
    "Monthly Salary": [
        70000, 25000, 90000, 22000, 65000,
        30000, 95000, 28000, 80000, 20000,
        60000, 35000, 98000, 24000, 75000,
        32000, 92000, 18000, 68000, 45000
    ],
    "Job Satisfaction Score": [
        9, 3, 8, 2, 8, 4, 9, 3, 7, 1,
        8, 5, 9, 2, 8, 3, 9, 1, 7, 6
    ],
    "Attrition": [
        "Stay", "Leave", "Stay", "Leave", "Stay",
        "Leave", "Stay", "Leave", "Stay", "Leave",
        "Stay", "Leave", "Stay", "Leave", "Stay",
        "Leave", "Stay", "Leave", "Stay", "Stay"
    ]
}

df = pd.DataFrame(data)
df.to_csv("employee_attrition.csv", index=False)
print("CSV dataset created successfully.")

# 2. Load dataset
df = pd.read_csv("employee_attrition.csv")
print("\nDataset:\n", df)

# 3. Separate features and target
X = df[
    [
        "Years of Experience",
        "Monthly Salary",
        "Job Satisfaction Score"
    ]
]
y = df["Attrition"]

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Create MLflow experiment
mlflow.set_experiment("Employee_Attrition")

# 6. Train and compare two models
for depth in [2, 4]:
    with mlflow.start_run(run_name=f"depth_{depth}"):
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # 7. Predict and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 8. Log parameters and metric
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        # 9. Log trained model (MLflow compatibility fix)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"\nMax Depth: {depth}")
        print(f"Accuracy: {accuracy:.4f}")

print("\nEmployee attrition experiments completed successfully.")

# cd C:\College\ASD
# py -3.12 employee_attrition.py
# py -0p
# py -3.12 --version
# py -3.12 -m pip install pandas scikit-learn mlflow
# py -3.12 -c "import pandas, sklearn, mlflow; print('All libraries installed successfully')"
# cd C:\College\ASD
# py -3.12 loan_approval.py
# py -3.12 -m mlflow ui
# http://127.0.0.1:5000