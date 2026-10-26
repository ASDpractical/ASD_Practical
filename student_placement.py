
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create dataset
data = {
    "CGPA": [
        8.5, 6.2, 7.8, 5.5, 9.1, 6.8, 8.0, 5.9, 7.2, 9.3,
        6.0, 8.7, 7.5, 5.2, 8.9, 6.5, 7.9, 5.8, 9.0, 7.0
    ],
    "Aptitude Score": [
        85, 55, 78, 45, 92, 60, 80, 50, 70, 95,
        52, 88, 75, 40, 90, 58, 82, 48, 93, 68
    ],
    "Technical Skills Score": [
        88, 50, 80, 42, 95, 62, 85, 48, 72, 96,
        53, 90, 77, 38, 94, 60, 84, 45, 91, 70
    ],
    "Placement": [
        "Placed", "Not Placed", "Placed", "Not Placed",
        "Placed", "Not Placed", "Placed", "Not Placed",
        "Placed", "Placed", "Not Placed", "Placed",
        "Placed", "Not Placed", "Placed", "Not Placed",
        "Placed", "Not Placed", "Placed", "Placed"
    ]
}

df = pd.DataFrame(data)
df.to_csv("student placement.csv", index=False)
print("CSV dataset created successfully.")

# 2. Load dataset
df = pd.read_csv("student placement.csv")
print("\nDataset:\n", df)

# 3. Separate features and target
X = df[["CGPA", "Aptitude Score", "Technical Skills Score"]]
y = df["Placement"]

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Create MLflow experiment
mlflow.set_experiment("Student_Placement")

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

print("\nStudent placement experiments completed successfully.")

# py -0p
# py -3.12 --version
# py -3.12 -m pip install pandas scikit-learn mlflow
# py -3.12 -c "import pandas, sklearn, mlflow; print('All libraries installed successfully')"
# cd C:\College\ASD
# py -3.12 student_placement.py
# py -3.12 -m mlflow ui
# http://127.0.0.1:5000