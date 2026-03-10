from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
model_path = os.path.join(
    os.path.dirname(__file__),
    "iris_model.pkl"
)

joblib.dump(model, model_path)

print("Model saved successfully")