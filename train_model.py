import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Save model info
model_info = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist()
}

pickle.dump(model_info, open('model_info.pkl', 'wb'))

print("Model and model_info saved successfully!")