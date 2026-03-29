import os
import joblib
import numpy as np
from django.shortcuts import render
from django.conf import settings

# Load the saved model once at module level
MODEL_PATH = os.path.join(settings.BASE_DIR, "iris_model.pkl")
model = joblib.load(MODEL_PATH)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

FEATURE_FIELDS = [
    ("sepal_length", "Sepal Length (cm)"),
    ("sepal_width", "Sepal Width (cm)"),
    ("petal_length", "Petal Length (cm)"),
    ("petal_width", "Petal Width (cm)"),
]


def home(request):
    return render(request, "predictor/home.html", {"features": FEATURE_FIELDS})


def predict(request):
    if request.method != "POST":
        return render(request, "predictor/home.html", {
            "features": FEATURE_FIELDS,
            "error": "Please submit the form using the button below.",
        })

    errors = []
    values = {}

    for field_name, label in FEATURE_FIELDS:
        raw = request.POST.get(field_name, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a valid number.")
            continue
        if val < 0 or val > 20:
            errors.append(f"{label} must be between 0 and 20.")
            continue
        values[field_name] = val

    if errors:
        return render(request, "predictor/home.html", {
            "features": FEATURE_FIELDS,
            "error": " ".join(errors),
            "submitted": request.POST,
        })

    input_data = np.array([[values[f] for f, _ in FEATURE_FIELDS]])
    prediction = model.predict(input_data)[0]
    predicted_class = IRIS_CLASSES[prediction]

    return render(request, "predictor/result.html", {
        "prediction": predicted_class,
        "features": list(zip(FEATURE_FIELDS, input_data[0])),
    })
