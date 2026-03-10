from django.shortcuts import render
import joblib
import numpy as np
import os


# Load model once when Django starts
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model/iris_model.pkl"
)

model = joblib.load(MODEL_PATH)


# Home page (input form)
def home(request):
    return render(request, "index.html")


# Prediction view
def predict(request):

    if request.method != "POST":
        return render(request, "index.html")

    try:
        # Get user inputs
        sepal_length = float(request.POST.get("sepal_length"))
        sepal_width = float(request.POST.get("sepal_width"))
        petal_length = float(request.POST.get("petal_length"))
        petal_width = float(request.POST.get("petal_width"))

        features = np.array([[

            sepal_length,
            sepal_width,
            petal_length,
            petal_width

        ]])

        prediction = model.predict(features)[0]

        species_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        result = species_map.get(prediction, "Unknown")

        return render(
            request,
            "result.html",
            {"prediction": result}
        )

    except ValueError:

        return render(
            request,
            "index.html",
            {"error": "Please enter valid numeric values."}
        )

    except Exception as e:

        return render(
            request,
            "index.html",
            {"error": f"Unexpected error: {str(e)}"}
        )