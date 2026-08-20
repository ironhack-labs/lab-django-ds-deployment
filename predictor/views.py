from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import joblib
import os


# Load the trained machine learning model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "iris_model.pkl"
)

model = joblib.load(MODEL_PATH)


# Iris flower names
IRIS_CLASSES = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]


def home(request):
    """
    Display the prediction form.
    """
    return render(request, "predictor/home.html")


def predict(request):
    """
    Receive user input and return the Iris prediction.
    """

    if request.method == "POST":

        try:
            # Get values from the form
            sepal_length = float(request.POST.get("sepal_length"))
            sepal_width = float(request.POST.get("sepal_width"))
            petal_length = float(request.POST.get("petal_length"))
            petal_width = float(request.POST.get("petal_width"))

            # Validate that measurements are positive
            if (
                sepal_length <= 0
                or sepal_width <= 0
                or petal_length <= 0
                or petal_width <= 0
            ):
                raise ValueError("Measurements must be greater than zero.")

            # Create input for the model
            features = [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]]

            # Make prediction
            prediction = model.predict(features)[0]

            # Convert prediction number to flower name
            prediction_name = IRIS_CLASSES[prediction]

            return render(
                request,
                "predictor/result.html",
                {
                    "prediction": prediction_name,
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                }
            )

        except (ValueError, TypeError):
            error_message = (
                "Please enter valid positive numbers "
                "for all measurements."
            )

            return render(
                request,
                "predictor/home.html",
                {
                    "error": error_message
                }
            )

    return render(request, "predictor/home.html")