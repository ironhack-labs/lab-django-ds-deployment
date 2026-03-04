# predictor/views.py
from django.shortcuts import render
import joblib
import numpy as np
import os

# Load the ML model from ../model.pkl
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
model = joblib.load(MODEL_PATH)

# List all 4 features your model expects (must match training order)
FEATURES = ['feature1', 'feature2', 'feature3', 'feature4']

def home(request):
    """
    Display form and handle prediction on POST.
    """
    context = {}
    # Initialize input values for pre-filling
    context['inputs'] = {f: '' for f in FEATURES}

    if request.method == "POST":
        input_values = []

        # Collect input values and validate
        for feature in FEATURES:
            value = request.POST.get(feature, '').strip()
            context['inputs'][feature] = value  # prefill
            if value == '':
                context['error'] = f"Please provide a value for '{feature}'."
                return render(request, 'predictor/home.html', context)
            try:
                input_values.append(float(value))
            except ValueError:
                context['error'] = f"Invalid value for '{feature}': must be a number."
                return render(request, 'predictor/home.html', context)

        # Make prediction
        features_array = np.array([input_values])
        try:
            prediction = model.predict(features_array)[0]
            context['prediction'] = prediction
        except Exception as e:
            context['error'] = f"Error during prediction: {e}"

    return render(request, 'predictor/home.html', context)


def predict(request):
    """
    Handle form submission, validate input, and return prediction.
    Includes debug/status messages in context for troubleshooting.
    """
    context = {}
    context['status'] = []  # List to collect debug info

    if request.method == "POST":
        input_values = []
        context['status'].append("POST request received.")

        # Validate and collect features
        for feature in FEATURES:
            value = request.POST.get(feature)
            context['status'].append(f"Received input for {feature}: {value}")
            if value is None or value.strip() == '':
                context['error'] = f"Please provide a value for '{feature}'."
                context['status'].append(f"Missing value for {feature}.")
                return render(request, 'predictor/home.html', context)
            try:
                float_value = float(value)
                input_values.append(float_value)
                context['status'].append(f"Parsed {feature} as float: {float_value}")
            except ValueError:
                context['error'] = f"Invalid value for '{feature}': must be a number."
                context['status'].append(f"Failed to parse {feature} as float.")
                return render(request, 'predictor/home.html', context)

        # Convert to numpy array
        features_array = np.array([input_values])
        context['status'].append(f"Feature array for model: {features_array}")

        try:
            # Make prediction
            prediction = model.predict(features_array)[0]
            context['prediction'] = prediction
            context['status'].append(f"Model prediction: {prediction}")
        except Exception as e:
            context['error'] = f"Error during prediction: {e}"
            context['status'].append(f"Exception during prediction: {e}")

    else:
        context['status'].append("GET request received. Displaying empty form.")

    return render(request, 'predictor/home.html', context)