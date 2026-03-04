import os
import pickle
from django.shortcuts import render
from django.conf import settings

# Try to load the trained model from the repository `models` folder.
# The repository layout places `models/model.pkl` as a sibling of the
# Django project folder, so we resolve the path relative to BASE_DIR.
MODEL_PATH = settings.BASE_DIR.parent / 'models' / 'model.pkl'
model = None
model_load_error = None
if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = f'Model file not found at {MODEL_PATH}'

# load model info (feature/target names) if available
MODEL_INFO_PATH = settings.BASE_DIR.parent / 'models' / 'model_info.pkl'
model_info = {}
if MODEL_INFO_PATH.exists():
    try:
        with open(MODEL_INFO_PATH, 'rb') as f:
            model_info = pickle.load(f)
    except Exception:
        model_info = {}


def home(request):
    return render(request, 'predictor/home.html')


def predict(request):
    if request.method == 'POST':
        # Read features from the submitted form. Adjust these names to
        # match the form fields and the model's expected input order.
        try:
            f1 = float(request.POST.get('feature1', 0))
            f2 = float(request.POST.get('feature2', 0))
            f3 = float(request.POST.get('feature3', 0))
            f4 = float(request.POST.get('feature4', 0))
        except ValueError:
            return render(request, 'predictor/predict.html', {
                'error': 'Invalid input: please submit numeric values for the features.'
            })

        if model is None:
            return render(request, 'predictor/predict.html', {
                'error': f'Model not available: {model_load_error}'
            })

        # Prepare input for the model. Many scikit-learn models accept a 2D
        # array-like of shape (n_samples, n_features).
        inputs = [[f1, f2, f3, f4]]
        try:
            # prefer probabilistic output so we can show class name + confidence
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(inputs)
                prob = float(max(probs[0]))
                pred_idx = int(probs[0].argmax())
                # try to map to target name from model_info
                target_names = model_info.get('target_names') or None
                if target_names and len(target_names) > pred_idx:
                    pred_name = target_names[pred_idx]
                else:
                    pred_name = str(pred_idx)
                prediction = f'{pred_name} ({prob*100:.2f}%)'
            else:
                pred = model.predict(inputs)
                if hasattr(pred, '__iter__'):
                    prediction_value = float(pred[0])
                else:
                    prediction_value = float(pred)
                prediction = f'{prediction_value:.4f}'
        except Exception as e:
            prediction = None
            return render(request, 'predictor/predict.html', {
                'error': f'Prediction failed: {e}'
            })

        return render(request, 'predictor/predict.html', { 'prediction': prediction })

    return render(request, 'predictor/predict.html')