from django.shortcuts import render
import joblib
import numpy as np

model = joblib.load('model.pkl')

def home(request):
    return render(request, 'predictor/home.html')

def predict(request):
    if request.method == 'POST':
        try:
            data = [
                float(request.POST['sepal_length']),
                float(request.POST['sepal_width']),
                float(request.POST['petal_length']),
                float(request.POST['petal_width'])
            ]
            prediction = model.predict([data])[0]
            species = ['Setosa', 'Versicolor', 'Virginica'][prediction]
            return render(request, 'predictor/result.html', {'prediction': species})
        except Exception as e:
            return render(request, 'predictor/result.html', {'prediction': f'Error: {e}'})
    return render(request, 'predictor/home.html')