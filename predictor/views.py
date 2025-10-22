from django.shortcuts import render
import joblib

# Load model
model = joblib.load('model.pkl')

def home(request):
    # This will render the input form
    return render(request, 'predictor/home.html')

def predict(request):
    # This will process input and return result - logic comes later
    if request.method == 'POST':
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
        
        prediction = model.predict(input_features)[0]

        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map[prediction]

        return render(request, 'predictor/result.html', {'prediction': predicted_species})


    return render(request, 'predictor/home.html')
