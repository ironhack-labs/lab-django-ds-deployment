# predictor/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required  # ← ADD THIS
from django.contrib.auth import login, logout
import joblib
import numpy as np
import json
from django.http import JsonResponse  # ← ADD THIS
from django.views.decorators.http import require_http_methods

# Load your trained model (we'll add error handling)
try:
    model = joblib.load('iris_model.pkl')
    # Iris feature names for reference
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_names = ['Setosa', 'Versicolor', 'Virginica']
except:
    model = None
    print("Model not loaded - run train_model.py first!")

def home(request):
    """Display the input form"""
    return render(request, 'predictor/home.html')

def predict(request):
    """Handle form submission and return predictions"""
    if request.method == 'POST':
        try:
            # Get input values from the form
            sepal_length = float(request.POST.get('sepal_length', 0))
            sepal_width = float(request.POST.get('sepal_width', 0))
            petal_length = float(request.POST.get('petal_length', 0))
            petal_width = float(request.POST.get('petal_width', 0))
            
            # Create feature array for prediction
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Make prediction
            if model:
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)) * 100
                species = species_names[prediction]
                
                result = {
                    'species': species,
                    'confidence': round(confidence, 2),
                    'features': {
                        'sepal_length': sepal_length,
                        'sepal_width': sepal_width,
                        'petal_length': petal_length,
                        'petal_width': petal_width,
                    }
                }
            else:
                result = {'error': 'Model not loaded. Please train the model first.'}
                
        except ValueError:
            result = {'error': 'Please enter valid numbers for all fields.'}
        except Exception as e:
            result = {'error': f'Prediction error: {str(e)}'}
        
        return render(request, 'predictor/result.html', {'result': result})
    
    # If not POST request, go back to home
    return render(request, 'predictor/home.html')


def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in after registration
            return redirect('home')
    else:
        form = UserCreationForm()
    
    return render(request, 'predictor/register.html', {'form': form})

@login_required  # ← THIS PROTECTS THE VIEW!
def api_predict(request):
    """Protected REST API endpoint for predictions"""
    if request.method == 'POST':
        try:
            # Get JSON data from API request
            data = json.loads(request.body)
            
            # Extract features
            features = np.array([[
                data.get('sepal_length', 0),
                data.get('sepal_width', 0), 
                data.get('petal_length', 0),
                data.get('petal_width', 0)
            ]])
            
            # Make prediction
            if model:
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)) * 100
                species = species_names[prediction]
                
                response_data = {
                    'species': species,
                    'confidence': round(confidence, 2),
                    'features': {
                        'sepal_length': data.get('sepal_length'),
                        'sepal_width': data.get('sepal_width'),
                        'petal_length': data.get('petal_length'), 
                        'petal_width': data.get('petal_width'),
                    },
                    'user': request.user.username  # Show which user made the request
                }
                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': 'Model not loaded'}, status=500)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

# def custom_logout(request):
#     """Custom logout view that handles both GET and POST"""
#     logout(request)
#     return redirect('home')  # Redirect to home page after logout

@require_http_methods(["GET", "POST"])
def custom_logout(request):
    """Enhanced logout view"""
    print(f"DEBUG: Logout requested via {request.method}")
    
    # Logout the user
    logout(request)
    
    # Clear session completely
    request.session.flush()
    
    print("DEBUG: User logged out and session cleared")
    return redirect('home')