# Django ML Deployment Project - Complete Guide

## 📋 Project Overview
Deploy a machine learning model (Iris classifier) as a web application with user authentication and REST API using Django.

---

## 🛠️ Step 1: Project Setup

### 1.1 Create Project Structure
```bash
# Create project directory
mkdir my_django_ml_project
cd my_django_ml_project

# Create virtual environment (optional - can use existing conda env)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Install Dependencies
```bash
pip install django scikit-learn pandas numpy joblib
```

### 1.3 Create Django Project and App
```bash
# Create Django project
django-admin startproject ml_project .

# Create ML app
python manage.py startapp predictor
```

### 1.4 Project Structure
```
my_django_ml_project/
├── ml_project/                 # Django project config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/                  # ML application
│   ├── migrations/
│   ├── templates/predictor/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── manage.py
└── requirements.txt
```

---

## 🧠 Step 2: Machine Learning Model

### 2.1 Create and Train Model
Create `train_model.py` in the root directory:

```python
# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load and prepare data
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved as iris_model.pkl!")
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")
```

### 2.2 Run Training
```bash
python train_model.py
```

---

## ⚙️ Step 3: Django Configuration

### 3.1 Update Settings
Edit `ml_project/settings.py`:

```python
# Add 'predictor' to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictor',  # ← Add this line
]

# ADD THESE LINES FOR LOGIN REDIRECT
LOGIN_REDIRECT_URL = '/'      # Redirect to home after login
LOGOUT_REDIRECT_URL = '/'     # Redirect to home after logout
```

### 3.2 Run Initial Migrations
```bash
python manage.py migrate
```

---

## 🌐 Step 4: Create Web Views

### 4.1 Update `predictor/views.py`
```python
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
import joblib
import numpy as np
import json
from django.http import JsonResponse

# Load ML model
try:
    model = joblib.load('iris_model.pkl')
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_names = ['Setosa', 'Versicolor', 'Virginica']
except:
    model = None

def home(request):
    return render(request, 'predictor/home.html')

def predict(request):
    if request.method == 'POST':
        try:
            sepal_length = float(request.POST.get('sepal_length', 0))
            sepal_width = float(request.POST.get('sepal_width', 0))
            petal_length = float(request.POST.get('petal_length', 0))
            petal_width = float(request.POST.get('petal_width', 0))
            
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
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
                result = {'error': 'Model not loaded'}
                
        except ValueError:
            result = {'error': 'Please enter valid numbers'}
        except Exception as e:
            result = {'error': f'Prediction error: {str(e)}'}
        
        return render(request, 'predictor/result.html', {'result': result})
    
    return render(request, 'predictor/home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    
    return render(request, 'predictor/register.html', {'form': form})

def custom_logout(request):
    """Custom logout view that redirects to home"""
    logout(request)
    return redirect('home')

@login_required
def api_predict(request):
    """Protected REST API endpoint for predictions"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = np.array([[
                data.get('sepal_length', 0),
                data.get('sepal_width', 0), 
                data.get('petal_length', 0),
                data.get('petal_width', 0)
            ]])
            
            if model:
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)) * 100
                species = species_names[prediction]
                
                response_data = {
                    'species': species,
                    'confidence': round(confidence, 2),
                    'features': data,
                    'user': request.user.username
                }
                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': 'Model not loaded'}, status=500)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
```

---

## 🔗 Step 5: URL Configuration

### 5.1 Main URLs (`ml_project/urls.py`)
```python
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),
    path('accounts/logout/', views.custom_logout, name='logout'),  # Uses our custom logout
]
```

### 5.2 App URLs (`predictor/urls.py`)
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('register/', views.register, name='register'),
    path('api/predict/', views.api_predict, name='api_predict'),
]
```

---

## 🎨 Step 6: HTML Templates

### 6.1 Base Template (`predictor/templates/predictor/base.html`)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Species Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'home' %}">🌺 Iris Predictor</a>
            <div class="navbar-nav ms-auto">
                {% if user.is_authenticated %}
                    <span class="navbar-text me-3">Hello, {{ user.username }}!</span>
                    <a class="nav-link" href="{% url 'logout' %}">Logout</a>
                {% else %}
                    <a class="nav-link" href="{% url 'login' %}">Login</a>
                    <a class="nav-link" href="{% url 'register' %}">Register</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1 class="text-center mb-4">🌺 Iris Flower Classification</h1>
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

### 6.2 Home Template (`predictor/templates/predictor/home.html`)
```html
{% extends "predictor/base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Enter Flower Measurements</h4>
            </div>
            <div class="card-body">
                <form method="POST" action="{% url 'predict' %}">
                    {% csrf_token %}
                    
                    <div class="mb-3">
                        <label for="sepal_length" class="form-label">Sepal Length (cm)</label>
                        <input type="number" step="0.1" class="form-control" id="sepal_length" 
                               name="sepal_length" required min="4" max="8" value="5.8">
                    </div>
                    
                    <div class="mb-3">
                        <label for="sepal_width" class="form-label">Sepal Width (cm)</label>
                        <input type="number" step="0.1" class="form-control" id="sepal_width" 
                               name="sepal_width" required min="2" max="4.5" value="3.0">
                    </div>
                    
                    <div class="mb-3">
                        <label for="petal_length" class="form-label">Petal Length (cm)</label>
                        <input type="number" step="0.1" class="form-control" id="petal_length" 
                               name="petal_length" required min="1" max="7" value="4.0">
                    </div>
                    
                    <div class="mb-3">
                        <label for="petal_width" class="form-label">Petal Width (cm)</label>
                        <input type="number" step="0.1" class="form-control" id="petal_width" 
                               name="petal_width" required min="0.1" max="2.5" value="1.2">
                    </div>
                    
                    <button type="submit" class="btn btn-primary w-100">Predict Species</button>
                </form>
            </div>
        </div>
        
        <!-- API Info Section -->
        <div class="card mt-4">
            <div class="card-header">
                <h5 class="mb-0">🔌 REST API Access</h5>
            </div>
            <div class="card-body">
                <p>Authenticated users can access the prediction API:</p>
                <code>POST /api/predict/</code>
                <pre class="mt-2 p-2 bg-light">
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}</pre>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### 6.3 Result Template (`predictor/templates/predictor/result.html`)
```html
{% extends "predictor/base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        {% if result.error %}
            <div class="alert alert-danger">
                <h4>Error</h4>
                <p>{{ result.error }}</p>
            </div>
        {% else %}
            <div class="card">
                <div class="card-header bg-success text-white">
                    <h4 class="mb-0">Prediction Result</h4>
                </div>
                <div class="card-body text-center">
                    <h2>🌺 {{ result.species }}</h2>
                    <p class="lead">Confidence: {{ result.confidence }}%</p>
                    
                    <hr>
                    <h5>Input Features:</h5>
                    <ul class="list-unstyled">
                        <li>Sepal Length: {{ result.features.sepal_length }} cm</li>
                        <li>Sepal Width: {{ result.features.sepal_width }} cm</li>
                        <li>Petal Length: {{ result.features.petal_length }} cm</li>
                        <li>Petal Width: {{ result.features.petal_width }} cm</li>
                    </ul>
                </div>
            </div>
        {% endif %}
        
        <div class="mt-3 text-center">
            <a href="{% url 'home' %}" class="btn btn-secondary">Make Another Prediction</a>
        </div>
    </div>
</div>
{% endblock %}
```

### 6.4 Login Template (`predictor/templates/predictor/login.html`)
```html
{% extends "predictor/base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Login Required</h4>
            </div>
            <div class="card-body">
                {% if form.errors %}
                <div class="alert alert-danger">
                    Invalid username or password. Please try again.
                </div>
                {% endif %}

                <form method="post">
                    {% csrf_token %}
                    
                    <div class="mb-3">
                        <label for="id_username" class="form-label">Username</label>
                        <input type="text" name="username" class="form-control" id="id_username" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="id_password" class="form-label">Password</label>
                        <input type="password" name="password" class="form-control" id="id_password" required>
                    </div>
                    
                    <button type="submit" class="btn btn-primary w-100">Login</button>
                </form>
                
                <div class="mt-3 text-center">
                    <a href="{% url 'register' %}">Don't have an account? Register here</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### 6.5 Register Template (`predictor/templates/predictor/register.html`)
```html
{% extends "predictor/base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Create Account</h4>
            </div>
            <div class="card-body">
                <form method="post">
                    {% csrf_token %}
                    
                    {% for field in form %}
                    <div class="mb-3">
                        <label for="{{ field.id_for_label }}" class="form-label">{{ field.label }}</label>
                        {{ field }}
                        {% if field.help_text %}
                        <div class="form-text">{{ field.help_text }}</div>
                        {% endif %}
                        {% for error in field.errors %}
                        <div class="text-danger">{{ error }}</div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                    
                    <button type="submit" class="btn btn-success w-100">Register</button>
                </form>
                
                <div class="mt-3 text-center">
                    <a href="{% url 'login' %}">Already have an account? Login here</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

---

## 👤 Step 7: User Management

### 7.1 Create Superuser
```bash
python manage.py createsuperuser
```
Follow prompts to create admin account.

### 7.2 Test Authentication
1. Register a new user
2. Login with credentials - should redirect to home page
3. Test logout functionality - should redirect to home page

---

## 🔌 Step 8: REST API Access

### 8.1 API Endpoint Details
- **URL**: `POST /api/predict/`
- **Authentication**: Required (user must be logged in)
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

### 8.2 How to Test the API

#### Method 1: Browser Console (Easiest)
1. **Login** through the web interface
2. **Open Developer Tools** (F12)
3. Go to **Console** tab
4. **Run this JavaScript**:

```javascript
// Test API from browser console
fetch('/api/predict/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
    },
    body: JSON.stringify({
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
    })
})
.then(response => response.json())
.then(data => console.log('API Response:', data))
.catch(error => console.error('Error:', error));
```

#### Method 2: Python Script
Create `test_api.py`:

```python
import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
SESSION = requests.Session()

def login(username, password):
    """Login to get session cookie"""
    # Get CSRF token
    login_page = SESSION.get(f"{BASE_URL}/accounts/login/")
    csrf_token = SESSION.cookies.get('csrftoken')
    
    # Login data
    login_data = {
        'username': username,
        'password': password,
        'csrfmiddlewaretoken': csrf_token
    }
    
    # Perform login
    login_response = SESSION.post(
        f"{BASE_URL}/accounts/login/",
        data=login_data,
        headers={'Referer': f"{BASE_URL}/accounts/login/"}
    )
    
    return login_response.status_code == 200

def test_api():
    """Test the protected API endpoint"""
    api_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    # Make API request
    response = SESSION.post(
        f"{BASE_URL}/api/predict/",
        json=api_data,
        headers={'X-CSRFToken': SESSION.cookies.get('csrftoken')}
    )
    
    print(f"Status Code: {response.status_code}")
    print("Response:", json.dumps(response.json(), indent=2))

# Usage
if __name__ == "__main__":
    # Replace with your credentials
    if login('your_username', 'your_password'):
        test_api()
    else:
        print("Login failed!")
```

#### Method 3: Using curl
1. **Login** through web interface first
2. **Get session cookie** from browser Developer Tools
3. **Run curl command**:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -H "Cookie: sessionid=YOUR_SESSION_ID_HERE" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### 8.3 Expected API Response
```json
{
    "species": "Setosa",
    "confidence": 95.23,
    "features": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },
    "user": "demo"
}
```

---

## 🚀 Step 9: Run and Test

### 9.1 Start Development Server
```bash
python manage.py runserver
```

### 9.2 Test All Features
1. **Homepage**: http://localhost:8000/
2. **Registration**: http://localhost:8000/register/
3. **Login**: http://localhost:8000/accounts/login/
4. **Web Prediction**: Use the form
5. **API**: Test with methods above

### 9.3 Expected Behavior
- Unauthenticated users see Login/Register
- Login redirects to home page (not /accounts/profile/)
- Logout redirects to home page
- Authenticated users can use prediction features
- API endpoints require authentication
- Clean, responsive Bootstrap interface

---

## 📁 Final Project Structure
```
my_django_ml_project/
├── ml_project/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── predictor/
│   ├── templates/predictor/
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── result.html
│   │   ├── login.html
│   │   └── register.html
│   ├── views.py
│   ├── urls.py
│   └── ...
├── iris_model.pkl
├── train_model.py
├── test_api.py
├── manage.py
└── db.sqlite3
```

---

## ✅ Deliverables Checklist
- [ ] Working web interface with forms
- [ ] User authentication (register/login/logout)
- [ ] Proper login/logout redirects
- [ ] Protected REST API endpoint
- [ ] Machine learning model integration
- [ ] Error handling for invalid inputs
- [ ] Bootstrap styling
- [ ] Proper project structure
- [ ] API documentation and testing methods

---

## 🎯 Common Issues & Solutions

### Issue: "no such table: auth_user"
**Solution**: Run `python manage.py migrate`

### Issue: "Method Not Allowed (GET): /accounts/logout/"
**Solution**: Use the custom logout view provided

### Issue: "Page not found: /accounts/profile/" after login
**Solution**: Add `LOGIN_REDIRECT_URL = '/'` to settings.py

### Issue: Model not loading
**Solution**: Ensure `iris_model.pkl` exists in project root

### Issue: API returns 403 Forbidden
**Solution**: Ensure user is logged in and include CSRF token

### Issue: Template not found
**Solution**: Check template paths and ensure app is in `INSTALLED_APPS`

---

## 🔧 Key Configuration Points

### Login/Logout Settings (Critical!)
```python
# In ml_project/settings.py
LOGIN_REDIRECT_URL = '/'      # After login, go to home page
LOGOUT_REDIRECT_URL = '/'     # After logout, go to home page
```

### Custom Logout View
```python
# In predictor/views.py
def custom_logout(request):
    logout(request)
    return redirect('home')  # Explicit redirect
```

### URL Configuration
```python
# In ml_project/urls.py
path('accounts/logout/', views.custom_logout, name='logout'),
```

This guide provides a complete walkthrough of deploying a machine learning model with Django, including user authentication, proper login/logout behavior, and REST API functionality!