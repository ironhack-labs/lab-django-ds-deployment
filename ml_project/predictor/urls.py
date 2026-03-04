# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),           # Home page with input form
    path('predict/', views.predict, name='predict'),  # Prediction endpoint
]