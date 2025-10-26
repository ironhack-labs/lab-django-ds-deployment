# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('register/', views.register, name='register'),  # ← ADD THIS
    path('api/predict/', views.api_predict, name='api_predict'),  # ← ADD THIS
]