from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),           # Home page with the form
    path('predict/', views.predict, name='predict')  # Prediction result page
]