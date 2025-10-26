"""
URL configuration for ml_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# In ml_project/urls.py - should look like this:

from django.contrib import admin
from django.urls import path, include  # ← Add 'include' here!
from django.contrib.auth import views as auth_views  # ← ADD THIS
from django.views.generic import RedirectView  # ← ADD THIS

from predictor import views  # Make sure this import exists



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),  # ← Add this line!

# ADD THESE AUTHENTICATION URLS
    path('accounts/login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),
    # CHANGE THIS LINE - Allow GET for logout
    # path('accounts/logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('accounts/logout/', views.custom_logout, name='logout'),
]