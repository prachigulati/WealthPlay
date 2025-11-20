# mentor/urls.py

from django.urls import path
from . import views # Import views from the current directory

urlpatterns = [
    # This pattern handles the fetch request from your JavaScript
    path("respond/", views.mentor_respond, name="mentor_respond"),
    
    # This pattern handles the root path that renders the home.html template
    path("", views.home, name="home"), 
]