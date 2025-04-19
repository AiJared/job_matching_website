from django.urls import path
from accounts.views import (register_view, login_view, logout_view, 
                            candidate_dashboard, recruiter_dashboard, 
                            admin_dashboard, home)

app_name = "accounts"

urlpatterns = [
    path('', home, name='home'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
]