from django.urls import path
from accounts.views import (register_view, login_view, logout_view, 
                            candidate_dashboard, recruiter_dashboard, 
                            admin_dashboard)

urlpatterns = [
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/candidate/', candidate_dashboard, name='candidate_dashboard'),
    path('dashboard/recruiter/', recruiter_dashboard, name='recruiter_dashboard'),
    path('dashboard/admin/', admin_dashboard, name='admin_dashboard'),
]