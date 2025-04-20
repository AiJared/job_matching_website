# dashboards/urls.py
from django.urls import path
from . import views

app_name = 'dashboards'

urlpatterns = [
    # Recruiter dashboard URLs
    path('recruiter/', views.recruiter_dashboard, name='recruiter_dashboard'),
    path('recruiter/jobs/', views.job_list, name='job_list'),
    path('recruiter/jobs/create/', views.create_job, name='create_job'),
    path('recruiter/jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    path('recruiter/jobs/<int:job_id>/edit/', views.edit_job, name='edit_job'),
    path('recruiter/jobs/<int:job_id>/status/', views.change_job_status, name='change_job_status'),
    path('recruiter/applications/', views.applications_list, name='applications_list'),
    path('recruiter/applications/<int:application_id>/', views.application_detail, name='application_detail'),
    
    # Placeholder for candidate dashboard URLs - we'll implement these later
    # path('candidate/', views.candidate_dashboard, name='candidate_dashboard'),
]

# Add a placeholder for the candidate dashboard view to avoid URL errors
# dashboards/views.py (additional function)
# @login_required
# def candidate_dashboard(request):
#     """Placeholder for candidate dashboard - will be implemented later"""
#     return render(request, 'dashboards/candidate/dashboard.html')