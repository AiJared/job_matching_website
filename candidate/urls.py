# candidate/urls.py
from django.urls import path
from . import views

app_name = 'candidate'

urlpatterns = [
    path('candidate_dashboard/', views.candidate_dashboard, name='candidate_dashboard'),
    path('complete_profile/', views.complete_profile, name='complete_profile'),
    path('resume/upload/', views.upload_resume, name='upload_resume'),
    path('jobs/recommendations/', views.job_recommendations, name='job_recommendations'),
    path('jobs/search/', views.job_search, name='job_search'),
    path('jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<int:job_id>/save/', views.toggle_save_job, name='toggle_save_job'),
    path('applications/', views.application_list, name='application_list'),
    path('applications/<int:application_id>/', views.application_detail, name='application_detail'),
    path('saved-jobs/', views.saved_jobs, name='saved_jobs'),
]