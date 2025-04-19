from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

# Create your views here.

# Placeholder dashboard views - we'll implement these later
@login_required
def candidate_dashboard(request):
    if request.user.role != 'Candidate':
        messages.error(request, "Access denied. You are not registered as a job seeker.")
        return redirect('login')
    
    return render(request, 'dashboards/candidate_dashboard.html')

@login_required
def recruiter_dashboard(request):
    if request.user.role != 'Recruiter':
        messages.error(request, "Access denied. You are not registered as a recruiter.")
        return redirect('login')
    
    return render(request, 'dashboards/recruiter_dashboard.html')

@login_required
def admin_dashboard(request):
    if not request.user.is_admin:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('login')
    
    return render(request, 'dashboards/admin_dashboard.html')