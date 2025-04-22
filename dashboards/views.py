# dashboards/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.urls import reverse
from django.utils import timezone
from django.db.models import Count, Avg
from django.http import JsonResponse
from django.core.paginator import Paginator

from dashboards.models import JobPosting, JobApplication
from dashboards.forms import JobPostingForm, ApplicationStatusUpdateForm
from dashboards.ai_utils import generate_job_embedding, update_job_matches

def is_recruiter(user):
    return user.is_authenticated and user.role == 'Recruiter'

@login_required
@user_passes_test(is_recruiter)
def recruiter_dashboard(request):
    """Main recruiter dashboard view"""
    recruiter_profile = request.user.recruiter
    
    # Get statistics for dashboard
    active_jobs = JobPosting.objects.filter(
        recruiter=recruiter_profile,
        status='active'
    ).count()
    
    total_applications = JobApplication.objects.filter(
        job__recruiter=recruiter_profile
    ).count()
    
    pending_applications = JobApplication.objects.filter(
        job__recruiter=recruiter_profile,
        status='pending'
    ).count()
    
    # Get latest 5 job postings
    latest_jobs = JobPosting.objects.filter(
        recruiter=recruiter_profile
    ).order_by('-created_at')[:5]
    
    # Get latest 5 applications
    latest_applications = JobApplication.objects.filter(
        job__recruiter=recruiter_profile
    ).order_by('-created_at')[:5]
    
    context = {
        'active_jobs': active_jobs,
        'total_applications': total_applications,
        'pending_applications': pending_applications,
        'latest_jobs': latest_jobs,
        'latest_applications': latest_applications,
    }
    
    return render(request, 'dashboards/recruiter_dashboard.html', context)

@login_required
@user_passes_test(is_recruiter)
def job_list(request):
    """View all jobs posted by the recruiter"""
    recruiter_profile = request.user.recruiter
    
    jobs = JobPosting.objects.filter(recruiter=recruiter_profile).order_by('-created_at')
    
    # Add application counts to each job
    for job in jobs:
        job.application_count = JobApplication.objects.filter(job=job).count()
        job.pending_count = JobApplication.objects.filter(job=job, status='pending').count()
    
    # Pagination
    paginator = Paginator(jobs, 10)  # Show 10 jobs per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
    }
    
    return render(request, 'dashboards/job_list.html', context)

@login_required
@user_passes_test(is_recruiter)
def create_job(request):
    """Create a new job posting"""
    if request.method == 'POST':
        form = JobPostingForm(request.POST)
        if form.is_valid():
            job_posting = form.save(commit=False)
            job_posting.recruiter = request.user.recruiter
            job_posting.status = 'active'
            job_posting.save()
            
            # Generate and save embedding vector
            try:
                job_embedding = generate_job_embedding(job_posting)
                if job_embedding is not None:
                    job_posting.embedding_vector = job_embedding.tobytes()
                    job_posting.save()
                    
                    # Update match scores for existing resumes
                    update_job_matches(job_posting)
            except Exception as e:
                messages.warning(request, f"Job created but AI matching had an error: {str(e)}")
            
            messages.success(request, "Job posting created successfully!")
            return redirect('dashboards:job_detail', job_id=job_posting.id)
    else:
        form = JobPostingForm()
    
    context = {
        'form': form,
    }
    
    return render(request, 'dashboards/create_job.html', context)

@login_required
@user_passes_test(is_recruiter)
def edit_job(request, job_id):
    """Edit an existing job posting"""
    job_posting = get_object_or_404(JobPosting, id=job_id, recruiter=request.user.recruiter)
    
    if request.method == 'POST':
        form = JobPostingForm(request.POST, instance=job_posting)
        if form.is_valid():
            job_posting = form.save()
            
            # Re-generate embedding vector if job details changed
            try:
                job_embedding = generate_job_embedding(job_posting)
                if job_embedding is not None:
                    job_posting.embedding_vector = job_embedding.tobytes()
                    job_posting.save()
                    
                    # Update match scores
                    update_job_matches(job_posting)
            except Exception as e:
                messages.warning(request, f"Job updated but AI matching had an error: {str(e)}")
            
            messages.success(request, "Job posting updated successfully!")
            return redirect('dashboards:job_detail', job_id=job_posting.id)
    else:
        form = JobPostingForm(instance=job_posting)
    
    context = {
        'form': form,
        'job': job_posting,
        'categories': JobCategory.objects.all(),
    }
    
    return render(request, 'dashboards/edit_job.html', context)

@login_required
@user_passes_test(is_recruiter)
def job_detail(request, job_id):
    """View details of a specific job posting"""
    job_posting = get_object_or_404(JobPosting, id=job_id, recruiter=request.user.recruiter)
    
    # Get all applications for this job
    applications = JobApplication.objects.filter(job=job_posting).order_by('-match_score')
    
    # Get application statistics
    application_stats = {
        'total': applications.count(),
        'pending': applications.filter(status='pending').count(),
        'reviewing': applications.filter(status='reviewing').count(),
        'shortlisted': applications.filter(status='shortlisted').count(),
        'interview': applications.filter(status='interview').count(),
        'offer': applications.filter(status='offer').count(),
        'hired': applications.filter(status='hired').count(),
        'rejected': applications.filter(status='rejected').count(),
    }
    
    context = {
        'job': job_posting,
        'applications': applications,
        'stats': application_stats,
    }
    
    return render(request, 'dashboards/job_detail.html', context)

@login_required
@user_passes_test(is_recruiter)
def change_job_status(request, job_id):
    """Change the status of a job posting (active/filled/expired)"""
    job_posting = get_object_or_404(JobPosting, id=job_id, recruiter=request.user.recruiter)
    
    if request.method == 'POST':
        new_status = request.POST.get('status')
        if new_status in dict(JobPosting.STATUS_CHOICES).keys():
            job_posting.status = new_status
            job_posting.save()
            messages.success(request, f"Job status updated to {new_status}.")
        else:
            messages.error(request, "Invalid status selected.")
        
        return redirect('dashboards:job_detail', job_id=job_posting.id)

@login_required
@user_passes_test(is_recruiter)
def application_detail(request, application_id):
    """View details of a specific application"""
    application = get_object_or_404(
        JobApplication, 
        id=application_id, 
        job__recruiter=request.user.recruiter
    )
    
    if request.method == 'POST':
        form = ApplicationStatusUpdateForm(request.POST, instance=application)
        if form.is_valid():
            form.save()
            messages.success(request, "Application status updated successfully!")
            return redirect('dashboards:application_detail', application_id=application.id)
    else:
        form = ApplicationStatusUpdateForm(instance=application)
    
    context = {
        'application': application,
        'form': form,
    }
    
    return render(request, 'dashboards/application_detail.html', context)

@login_required
@user_passes_test(is_recruiter)
def applications_list(request):
    """View all applications for all jobs posted by the recruiter"""
    recruiter_profile = request.user.recruiter
    
    # Filter applications
    status_filter = request.GET.get('status', '')
    job_filter = request.GET.get('job', '')
    
    applications = JobApplication.objects.filter(job__recruiter=recruiter_profile)
    
    if status_filter:
        applications = applications.filter(status=status_filter)
    
    if job_filter:
        applications = applications.filter(job_id=job_filter)
    
    applications = applications.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(applications, 15)  # Show 15 applications per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get filtering options
    jobs = JobPosting.objects.filter(recruiter=recruiter_profile)
    status_choices = JobApplication.STATUS_CHOICES
    
    context = {
        'page_obj': page_obj,
        'jobs': jobs,
        'status_choices': status_choices,
        'current_job': job_filter,
        'current_status': status_filter,
    }
    
    return render(request, 'dashboards/applications_list.html', context)