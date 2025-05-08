# candidate/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from django.utils import timezone
from django.db.models import Q
from numpy import frombuffer

from accounts.models import Candidate
from dashboards.models import Resume, JobPosting, JobApplication
from .models import SavedJob, JobSearchHistory, ProfileCompletionTask
from .forms import CandidateProfileForm, ResumeUploadForm, JobApplicationForm, JobSearchForm
from .utils import process_resume, get_recommended_jobs, get_profile_completion_percentage, update_candidate_matches
from dashboards.ai_utils import predict_specific_job_candidate_match


def is_candidate(user):
    """Check if user is a candidate"""
    return user.is_authenticated and user.role == 'Candidate'

@login_required
@user_passes_test(is_candidate)
def complete_profile(request):
    """View for candidates to complete their profile (skills, education, experience)"""
    candidate = request.user.candidate

    if request.method == 'POST':
        form = CandidateProfileForm(request.POST, instance=candidate)
        if form.is_valid():
            form.save()

            # Auto-sync Resume model fields
            try:
                resume = Resume.objects.get(candidate=candidate)

                resume.skills = candidate.skills
                resume.education = candidate.education
                resume.experience = candidate.experience

                # Reset processed embedding
                resume.embedding_vector = None
                resume.is_processed = False

                resume.save()

                # Trigger re-processing embedding
                process_resume(resume.id)

                messages.success(request, "Your profile and resume have been updated and reprocessed successfully!")
            except Resume.DoesNotExist:
                # No resume uploaded yet; nothing to sync
                messages.success(request, "Your profile has been updated successfully!")

            return redirect('candidate:candidate_dashboard')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = CandidateProfileForm(instance=candidate)

    return render(request, 'candidate/complete_profile.html', {'form': form})


# 🧠 Class to wrap tuple (job, score) for Django template compatibility
class JobRecommendation:
    def __init__(self, job, match_score):
        self.job = job
        self.match_score = match_score

@login_required
@user_passes_test(is_candidate)
def candidate_dashboard(request):
    """Main candidate dashboard view"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')  # Redirect to profile creation page
    
    # Check if resume exists
    try:
        resume = Resume.objects.get(candidate=candidate)
        has_resume = True
        resume_processed = resume.is_processed  # ✅ Use the boolean flag we manually set
    except Resume.DoesNotExist:
        has_resume = False
        resume_processed = False


    # Profile completion logic
    completion_percentage = get_profile_completion_percentage(candidate)
    
    # Job recommendations (fixing the structure)
    recommended_jobs = []
    if has_resume and resume_processed:
        job_matches = get_recommended_jobs(candidate, limit=5)
        recommended_jobs = [JobRecommendation(job, score) for job, score in job_matches]

    # Applications and saved jobs
    recent_applications = JobApplication.objects.filter(
        candidate=candidate
    ).order_by('-created_at')[:5]

    saved_jobs = SavedJob.objects.filter(
        candidate=candidate
    ).order_by('-saved_at')[:5]

    context = {
        'has_resume': has_resume,
        'resume_processed': resume_processed,
        'completion_percentage': completion_percentage,
        'recommended_jobs': recommended_jobs,
        'recent_applications': recent_applications,
        'saved_jobs': saved_jobs,
    }

    return render(request, 'candidate/candidate_dashboard.html', context)

@login_required
@user_passes_test(is_candidate)
def upload_resume(request):
    """View for uploading a resume"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    # Profile Completion Enforcement
    if not candidate.skills or not candidate.education or not candidate.experience:
        messages.warning(request, "Please complete your profile before uploading your resume.")
        return redirect('candidate:complete_profile')
    
    # Check if resume already exists
    try:
        resume = Resume.objects.get(candidate=candidate)
        is_update = True
    except Resume.DoesNotExist:
        resume = None
        is_update = False
    
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES, instance=resume)
        if form.is_valid():
            resume = form.save(commit=False)
            resume.candidate = candidate
            # Reset processed fields if file is being updated
            if is_update:
                resume.embedding_vector = None
                resume.is_processed = False
            resume.save()
            
            # Process resume
            process_resume(resume.id)
            
            # Update job matches
            update_candidate_matches(candidate)
            
            messages.success(request, "Your resume has been uploaded and is being processed.")
            return redirect('candidate:candidate_dashboard')
    else:
        form = ResumeUploadForm(instance=resume)
    
    context = {
        'form': form,
        'is_update': is_update,
    }
    
    return render(request, 'candidate/upload_resume.html', context)

@login_required
@user_passes_test(is_candidate)
def job_recommendations(request):
    """View for displaying job recommendations"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    # Check if resume exists and is processed
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            messages.warning(request, "Your resume is still being processed. Please check back later.")
            return redirect('candidate:candidate_dashboard')
    except Resume.DoesNotExist:
        messages.error(request, "Please upload your resume first.")
        return redirect('candidate:upload_resume')
    
    # Get recommended jobs
    job_matches = get_recommended_jobs(candidate)
    recommended_jobs = [{'job': job, 'match_score': round(score, 1)} for job, score in job_matches]
    
    # 🛠 Fix: define saved_job_ids
    saved_job_ids = SavedJob.objects.filter(candidate=candidate).values_list('job_id', flat=True)

    context = {
        'recommended_jobs': recommended_jobs,
        'saved_job_ids': list(saved_job_ids),  # convert QuerySet to list
    }
    
    return render(request, 'candidate/job_recommendations.html', context)

@login_required
@user_passes_test(is_candidate)
def job_search(request):
    """View for searching jobs"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    form = JobSearchForm(request.GET or None)
    jobs = JobPosting.objects.filter(status='active')
    
    if form.is_valid():
        query = form.cleaned_data.get('query')
        location = form.cleaned_data.get('location')
        category = form.cleaned_data.get('category')
        
        # Apply filters
        if query:
            jobs = jobs.filter(
                Q(title__icontains=query) | 
                Q(company_name__icontains=query) |
                Q(description__icontains=query) |
                Q(skills_required__icontains=query)
            )
        
        if location:
            jobs = jobs.filter(location__icontains=location)
        
        if category:
            jobs = jobs.filter(category__icontains=category)
        
        # Save search to history if not empty
        if query or location or category:
            JobSearchHistory.objects.create(
                candidate=candidate,
                query=query or "",
                location=location or "",
                category=category or ""
            )
    
    # Sort by date (newest first)
    jobs = jobs.order_by('-created_at')
    
    # Calculate match scores if resume exists
    try:
        resume = Resume.objects.get(candidate=candidate)
        has_resume = True
        
        if resume.embedding_vector:
            # Add match score to each job
            jobs_with_scores = []
            for job in jobs:
                match_score = 0
                if job.embedding_vector:
                    from dashboards.ai_utils import calculate_match_score
                    match_score = calculate_match_score(job.embedding_vector, resume.embedding_vector)
                jobs_with_scores.append((job, round(match_score, 1)))
            
            # Sort by match score if search query is empty
            if not any(form.cleaned_data.values()):
                jobs_with_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            jobs_with_scores = [(job, 0) for job in jobs]
    except Resume.DoesNotExist:
        has_resume = False
        jobs_with_scores = [(job, 0) for job in jobs]
    
    # Get recent searches
    recent_searches = JobSearchHistory.objects.filter(
        candidate=candidate
    ).order_by('-timestamp')[:5]

    # Get list of saved job IDs
    saved_job_ids = SavedJob.objects.filter(candidate=candidate).values_list('job_id', flat=True)
    
    context = {
        'form': form,
        'jobs_with_scores': jobs_with_scores,
        'has_resume': has_resume,
        'recent_searches': recent_searches,
        'saved_job_ids': list(saved_job_ids),  # Convert QuerySet to list for template comparison
    }
    
    return render(request, 'candidate/job_search.html', context)

@login_required
@user_passes_test(is_candidate)
def job_detail(request, job_id):
    """View for displaying job details and application form"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')

    job = get_object_or_404(JobPosting, id=job_id, status='active')

    # Check resume
    resume = Resume.objects.filter(candidate=candidate).first()
    resume_uploaded = bool(resume)
    resume_processed = resume.is_processed if resume else False

    # ✅ Always predict match score if resume is processed
    match_score = 0
    if resume_uploaded and resume_processed:
        match_score = predict_specific_job_candidate_match(job, candidate)

    # Application status
    application = JobApplication.objects.filter(job=job, candidate=candidate).first()
    already_applied = application is not None
    is_saved = SavedJob.objects.filter(job=job, candidate=candidate).exists()

    if request.method == 'POST' and not already_applied:
        if not resume_uploaded:
            messages.error(request, "Please upload your resume before applying.")
            return redirect('candidate:upload_resume')

        if not resume_processed:
            messages.warning(request, "Your resume is still being processed.")
            return redirect('candidate:candidate_dashboard')

        form = JobApplicationForm(request.POST)
        if form.is_valid():
            application = form.save(commit=False)
            application.candidate = candidate
            application.job = job
            application.resume = resume
            application.match_score = match_score  # ✅ Ensure it's saved
            application.save()
            messages.success(request, f"You have successfully applied for {job.title}.")
            return redirect('candidate:application_list')
    else:
        form = JobApplicationForm()

    context = {
        'job': job,
        'form': form,
        'already_applied': already_applied,
        'application': application,
        'is_saved': is_saved,
        'resume_uploaded': resume_uploaded,
        'resume_processed': resume_processed,
        'match_score': match_score,
    }

    return render(request, 'candidate/job_detail.html', context)


@login_required
@user_passes_test(is_candidate)
def toggle_save_job(request, job_id):
    """Toggle job saved status"""
    if request.method != 'POST':
        return redirect('candidate:job_recommendations')  # Redirect instead of JSON error
    
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    job = get_object_or_404(JobPosting, id=job_id)
    
    saved_job, created = SavedJob.objects.get_or_create(job=job, candidate=candidate)
    
    if not created:
        # Already saved, so unsave it
        saved_job.delete()
        messages.info(request, "Job removed from your saved list.")
    else:
        messages.success(request, "Job saved successfully.")
    
    return redirect('candidate:job_recommendations')

@login_required
@user_passes_test(is_candidate)
def application_list(request):
    """View for listing all job applications"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    applications = JobApplication.objects.filter(
        candidate=candidate
    ).order_by('-created_at')
    
    context = {
        'applications': applications,
    }
    
    return render(request, 'candidate/application_list.html', context)

@login_required
@user_passes_test(is_candidate)
def saved_jobs(request):
    """View for listing saved jobs"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    saved = SavedJob.objects.filter(candidate=candidate).order_by('-saved_at')
    
    # Check which jobs the candidate has already applied to
    applied_job_ids = JobApplication.objects.filter(
        candidate=candidate
    ).values_list('job_id', flat=True)
    
    context = {
        'saved_jobs': saved,
        'applied_job_ids': list(applied_job_ids),
    }
    
    return render(request, 'candidate/saved_jobs.html', context)

@login_required
@user_passes_test(is_candidate)
def application_detail(request, application_id):
    """View for displaying application details"""
    try:
        candidate = request.user.candidate
    except Candidate.DoesNotExist:
        messages.error(request, "Candidate profile not found")
        return redirect('accounts:profile')
    
    application = get_object_or_404(
        JobApplication, 
        id=application_id, 
        candidate=candidate
    )
    
    context = {
        'application': application,
    }
    
    return render(request, 'candidate/application_detail.html', context)