# candidate/utils.py
import os
import numpy as np
import tensorflow as tf
import pickle
from django.conf import settings
from dashboards.models import Resume, JobPosting, JobApplication
from dashboards.ai_utils import load_tf_model, load_preprocessor, calculate_match_score
from accounts.models import Candidate

def process_resume(resume_id):
    """Process a resume, generate embedding based ONLY on form fields (skills, education, experience)"""
    from dashboards.ai_utils import generate_resume_embedding
    import traceback

    try:
        resume = Resume.objects.get(id=resume_id)
        candidate = resume.candidate

        print(f"Processing resume ID: {resume_id}")
        print("Generating embedding from candidate-provided fields...")

        # 🛠 SYNC candidate profile fields into resume before generating embedding
        resume.skills = candidate.skills
        resume.education = candidate.education
        resume.experience = candidate.experience
        resume.save()

        # We DO NOT care about extracted_text anymore

        # Generate embedding based on the fields
        embedding = generate_resume_embedding(resume)

        if embedding is not None:
            print(f"Embedding generated successfully. Shape: {embedding.shape}")
            resume.embedding_vector = embedding.tobytes()
            resume.is_processed = True
            resume.save()

            print("Resume marked as processed.")

            # Update job matches
            update_candidate_matches(resume.candidate)

            return True
        else:
            print("Failed to generate embedding - returned None")
            return False

    except Exception as e:
        print(f"Error processing resume: {e}")
        print(traceback.format_exc())
        return False


def get_recommended_jobs(candidate, limit=10):
    """Get job recommendations for a candidate based on resume match score"""
    try:
        # Get candidate's resume
        resume = Resume.objects.get(candidate=candidate)
        
        if not resume.embedding_vector:
            return []
        
        # Find active job postings
        active_jobs = JobPosting.objects.filter(status='active')
        
        # Calculate match scores for each job
        job_matches = []
        
        for job in active_jobs:
            if job.embedding_vector:
                match_score = calculate_match_score(job.embedding_vector, resume.embedding_vector)
                job_matches.append((job, match_score))
        
        # Sort by match score (highest first)
        job_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        return job_matches[:limit]
    
    except Resume.DoesNotExist:
        return []
    except Exception as e:
        print(f"Error getting job recommendations: {e}")
        return []

def get_profile_completion_percentage(candidate):
    """Calculate profile completion percentage"""
    total_tasks = 6  # Total number of profile completion tasks
    completed_tasks = 0
    
    # Check if resume exists
    has_resume = Resume.objects.filter(candidate=candidate).exists()
    if has_resume:
        completed_tasks += 1
        
        # Check if resume has been processed with embedding
        resume = Resume.objects.get(candidate=candidate)
        if resume.embedding_vector:
            completed_tasks += 1
            
        # Check for skills, education, experience
        if resume.skills:
            completed_tasks += 1
        if resume.education:
            completed_tasks += 1
        if resume.experience:
            completed_tasks += 1
    
    # Check if user profile is complete (assuming you check for profile picture and contact info)
    user = candidate.user
    if user.phone and user.country and user.city:
        completed_tasks += 1
    
    # Calculate percentage
    completion_percentage = (completed_tasks / total_tasks) * 100
    return round(completion_percentage)

def update_candidate_matches(candidate):
    """Update match scores for all active jobs for this candidate"""
    try:
        resume = Resume.objects.get(candidate=candidate)
        
        if not resume.embedding_vector:
            return
            
        active_jobs = JobPosting.objects.filter(status='active')
        
        for job in active_jobs:
            if job.embedding_vector:
                match_score = calculate_match_score(job.embedding_vector, resume.embedding_vector)
                
                # Check if application already exists
                application, created = JobApplication.objects.get_or_create(
                    job=job,
                    candidate=candidate,
                    defaults={
                        'resume': resume,
                        'match_score': match_score,
                        'status': 'pending'
                    }
                )
                
                # Update match score if application already exists but don't change the status
                if not created:
                    application.match_score = match_score
                    application.save(update_fields=['match_score', 'updated_at'])
    
    except Resume.DoesNotExist:
        pass
    except Exception as e:
        print(f"Error updating candidate matches: {e}")