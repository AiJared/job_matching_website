# candidate/utils.py
import numpy as np
import pandas as pd
from dashboards.models import Resume, JobPosting, JobApplication
from dashboards.ai_utils import (
    matching_model, candidate_preprocessor, job_preprocessor,
    prepare_candidate_features, prepare_job_features, 
    predict_match_score_raw, predict_specific_job_candidate_match
)
from accounts.models import Candidate

def process_resume(resume_id):
    """
    Process a resume and create an embedding vector
    """
    try:
        resume = Resume.objects.get(id=resume_id)
        candidate = resume.candidate
        print(f"Processing resume ID: {resume_id}")

        # Make sure resume has candidate skills, education, and experience
        resume.skills = candidate.skills
        resume.education = candidate.education
        resume.experience = candidate.experience
        resume.save()

        # Create embedding vector
        features_df = prepare_candidate_features(candidate)
        processed = candidate_preprocessor.transform(features_df)
        
        # Convert to float32 before storing
        if hasattr(processed, "toarray"):
            resume.embedding_vector = processed.toarray().astype(np.float32).tobytes()
        else:
            resume.embedding_vector = processed.astype(np.float32).tobytes()

        resume.is_processed = True
        resume.save()

        print("Resume marked as processed.")
        
        # Update candidate matches after processing
        update_candidate_matches(candidate)
        return True
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        return False


def get_recommended_jobs(candidate, limit=10):
    """
    Get recommended jobs for a candidate
    """
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector or not resume.is_processed:
            print("Resume not processed yet")
            return []

        # Get candidate input
        candidate_input = candidate_preprocessor.transform(prepare_candidate_features(candidate))
        job_matches = []

        # Get all active jobs
        active_jobs = JobPosting.objects.filter(status='active')
        print(f"Found {active_jobs.count()} active jobs to match against")

        # Match against each job
        for job in active_jobs:
            # Use the improved specific matching function
            score = predict_specific_job_candidate_match(job, candidate)
            
            # Add to matches if score is above threshold
            if score >= 40:  # Lower threshold to show more potential matches
                job_matches.append((job, score))

        # Sort by score (highest first)
        job_matches.sort(key=lambda x: x[1], reverse=True)
        return job_matches[:limit]
    except Resume.DoesNotExist:
        print("No resume found for candidate")
        return []
    except Exception as e:
        print(f"❌ Error getting job recommendations: {e}")
        return []


def get_profile_completion_percentage(candidate):
    """
    Calculate profile completion percentage
    """
    total = 6
    completed = 0

    if Resume.objects.filter(candidate=candidate).exists():
        completed += 1
        resume = Resume.objects.get(candidate=candidate)
        if resume.embedding_vector:
            completed += 1
        if resume.skills: completed += 1
        if resume.education: completed += 1
        if resume.experience: completed += 1

    if candidate.user.phone and candidate.user.country and candidate.user.city:
        completed += 1

    return round((completed / total) * 100)


def update_candidate_matches(candidate):
    """
    Just scores candidate-to-job matches. No JobApplication creation.
    """
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector or not resume.is_processed:
            print("Resume not processed yet")
            return

        for job in JobPosting.objects.filter(status='active'):
            score = predict_specific_job_candidate_match(job, candidate)

            # Just log or cache matches, don't apply.
            if score >= 50:
                print(f"🔎 Match found → Candidate {candidate.id} | Job {job.id} | Score: {score:.2f}%")
                # Optionally cache to Redis, DB field, or display only (not JobApplication)

    except Resume.DoesNotExist:
        print("No resume found for candidate")
    except Exception as e:
        print(f"❌ Error updating candidate matches: {e}")


def get_match_score_for_job(candidate, job_id):
    """
    Get match score for a specific job
    """
    try:
        job = JobPosting.objects.get(id=job_id)
        
        # Check if we have a stored match
        try:
            application = JobApplication.objects.get(
                job=job, 
                candidate=candidate
            )
            return application.match_score
        except JobApplication.DoesNotExist:
            # Calculate match score on-the-fly
            score = predict_specific_job_candidate_match(job, candidate)
            return score
            
    except JobPosting.DoesNotExist:
        print(f"Job with ID {job_id} not found")
        return 0
    except Exception as e:
        print(f"❌ Error getting match score: {e}")
        return 0