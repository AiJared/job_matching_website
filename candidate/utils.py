# candidate/utils.py
import os
import numpy as np
import tensorflow as tf
import pickle
from django.conf import settings
from dashboards.models import Resume, JobPosting, JobApplication
from dashboards.ai_utils import embed_candidate, update_candidate_matches, prepare_candidate_features, prepare_job_features, candidate_preprocessor, job_preprocessor, predict_match_score_raw
from accounts.models import Candidate

def process_resume(resume_id):
    """Process a resume and generate embedding based on candidate manual fields."""
    import traceback

    try:
        resume = Resume.objects.get(id=resume_id)
        candidate = resume.candidate

        print(f"Processing resume ID: {resume_id}")
        print("Generating embedding from candidate-provided fields...")

        # 🛠 SYNC candidate profile fields into resume
        resume.skills = candidate.skills
        resume.education = candidate.education
        resume.experience = candidate.experience
        resume.save()

        # Generate embedding from candidate fields
        candidate_embedding = embed_candidate(candidate)

        if candidate_embedding is not None:
            print(f"Embedding generated successfully. Shape: {candidate_embedding.shape}")
            resume.embedding_vector = candidate_embedding.tobytes()
            resume.is_processed = True
            resume.save()

            print("Resume marked as processed.")

            # Update matches
            update_candidate_matches(candidate)

            return True
        else:
            print("Failed to generate embedding - returned None")
            return False

    except Exception as e:
        print(f"Error processing resume: {e}")
        print(traceback.format_exc())
        return False

def get_recommended_jobs(candidate, limit=10):
    """Get job recommendations using the raw feature inputs for proper scoring."""
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            return []

        active_jobs = JobPosting.objects.filter(status='active')
        job_matches = []

        # Preprocess candidate input
        candidate_features = prepare_candidate_features(candidate)
        candidate_input = candidate_preprocessor.transform(candidate_features)

        for job in active_jobs:
            job_features = prepare_job_features(job)
            job_input = job_preprocessor.transform(job_features)

            match_score = predict_match_score_raw(candidate_input, job_input)

            if match_score >= 70:
                job_matches.append((job, match_score))

        job_matches.sort(key=lambda x: x[1], reverse=True)
        return job_matches[:limit]

    except Resume.DoesNotExist:
        return []
    except Exception as e:
        print(f"Error getting job recommendations: {e}")
        return []

def get_profile_completion_percentage(candidate):
    """Calculate profile completion percentage"""
    total_tasks = 6
    completed_tasks = 0

    has_resume = Resume.objects.filter(candidate=candidate).exists()
    if has_resume:
        completed_tasks += 1

        resume = Resume.objects.get(candidate=candidate)
        if resume.embedding_vector:
            completed_tasks += 1
        
        if resume.skills:
            completed_tasks += 1
        if resume.education:
            completed_tasks += 1
        if resume.experience:
            completed_tasks += 1

    user = candidate.user
    if user.phone and user.country and user.city:
        completed_tasks += 1

    completion_percentage = (completed_tasks / total_tasks) * 100
    return round(completion_percentage)

def update_candidate_matches(candidate):
    """Update match scores for all active jobs for this candidate"""
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            return

        active_jobs = JobPosting.objects.filter(status='active')

        # Get preprocessed candidate input
        candidate_features = prepare_candidate_features(candidate)
        candidate_input = candidate_preprocessor.transform(candidate_features)

        for job in active_jobs:
            job_features = prepare_job_features(job)
            job_input = job_preprocessor.transform(job_features)

            match_score = predict_match_score_raw(candidate_input, job_input)

            if match_score >= 70:
                application, created = JobApplication.objects.get_or_create(
                    job=job,
                    candidate=candidate,
                    defaults={
                        'resume': resume,
                        'match_score': match_score,
                        'status': 'pending'
                    }
                )

                if not created:
                    application.match_score = match_score
                    application.save(update_fields=['match_score', 'updated_at'])

    except Resume.DoesNotExist:
        pass
    except Exception as e:
        print(f"Error updating candidate matches: {e}")
