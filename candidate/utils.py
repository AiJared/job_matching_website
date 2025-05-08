# candidate/utils.py
import numpy as np
import pandas as pd
from dashboards.models import Resume, JobPosting, JobApplication
from dashboards.ai_utils import (
    predict_specific_job_candidate_match  #  only needed one now
)
from accounts.models import Candidate

def process_resume(resume_id):
    """
    Process a resume and mark it as processed (no embedding used in unified model).
    """
    try:
        resume = Resume.objects.get(id=resume_id)
        candidate = resume.candidate
        print(f"Processing resume ID: {resume_id}")

        # Sync resume fields with candidate profile
        resume.skills = candidate.skills
        resume.education = candidate.education
        resume.experience = candidate.experience
        resume.is_processed = True
        resume.save()

        print("Resume marked as processed.")
        
        # Update candidate matches
        update_candidate_matches(candidate)
        return True
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        return False


def get_recommended_jobs(candidate, limit=10):
    """
    Get recommended jobs for a candidate using unified AI model.
    """
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.is_processed:
            print("Resume not processed yet")
            return []

        job_matches = []
        active_jobs = JobPosting.objects.filter(status='active')
        print(f"Found {active_jobs.count()} active jobs to match against")

        for job in active_jobs:
            score = predict_specific_job_candidate_match(job, candidate)
            if score >= 65:
                job_matches.append((job, score))

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
        if resume.is_processed: completed += 1
        if resume.skills: completed += 1
        if resume.education: completed += 1
        if resume.experience: completed += 1

    if candidate.user.phone and candidate.user.country and candidate.user.city:
        completed += 1

    return round((completed / total) * 100)


def update_candidate_matches(candidate):
    """
    Scores candidate-to-job matches using the unified AI model.
    """
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.is_processed:
            print("Resume not processed yet")
            return

        for job in JobPosting.objects.filter(status='active'):
            score = predict_specific_job_candidate_match(job, candidate)
            if score >= 50:
                print(f"🔎 Match found → Candidate {candidate.id} | Job {job.id} | Score: {score:.2f}%")
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

        try:
            application = JobApplication.objects.get(
                job=job, 
                candidate=candidate
            )
            return application.match_score
        except JobApplication.DoesNotExist:
            return predict_specific_job_candidate_match(job, candidate)
    except JobPosting.DoesNotExist:
        print(f"Job with ID {job_id} not found")
        return 0
    except Exception as e:
        print(f"❌ Error getting match score: {e}")
        return 0
