# candidate/utils.py
import numpy as np
import pandas as pd
from dashboards.models import Resume, JobPosting, JobApplication
from dashboards.ai_utils import (
    matching_model, candidate_preprocessor, job_preprocessor,
    prepare_candidate_features, prepare_job_features, predict_match_score_raw
)
from accounts.models import Candidate

def process_resume(resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        candidate = resume.candidate
        print(f"Processing resume ID: {resume_id}")

        resume.skills = candidate.skills
        resume.education = candidate.education
        resume.experience = candidate.experience
        resume.save()

        features_df = prepare_candidate_features(candidate)
        processed = candidate_preprocessor.transform(features_df)
        resume.embedding_vector = processed.toarray().astype(np.float32).tobytes()

        resume.is_processed = True
        resume.save()

        print("Resume marked as processed.")
        update_candidate_matches(candidate)
        return True
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        return False

def get_recommended_jobs(candidate, limit=10):
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            return []

        candidate_input = candidate_preprocessor.transform(prepare_candidate_features(candidate))
        job_matches = []

        for job in JobPosting.objects.filter(status='active'):
            job_input = job_preprocessor.transform(prepare_job_features(job))
            score = predict_match_score_raw(candidate_input, job_input)
            print(f"🔍 Job {job.id} Match Score: {score}")

            if score >= 70:
                job_matches.append((job, score))

        job_matches.sort(key=lambda x: x[1], reverse=True)
        return job_matches[:limit]
    except Exception as e:
        print(f"❌ Error getting job recommendations: {e}")
        return []

def get_profile_completion_percentage(candidate):
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
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            return

        candidate_input = candidate_preprocessor.transform(prepare_candidate_features(candidate))
        for job in JobPosting.objects.filter(status='active'):
            job_input = job_preprocessor.transform(prepare_job_features(job))
            score = predict_match_score_raw(candidate_input, job_input)

            if score >= 70:
                app, created = JobApplication.objects.get_or_create(
                    job=job, candidate=candidate,
                    defaults={'resume': resume, 'match_score': score, 'status': 'pending'}
                )
                if not created:
                    app.match_score = score
                    app.save(update_fields=['match_score', 'updated_at'])
    except Exception as e:
        print(f"❌ Error updating candidate matches: {e}")
