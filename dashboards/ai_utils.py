import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from django.conf import settings
from dashboards.models import JobPosting, Resume, JobApplication

# Set model paths
AI_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ai_models')

MATCHING_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'matching_model.keras')
CANDIDATE_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'candidate_model.keras')
JOB_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'job_model.keras')

CANDIDATE_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'candidate_preprocessor.pkl')
JOB_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'job_preprocessor.pkl')

# Global caches
candidate_model = None
job_model = None
matching_model = None
candidate_preprocessor = None
job_preprocessor = None

# Threshold
MATCH_SCORE_THRESHOLD = 70  # Minimum 70%

def load_models_and_preprocessors():
    """Load AI models and preprocessors into memory."""
    global candidate_model, job_model, matching_model
    global candidate_preprocessor, job_preprocessor

    try:
        import keras
        keras.config.enable_unsafe_deserialization()

        candidate_model = tf.keras.models.load_model(CANDIDATE_MODEL_PATH)
        job_model = tf.keras.models.load_model(JOB_MODEL_PATH)
        matching_model = tf.keras.models.load_model(MATCHING_MODEL_PATH)

        with open(CANDIDATE_PREPROCESSOR_PATH, 'rb') as f:
            candidate_preprocessor = pickle.load(f)

        with open(JOB_PREPROCESSOR_PATH, 'rb') as f:
            job_preprocessor = pickle.load(f)

        print("✅ AI models and preprocessors loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models or preprocessors: {e}")

# Load immediately
load_models_and_preprocessors()

def prepare_candidate_features(candidate):
    """Prepare candidate features from their manual profile."""
    data = {
        'skills': candidate.skills or '',
        'education': candidate.education or '',
        'experience': candidate.experience or '',
    }
    return pd.DataFrame([data])

def prepare_job_features(job):
    """Prepare job features from job posting."""
    data = {
        'title': job.title or '',
        'required_skills': job.skills_required or '',
        'description': job.description or '',
        'requirements': job.requirements or '',
        'responsibilities': job.responsibilities or '',
        'min_experience': job.experience_required or 0,
        'education_level': job.education_level or '',
        'location': job.location or '',
        'employment_type': job.employment_type or '',
    }
    return pd.DataFrame([data])

def embed_candidate(candidate):
    """Generate candidate embedding."""
    if not candidate_model or not candidate_preprocessor:
        raise Exception("Models or preprocessors not loaded.")

    features_df = prepare_candidate_features(candidate)
    processed = candidate_preprocessor.transform(features_df)
    embedding = candidate_model.predict(processed)
    return embedding

def embed_job(job):
    """Generate job embedding."""
    if not job_model or not job_preprocessor:
        raise Exception("Models or preprocessors not loaded.")

    features_df = prepare_job_features(job)
    processed = job_preprocessor.transform(features_df)
    embedding = job_model.predict(processed)
    return embedding

def predict_match_score(candidate_embedding, job_embedding):
    """Predict match score using the matching model."""
    if not matching_model:
        raise Exception("Matching model not loaded.")

    # Matching model expects normalized embeddings
    score = matching_model.predict([candidate_embedding, job_embedding])[0][0]
    return round(score * 100, 2)  # Scale to 0-100%

def update_candidate_matches(candidate):
    """Update all job matches for a candidate when profile/resume is updated."""
    try:
        resume = Resume.objects.get(candidate=candidate)
    except Resume.DoesNotExist:
        return

    if not resume:
        return

    try:
        candidate_emb = embed_candidate(candidate)
        active_jobs = JobPosting.objects.filter(status='active')

        for job in active_jobs:
            if not job.embedding_vector:
                continue

            job_emb = np.frombuffer(job.embedding_vector, dtype=np.float32).reshape(1, -1)
            match_score = predict_match_score(candidate_emb, job_emb)

            if match_score >= MATCH_SCORE_THRESHOLD:
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
    except Exception as e:
        print(f"Error updating candidate matches: {e}")

def update_job_matches(job_posting):
    """Update all candidate matches when a new job is posted."""
    if not job_posting:
        return

    try:
        job_emb = embed_job(job_posting)
        job_posting.embedding_vector = job_emb.tobytes()
        job_posting.save(update_fields=['embedding_vector'])

        resumes = Resume.objects.all()

        for resume in resumes:
            candidate = resume.candidate
            if not resume or not candidate:
                continue

            candidate_emb = embed_candidate(candidate)

            match_score = predict_match_score(candidate_emb, job_emb)

            if match_score >= MATCH_SCORE_THRESHOLD:
                application, created = JobApplication.objects.get_or_create(
                    job=job_posting,
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
    except Exception as e:
        print(f"Error updating job matches: {e}")

def generate_job_embedding(job_posting):
    """Generate and return an embedding vector for a JobPosting instance."""
    if not job_model or not job_preprocessor:
        raise Exception("Job model or preprocessor not loaded.")

    try:
        features_df = prepare_job_features(job_posting)
        processed_features = job_preprocessor.transform(features_df)
        embedding = job_model.predict(processed_features)
        return embedding
    except Exception as e:
        print(f"Error generating job embedding: {e}")
        return None
