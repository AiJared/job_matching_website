import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from django.conf import settings
from dashboards.models import JobPosting, Resume, JobApplication

# -------------------------------
# ✅ Register custom layer
# -------------------------------
@keras.saving.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()

# -------------------------------
# 🧠 Model & Preprocessor Paths
# -------------------------------
AI_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ai_models')

MATCHING_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'matching_model.keras')
CANDIDATE_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'candidate_model.keras')
JOB_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'job_model.keras')

CANDIDATE_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'candidate_preprocessor.pkl')
JOB_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'job_preprocessor.pkl')

# -------------------------------
# 🔁 Global Load Cache
# -------------------------------
candidate_model = None
job_model = None
matching_model = None
candidate_preprocessor = None
job_preprocessor = None

MATCH_SCORE_THRESHOLD = 70  # Percent

# -------------------------------
# 🔃 Load Everything
# -------------------------------
def load_models_and_preprocessors():
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
        import traceback
        traceback.print_exc()
        raise

# Call once at import
load_models_and_preprocessors()

# -------------------------------
# 🧪 Feature Preparation
# -------------------------------
def prepare_candidate_features(candidate):
    return pd.DataFrame([{
        'skills': candidate.skills or '',
        'education': candidate.education or '',
        'experience': candidate.experience or '',
        'location': candidate.location or '',
    }])

def prepare_job_features(job):
    return pd.DataFrame([{
        'title': job.title or '',
        'required_skills': job.skills_required or '',
        'description': job.description or '',
        'requirements': job.requirements or '',
        'responsibilities': job.responsibilities or '',
        'min_experience': job.experience_required or 0,
        'required_education': job.education_level or '',
        'location': job.location or '',
        'employment_type': job.employment_type or '',
    }])

# -------------------------------
# ⚙️ Embedding Functions
# -------------------------------
def embed_candidate(candidate):
    features_df = prepare_candidate_features(candidate)
    processed = candidate_preprocessor.transform(features_df)
    return candidate_model.predict(processed)

def embed_job(job):
    features_df = prepare_job_features(job)
    processed = job_preprocessor.transform(features_df)
    return job_model.predict(processed)

def generate_job_embedding(job_posting):
    try:
        features_df = prepare_job_features(job_posting)
        processed_features = job_preprocessor.transform(features_df)
        embedding = job_model.predict(processed_features)
        return embedding
    except Exception as e:
        print(f"Error generating job embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

# -------------------------------
# 🔎 Match Score
# -------------------------------
def predict_match_score_raw(candidate_input, job_input):
    """Predict match score using raw preprocessed features."""
    if not matching_model:
        load_models_and_preprocessors()

    with tf.device('/CPU:0'):
        score = matching_model.predict([candidate_input, job_input])[0][0]
    return round(score * 100, 2)


# -------------------------------
# 📡 Match Updating Logic
# -------------------------------
def update_candidate_matches(candidate):
    """Update all job matches for a candidate when profile/resume is updated."""
    try:
        resume = Resume.objects.get(candidate=candidate)
    except Resume.DoesNotExist:
        return

    if not resume:
        return

    try:
        candidate_features_df = prepare_candidate_features(candidate)
        candidate_input = candidate_preprocessor.transform(candidate_features_df)

        active_jobs = JobPosting.objects.filter(status='active')

        for job in active_jobs:
            job_features_df = prepare_job_features(job)
            job_input = job_preprocessor.transform(job_features_df)

            match_score = predict_match_score_raw(candidate_input, job_input)

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
        import traceback
        traceback.print_exc()


def update_job_matches(job_posting):
    """Update all candidate matches when a new job is posted."""
    if not job_posting:
        return

    try:
        job_features_df = prepare_job_features(job_posting)
        job_input = job_preprocessor.transform(job_features_df)

        job_posting.embedding_vector = embed_job(job_posting).tobytes()
        job_posting.save(update_fields=['embedding_vector'])

        resumes = Resume.objects.all()

        for resume in resumes:
            candidate = resume.candidate
            if not resume or not candidate:
                continue

            candidate_features_df = prepare_candidate_features(candidate)
            candidate_input = candidate_preprocessor.transform(candidate_features_df)

            match_score = predict_match_score_raw(candidate_input, job_input)

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
        import traceback
        traceback.print_exc()

