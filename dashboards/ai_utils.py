# dashboards/ai_utils.py
import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from django.conf import settings
from dashboards.models import JobPosting, Resume, JobApplication

@keras.saving.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()

# Model paths
AI_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ai_models')

MATCHING_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'matching_model.keras')
CANDIDATE_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'candidate_model.keras')
JOB_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'job_model.keras')

CANDIDATE_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'candidate_preprocessor.pkl')
JOB_PREPROCESSOR_PATH = os.path.join(AI_MODELS_DIR, 'job_preprocessor.pkl')

# Globals
candidate_model = None
job_model = None
matching_model = None
candidate_preprocessor = None
job_preprocessor = None

MATCH_SCORE_THRESHOLD = 50

def load_models_and_preprocessors():
    global candidate_model, job_model, matching_model
    global candidate_preprocessor, job_preprocessor

    try:
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

load_models_and_preprocessors()

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

def embed_candidate(candidate):
    df = prepare_candidate_features(candidate)
    x = candidate_preprocessor.transform(df)
    return candidate_model.predict(x)

def embed_job(job):
    df = prepare_job_features(job)
    x = job_preprocessor.transform(df)
    return job_model.predict(x)

def generate_job_embedding(job):
    try:
        return embed_job(job)
    except Exception as e:
        print(f"❌ Error generating job embedding: {e}")
        return None

def predict_match_score_raw(candidate_input, job_input):
    score = matching_model.predict([candidate_input, job_input])[0][0]
    return round(score * 100, 2)

def update_candidate_matches(candidate):
    try:
        resume = Resume.objects.get(candidate=candidate)
        if not resume.embedding_vector:
            return

        candidate_input = candidate_preprocessor.transform(prepare_candidate_features(candidate))
        active_jobs = JobPosting.objects.filter(status='active')

        for job in active_jobs:
            job_input = job_preprocessor.transform(prepare_job_features(job))
            score = predict_match_score_raw(candidate_input, job_input)

            if score >= MATCH_SCORE_THRESHOLD:
                app, created = JobApplication.objects.get_or_create(
                    job=job, candidate=candidate,
                    defaults={'resume': resume, 'match_score': score, 'status': 'pending'}
                )
                if not created:
                    app.match_score = score
                    app.save(update_fields=['match_score', 'updated_at'])
    except Exception as e:
        print(f"❌ Error updating candidate matches: {e}")

def update_job_matches(job_posting):
    try:
        job_input = job_preprocessor.transform(prepare_job_features(job_posting))
        job_posting.embedding_vector = embed_job(job_posting).tobytes()
        job_posting.save(update_fields=['embedding_vector'])

        for resume in Resume.objects.all():
            if not resume.embedding_vector:
                continue
            candidate = resume.candidate
            candidate_input = candidate_preprocessor.transform(prepare_candidate_features(candidate))
            score = predict_match_score_raw(candidate_input, job_input)

            if score >= MATCH_SCORE_THRESHOLD:
                app, created = JobApplication.objects.get_or_create(
                    job=job_posting, candidate=candidate,
                    defaults={'resume': resume, 'match_score': score, 'status': 'pending'}
                )
                if not created:
                    app.match_score = score
                    app.save(update_fields=['match_score', 'updated_at'])
    except Exception as e:
        print(f"❌ Error updating job matches: {e}")

def calculate_match_score(job_vector_bytes, candidate_vector_bytes):
    """Decode vectors and get score from matching model"""
    try:
        job_vector = np.frombuffer(job_vector_bytes, dtype=np.float32).reshape(1, -1)
        candidate_vector = np.frombuffer(candidate_vector_bytes, dtype=np.float32).reshape(1, -1)

        # 🔥 Use model to predict match
        score = matching_model.predict([candidate_vector, job_vector])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        print(f"❌ Error in calculate_match_score: {e}")
        return 0.0
