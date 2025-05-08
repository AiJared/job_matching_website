# dashboards/ai_utils.py
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from django.conf import settings
from dashboards.models import Resume
from accounts.models import Candidate

# Unsafe deserialization for trusted preprocessor
import keras
keras.config.enable_unsafe_deserialization()

# ======================
# Unified Model Setup
# ======================
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ai')
unified_model = None
unified_preprocessor = None

def load_unified_model():
    global unified_model, unified_preprocessor
    try:
        unified_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'unified_matching_model.keras'))
        with open(os.path.join(MODEL_DIR, 'unified_preprocessor.pkl'), 'rb') as f:
            unified_preprocessor = pickle.load(f)
        print("✅ Unified AI model + preprocessor loaded.")
    except Exception as e:
        print(f"❌ Error loading AI model or preprocessor: {e}")

load_unified_model()

# ===================================
# Data Preparation for Prediction
# ===================================
def build_match_input_dataframe(job, candidate):
    """
    Convert job and candidate models into a single-row dataframe for prediction.
    """
    return pd.DataFrame([{
        "job_title": job.title,
        "job_description": job.description,
        "job_required_skills": job.skills_required,
        "job_level": job.employment_type,  # assuming job level ≈ employment_type
        "job_location": job.location,
        "job_type": job.employment_type,
        "job_salary_range": str(job.salary_max or 0),  # string format to match training input
        "job_education": job.education_level,
        "job_industry": job.category,  # hardcoded or map from category if you have that

        "candidate_skills": candidate.skills or "",
        "candidate_experience": str(candidate.experience or 0),
        "candidate_education": candidate.education or "",
        "candidate_location": candidate.location or "",
        "candidate_preference": candidate.job_type_preference or "",  # placeholder, could be inferred from candidate form
        "candidate_expected_salary": candidate.expected_salary or "",  # not captured currently
        "availability_in_weeks": candidate.availability_in_weeks or "",      # default fallback
    }])

# =====================================
# Match Score Prediction (Unified AI)
# =====================================
def predict_specific_job_candidate_match(job, candidate):
    """
    Predict a match score using the unified AI model.
    Prevents predictions when no skill overlap exists.
    """
    try:
        if not unified_model or not unified_preprocessor:
            raise ValueError("Model or preprocessor not loaded.")

        # Basic overlap filter to prevent garbage matches
        job_skills = {s.strip().lower() for s in job.skills_required.replace('\n', ',').split(',') if s.strip()}
        cand_skills = {s.strip().lower() for s in candidate.skills.replace('\n', ',').split(',') if s.strip()}
        overlap = job_skills & cand_skills

        if not overlap:
            print("🚫 No skill overlap — skipping AI prediction.")
            return 10.0  # Minimum fallback score if there's no match

        input_df = build_match_input_dataframe(job, candidate)
        print("🧾 Sample prediction row:")
        print(input_df.to_dict(orient='records')[0])

        processed = unified_preprocessor.transform(input_df)
        prediction = unified_model.predict(processed)
        score = float(prediction[0][0]) * 100  # scale to percent
        return round(score, 2)

    except Exception as e:
        print(f"❌ Error in AI match prediction: {e}")
        return 50.0

# =================================================
# Optional: Placeholder for Legacy Compatibility
# =================================================
def prepare_candidate_features(candidate):
    """
    Retained for backward compatibility (not used by unified model).
    """
    return {
        'skills': candidate.skills,
        'experience': float(candidate.experience or 0),
        'education': candidate.education,
        'location': candidate.location
    }

def prepare_job_features(job):
    """
    Retained for backward compatibility (not used by unified model).
    """
    return {
        'title': job.title,
        'required_skills': job.skills_required,
        'experience_required': float(job.experience_required or 0),
        'education_level': job.education_level,
        'location': job.location
    }

# ========================
# Job Match Update Loop
# ========================
def update_job_matches(job_posting):
    """
    Recalculate match scores for all processed resumes against a new/updated job.
    """
    try:
        for resume in Resume.objects.filter(is_processed=True):
            candidate = resume.candidate
            score = predict_specific_job_candidate_match(job_posting, candidate)

            if score >= 50:
                print(f"📌 Job {job_posting.id} matches Candidate {candidate.id} with score {score:.2f}%")
    except Exception as e:
        print(f"❌ Error in job match evaluation: {e}")
