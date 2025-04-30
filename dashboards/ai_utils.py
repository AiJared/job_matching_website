# dashboards/ai_utils.py
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from dashboards.models import Resume

import keras

# 🔒 Register custom layer
@keras.saving.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    def get_config(self):
        return super().get_config()

# ================================
# 🔁 Model and Preprocessor Loader
# ================================
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ai_models')
candidate_model = None
job_model = None
matching_model = None
candidate_preprocessor = None
job_preprocessor = None

def load_models():
    global candidate_model, job_model, matching_model
    global candidate_preprocessor, job_preprocessor

    try:
        keras.config.enable_unsafe_deserialization()

        candidate_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'candidate_model.keras'))
        job_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'job_model.keras'))
        matching_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'matching_model.keras'))

        with open(os.path.join(MODEL_DIR, 'candidate_preprocessor.pkl'), 'rb') as f:
            candidate_preprocessor = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'job_preprocessor.pkl'), 'rb') as f:
            job_preprocessor = pickle.load(f)

        print("✅ AI models loaded successfully.")
    except Exception as e:
        print(f"❌ Model load failure: {e}")

load_models()

# =====================
# 🧪 Preprocessing Utils
# =====================
def prepare_candidate_features(candidate):
    return pd.DataFrame([{
        'id': candidate.id,
        'skills': candidate.skills or '',
        'experience': float(candidate.experience or 0),
        'education': candidate.education or '',
        'location': candidate.location or ''
    }])

def prepare_job_features(job):
    return pd.DataFrame([{
        'id': job.id,
        'title': job.title or '',
        'required_skills': job.skills_required or '',
        'min_experience': float(job.experience_required or 0),
        'required_education': job.education_level or '',
        'location': job.location or ''
    }])

# ================================
# 🧠 Matching Score Calculations
# ================================
def sigmoid_transform(x, center=0.5, steepness=5):
    return 1 / (1 + np.exp(-steepness * (x - center)))

def predict_match_score_raw(candidate_input, job_input):
    """
    Use only the matching model and apply sigmoid contrast enhancement.
    """
    try:
        # Convert sparse to dense
        if hasattr(candidate_input, "toarray"):
            candidate_input = candidate_input.toarray()
        if hasattr(job_input, "toarray"):
            job_input = job_input.toarray()

        # Ensure 2D shape
        if len(candidate_input.shape) == 1:
            candidate_input = candidate_input.reshape(1, -1)
        if len(job_input.shape) == 1:
            job_input = job_input.reshape(1, -1)

        # 🧠 Get prediction from matching model
        model_score = float(matching_model.predict([candidate_input, job_input])[0][0])

        # 🎯 Apply contrast enhancement
        adjusted_score = sigmoid_transform(model_score, center=0.5, steepness=5)

        # 🔢 Scale to percentage
        return adjusted_score * 100

    except Exception as e:
        print(f"❌ Error predicting match score: {e}")
        return 50.0


def predict_specific_job_candidate_match(job, candidate):
    """
    Rule-based match score between a job and candidate (no AI model used).
    """
    try:
        print("🎯 DEBUG MATCH TEST")
        print(f"Candidate skills: {candidate.skills}")
        print(f"Job required skills: {job.skills_required}")
        print(f"Candidate experience: {candidate.experience}")
        print(f"Job experience required: {job.experience_required}")
        print(f"Candidate education: {candidate.education}")
        print(f"Job education required: {job.education_level}")

        # 🚫 Skip model prediction
        base_score = 0.0  # Start from zero
        score = apply_rule_based_adjustments(base_score, job, candidate)

        return score
    except Exception as e:
        print(f"❌ Error in specific job-candidate match prediction: {e}")
        return 0.0


def apply_rule_based_adjustments(score, job, candidate):
    try:
        adj = 0

        # 🔍 Skills Match (Up to 50)
        job_skills = {s.strip().lower() for s in job.skills_required.replace('\n', ',').split(',') if s.strip()}
        cand_skills = {s.strip().lower() for s in candidate.skills.replace('\n', ',').split(',') if s.strip()}
        overlap = job_skills & cand_skills

        if job_skills:
            ratio = len(overlap) / len(job_skills)
            if ratio >= 0.7:
                adj += 50
            elif ratio >= 0.5:
                adj += 35
            elif ratio >= 0.3:
                adj += 20
            elif ratio < 0.1 and len(job_skills) > 3:
                adj -= 10

        # 🔧 Experience Match (Up to 25)
        try:
            cand_exp = float(candidate.experience)
            job_exp = float(job.experience_required)
            if cand_exp >= job_exp:
                adj += 25
            elif cand_exp >= job_exp * 0.5:
                adj += 10
            else:
                adj -= 10
        except (TypeError, ValueError):
            pass

        # 🎓 Education Match (Up to 25)
        levels = {
            'high school': 1, 'diploma': 2, 'associate': 3,
            'bachelor': 4, 'master': 5, 'phd': 6, 'doctorate': 6
        }

        c_lvl = 0
        j_lvl = 0

        for lvl, val in levels.items():
            if candidate.education and lvl in candidate.education.lower():
                c_lvl = max(c_lvl, val)
            if job.education_level and lvl in job.education_level.lower():
                j_lvl = max(j_lvl, val)

        if c_lvl and j_lvl:
            if c_lvl >= j_lvl:
                adj += 25
            elif c_lvl == j_lvl - 1:
                adj += 10
            elif c_lvl < j_lvl - 1:
                adj -= 10

        final_score = max(min(adj, 99), 10)
        print(f"🔍 Score is {final_score:.1f}")
        return final_score

    except Exception as e:
        print(f"❌ Error applying rule-based adjustments: {e}")
        return 25.0


# ============================
# 📦 Embedding (for job saving)
# ============================
def generate_job_embedding(job_posting):
    try:
        job_df = prepare_job_features(job_posting)
        job_input = job_preprocessor.transform(job_df)
        job_emb = job_model.predict(job_input)
        return job_emb
    except Exception as e:
        print(f"❌ Error generating job embedding: {e}")
        return None

def update_job_matches(job_posting):
    """
    Evaluate matches for a given job. Does NOT create applications.
    """
    try:
        for resume in Resume.objects.filter(is_processed=True):
            candidate = resume.candidate
            score = predict_specific_job_candidate_match(job_posting, candidate)

            if score >= 50:
                print(f"📌 Job {job_posting.id} matches Candidate {candidate.id} with score {score:.2f}%")

    except Exception as e:
        print(f"❌ Error in job match evaluation: {e}")
