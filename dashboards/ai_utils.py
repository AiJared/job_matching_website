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
    try:
        if hasattr(candidate_input, "toarray"):
            candidate_input = candidate_input.toarray()
        if hasattr(job_input, "toarray"):
            job_input = job_input.toarray()

        if len(candidate_input.shape) == 1:
            candidate_input = candidate_input.reshape(1, -1)
        if len(job_input.shape) == 1:
            job_input = job_input.reshape(1, -1)

        model_score = float(matching_model.predict([candidate_input, job_input])[0][0])
        candidate_emb = candidate_model.predict(candidate_input)
        job_emb = job_model.predict(job_input)

        sim_score = cosine_similarity(candidate_emb, job_emb)[0][0]
        combined = (model_score * 0.4) + (sim_score * 0.6)
        adjusted = sigmoid_transform(combined, center=0.5, steepness=5)

        return adjusted * 100
    except Exception as e:
        print(f"❌ Error predicting match score: {e}")
        return 50.0

def predict_specific_job_candidate_match(job, candidate):
    """
    Predict match score between a specific job and candidate
    """
    try:
        # 🔍 DEBUG: See actual input content
        print("🎯 DEBUG MATCH TEST")
        print(f"Candidate skills: {candidate.skills}")
        print(f"Job required skills: {job.skills_required}")
        print(f"Candidate experience: {candidate.experience}")
        print(f"Job experience required: {job.experience_required}")
        print(f"Candidate education: {candidate.education}")
        print(f"Job education required: {job.education_level}")

        # Prepare features
        candidate_features = prepare_candidate_features(candidate)
        job_features = prepare_job_features(job)

        # Transform features
        candidate_input = candidate_preprocessor.transform(candidate_features)
        job_input = job_preprocessor.transform(job_features)

        # Predict match score
        score = predict_match_score_raw(candidate_input, job_input)

        # Apply rule-based adjustments
        score = apply_rule_based_adjustments(score, job, candidate)

        return score
    except Exception as e:
        print(f"❌ Error in specific job-candidate match prediction: {e}")
        return 50.0


def apply_rule_based_adjustments(score, job, candidate):
    """
    Adjust match score using skills, experience, and education overlap.
    Stronger weighting to overcome weak base scores.
    """
    try:
        adj = 0

        # 🧠 Skill overlap
        job_skills = {s.strip().lower() for s in job.skills_required.replace('\n', ',').split(',') if s.strip()}
        cand_skills = {s.strip().lower() for s in candidate.skills.replace('\n', ',').split(',') if s.strip()}
        overlap = job_skills & cand_skills

        if job_skills:
            ratio = len(overlap) / len(job_skills)
            if ratio > 0.7: adj += 35  # was 15
            elif ratio > 0.5: adj += 25  # was 10
            elif ratio > 0.3: adj += 15  # was 5
            elif ratio < 0.1 and len(job_skills) > 3: adj -= 10

        # 🔧 Experience adjustment
        try:
            cand_exp = float(candidate.experience)
            job_exp = float(job.experience_required)
            if cand_exp >= job_exp:
                adj += 15  # was 5
            elif cand_exp < job_exp / 2:
                adj -= 10
        except (TypeError, ValueError):
            pass

        # 🎓 Education level adjustment
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
                adj += 10  # was 5
            elif c_lvl < j_lvl - 1:
                adj -= 10

        # 🧮 Final adjustment
        final_score = max(min(score + adj, 99), 10)
        print(f"🔍 Score adjusted from {score:.1f} to {final_score:.1f}")
        return final_score

    except Exception as e:
        print(f"❌ Error applying rule-based adjustments: {e}")
        return score


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
