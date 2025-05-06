import tensorflow as tf
import pickle
import os
import keras
import numpy as np
import pandas as pd

# keras.config.enable_unsafe_deserialization()

# # Register your custom layer with Keras (MUST happen before loading the model)
# @keras.saving.register_keras_serializable()
# class L2Normalize(tf.keras.layers.Layer):
#     def call(self, inputs):
#         return tf.math.l2_normalize(inputs, axis=1)

#     def get_config(self):
#         return super().get_config()
# # Define the save directory
# SAVE_DIR = 'D:/Projects/DJ/job_matching/website/job_matching/ai_models'

# # 1. Verify TensorFlow Models
# try:
#     matching_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'matching_model.keras'))
#     candidate_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'candidate_model.keras'))
#     job_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'job_model.keras'))
#     print("✅ All models loaded successfully!")
# except Exception as e:
#     print("❌ Error loading models:", e)

# # 2. Verify Preprocessors
# try:
#     with open(os.path.join(SAVE_DIR, 'candidate_preprocessor.pkl'), 'rb') as f:
#         candidate_preprocessor_loaded = pickle.load(f)
#     with open(os.path.join(SAVE_DIR, 'job_preprocessor.pkl'), 'rb') as f:
#         job_preprocessor_loaded = pickle.load(f)
#     print("✅ All preprocessors loaded successfully!")
# except Exception as e:
#     print("❌ Error loading preprocessors:", e)

# print("🔎 Model and preprocessor verification complete!")

# Unsafe deserialization only needed if you're 100% sure it's trusted
keras.config.enable_unsafe_deserialization()

# Define the save directory (updated path)
SAVE_DIR = 'D:/Projects/DJ/job_matching/website/job_matching/ai'

# 1. Verify TensorFlow Model
try:
    unified_model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'unified_matching_model.keras'))
    print("✅ Unified matching model loaded successfully!")
except Exception as e:
    print("❌ Error loading unified matching model:", e)

# 2. Verify Preprocessor
try:
    with open(os.path.join(SAVE_DIR, 'unified_preprocessor.pkl'), 'rb') as f:
        unified_preprocessor = pickle.load(f)
    print("✅ Unified preprocessor loaded successfully!")
except Exception as e:
    print("❌ Error loading unified preprocessor:", e)

print("🔎 Unified model and preprocessor verification complete!")

# Define example input data for prediction
# IMPORTANT: Must match training column structure exactly
sample_data = pd.DataFrame([
    {
        "job_title": "Software Engineer",
        "job_description": "Develop scalable web apps with React and Node.js",
        "job_required_skills": "JavaScript, React, Node.js, MongoDB",
        "job_level": "Mid",
        "job_location": "San Francisco",
        "job_type": "Full-time",
        "job_salary_range": "120000",
        "job_education": "Bachelor's",
        "job_industry": "Technology",

        "candidate_skills": "JavaScript, React, Node.js, MongoDB, Git",
        "candidate_experience": "4",
        "candidate_education": "Bachelor's in Computer Science",
        "candidate_location": "San Francisco",
        "candidate_preference": "Onsite",
        "candidate_expected_salary": "125000",
        "availability_in_weeks": "2"
    },
    {
        "job_title": "Data Scientist",
        "job_description": "Analyze data for insights using Python and ML",
        "job_required_skills": "Python, SQL, Statistics, Machine Learning",
        "job_level": "Senior",
        "job_location": "Remote",
        "job_type": "Full-time",
        "job_salary_range": "150000",
        "job_education": "Master's",
        "job_industry": "Technology",

        "candidate_skills": "Python, SQL, R, Machine Learning, Statistics",
        "candidate_experience": "6",
        "candidate_education": "Master's in Data Science",
        "candidate_location": "Chicago",
        "candidate_preference": "Remote",
        "candidate_expected_salary": "145000",
        "availability_in_weeks": "3"
    }
])

# Predict match scores
try:
    processed = unified_preprocessor.transform(sample_data)
    predictions = unified_model.predict(processed)

    print("\n🧪 Prediction Results:")
    for i, score in enumerate(predictions):
        print(f"Example {i + 1}: Match Score = {score[0]:.4f}")
except Exception as e:
    print("❌ Error during prediction:", e)

print("\n✅ Unified model and preprocessor test complete!")