import tensorflow as tf
import pickle
import os
import keras
import numpy as np

keras.config.enable_unsafe_deserialization()

# Define the save directory
SAVE_DIR = 'D:/Projects/DJ/job_matching/website/job_matching/ai_models'

# 1. Verify TensorFlow Models
try:
    matching_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'matching_model.keras'))
    candidate_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'candidate_model.keras'))
    job_model_loaded = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'job_model.keras'))
    print("✅ All models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", e)

# 2. Verify Preprocessors
try:
    with open(os.path.join(SAVE_DIR, 'candidate_preprocessor.pkl'), 'rb') as f:
        candidate_preprocessor_loaded = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'job_preprocessor.pkl'), 'rb') as f:
        job_preprocessor_loaded = pickle.load(f)
    print("✅ All preprocessors loaded successfully!")
except Exception as e:
    print("❌ Error loading preprocessors:", e)

print("🔎 Model and preprocessor verification complete!")