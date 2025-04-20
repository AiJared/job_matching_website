# dashboards/ai_utils.py
import numpy as np
import pickle
import os
import tensorflow as tf
from django.conf import settings
from .models import JobPosting, Resume, JobApplication

# Paths to saved models and preprocessors
JOB_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_model.h5')
CANDIDATE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_model.h5')
MATCHING_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'matching_model.h5')
JOB_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_preprocessor.pkl')
CANDIDATE_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_preprocessor.pkl')

# Load TensorFlow models
def load_tf_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        print(f"Error loading TensorFlow model from {path}: {e}")
        return None

# Load preprocessor pickles
def load_preprocessor(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.PickleError) as e:
        print(f"Error loading preprocessor from {path}: {e}")
        return None

def prepare_job_features(job_posting):
    """Extract features from a job posting for the model"""
    # Extract relevant features based on the structure expected by the preprocessor
    features = {
        'title': job_posting.title,
        'required_skills': job_posting.skills_required,
        'description': job_posting.description,
        'requirements': job_posting.requirements,
        'responsibilities': job_posting.responsibilities,
        'min_experience': job_posting.experience_required,
        'education_level': job_posting.education_level,
        'location': job_posting.location,
        'employment_type': job_posting.employment_type,
    }
    return features

def prepare_resume_features(resume):
    """Extract features from a resume for the model"""
    # Extract relevant features based on the structure expected by the preprocessor
    features = {
        'skills': resume.skills,
        'education': resume.education,
        'experience': resume.experience,
        'extracted_text': resume.extracted_text,
    }
    return features

def generate_job_embedding(job_posting):
    """Generate embedding vector for a job posting using the job encoder model"""
    job_model = load_tf_model(JOB_MODEL_PATH)
    job_preprocessor = load_preprocessor(JOB_PREPROCESSOR_PATH)
    
    if not job_model or not job_preprocessor:
        return None
    
    # Extract features from job posting
    features = prepare_job_features(job_posting)
    
    # Apply preprocessor transformation
    try:
        processed_features = job_preprocessor.transform(features)
        
        # Get the embedding from the job encoder model
        embedding = job_model.predict(processed_features)
        return embedding
    except Exception as e:
        print(f"Error generating job embedding: {e}")
        return None

def generate_resume_embedding(resume):
    """Generate embedding vector for a resume using the candidate encoder model"""
    candidate_model = load_tf_model(CANDIDATE_MODEL_PATH)
    candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    
    if not candidate_model or not candidate_preprocessor:
        return None
    
    # Extract features from resume
    features = prepare_resume_features(resume)
    
    # Apply preprocessor transformation
    try:
        processed_features = candidate_preprocessor.transform(features)
        
        # Get the embedding from the candidate encoder model
        embedding = candidate_model.predict(processed_features)
        return embedding
    except Exception as e:
        print(f"Error generating resume embedding: {e}")
        return None

def calculate_match_score(job_embedding, resume_embedding):
    """Calculate match score using the matching model or cosine similarity"""
    # Try to use the matching model if available
    matching_model = load_tf_model(MATCHING_MODEL_PATH)
    
    if matching_model and job_embedding is not None and resume_embedding is not None:
        try:
            # Prepare input for matching model
            job_vector = np.frombuffer(job_embedding, dtype=np.float32).reshape(1, -1)
            resume_vector = np.frombuffer(resume_embedding, dtype=np.float32).reshape(1, -1)
            
            # Concatenate or prepare the inputs as expected by the matching model
            model_input = [job_vector, resume_vector]  # Adjust based on how your model expects inputs
            
            # Get predicted match score
            match_score = float(matching_model.predict(model_input)[0][0])
            
            # Scale to 0-100 if needed
            if match_score <= 1.0:
                match_score = match_score * 100
                
            return match_score
        except Exception as e:
            print(f"Error using matching model: {e}")
            # Fall back to cosine similarity if there's an error
    
    # Fall back to cosine similarity if matching model isn't available or there was an error
    if job_embedding is None or resume_embedding is None:
        return 0
    
    # Convert binary fields to numpy arrays if needed
    if isinstance(job_embedding, bytes):
        job_vector = np.frombuffer(job_embedding, dtype=np.float32)
    else:
        job_vector = job_embedding.flatten()
        
    if isinstance(resume_embedding, bytes):
        resume_vector = np.frombuffer(resume_embedding, dtype=np.float32)
    else:
        resume_vector = resume_embedding.flatten()
    
    # Calculate cosine similarity
    dot_product = np.dot(job_vector, resume_vector)
    norm_job = np.linalg.norm(job_vector)
    norm_resume = np.linalg.norm(resume_vector)
    
    if norm_job == 0 or norm_resume == 0:
        return 0
    
    similarity = dot_product / (norm_job * norm_resume)
    
    # Convert similarity to a percentage score (0-100)
    match_score = (similarity + 1) / 2 * 100
    
    return match_score

def update_job_matches(job_posting):
    """Update match scores for all candidates with this job posting"""
    if not job_posting.embedding_vector:
        return
        
    resumes = Resume.objects.all()
    
    for resume in resumes:
        if resume.embedding_vector:
            match_score = calculate_match_score(job_posting.embedding_vector, resume.embedding_vector)
            
            # Check if application already exists
            application, created = JobApplication.objects.get_or_create(
                job=job_posting,
                candidate=resume.candidate,
                defaults={
                    'resume': resume,
                    'match_score': match_score,
                    'status': 'pending'
                }
            )
            
            # Update match score if application already exists but don't change the status
            if not created:
                application.match_score = match_score
                application.save(update_fields=['match_score', 'updated_at'])