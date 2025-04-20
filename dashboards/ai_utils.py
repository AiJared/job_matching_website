# dashboards/ai_utils.py
import numpy as np
import pickle
import os
from django.conf import settings
from .models import JobPosting, Resume, JobApplication

# This would be replaced with actual paths to your saved models
JOB_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_encoder.pkl')
CANDIDATE_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_encoder.pkl')
JOB_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_preprocessor.pkl')
CANDIDATE_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_preprocessor.pkl')

def load_model(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.PickleError) as e:
        print(f"Error loading model from {path}: {e}")
        return None

def prepare_job_features(job_posting):
    """Extract features from a job posting for the model"""
    # This is a placeholder - implement based on your actual feature extraction logic
    features = {
        'title': job_posting.title,
        'description': job_posting.description,
        'requirements': job_posting.requirements,
        'responsibilities': job_posting.responsibilities,
        'skills': job_posting.skills_required,
        'experience': job_posting.experience_required,
        'education': job_posting.education_level,
        'location': job_posting.location,
    }
    return features

def prepare_resume_features(resume):
    """Extract features from a resume for the model"""
    # This is a placeholder - implement based on your actual feature extraction logic
    features = {
        'skills': resume.skills,
        'education': resume.education,
        'experience': resume.experience,
        'extracted_text': resume.extracted_text,
    }
    return features

def generate_job_embedding(job_posting):
    """Generate embedding vector for a job posting"""
    preprocessor = load_model(JOB_PREPROCESSOR_PATH)
    encoder = load_model(JOB_ENCODER_PATH)
    
    if not preprocessor or not encoder:
        return None
    
    features = prepare_job_features(job_posting)
    processed_features = preprocessor.transform(features)
    embedding = encoder.predict(processed_features)
    
    return embedding

def generate_resume_embedding(resume):
    """Generate embedding vector for a resume"""
    preprocessor = load_model(CANDIDATE_PREPROCESSOR_PATH)
    encoder = load_model(CANDIDATE_ENCODER_PATH)
    
    if not preprocessor or not encoder:
        return None
    
    features = prepare_resume_features(resume)
    processed_features = preprocessor.transform(features)
    embedding = encoder.predict(processed_features)
    
    return embedding

def calculate_match_score(job_embedding, resume_embedding):
    """Calculate cosine similarity between job and resume embeddings"""
    if job_embedding is None or resume_embedding is None:
        return 0
    
    # Convert binary fields to numpy arrays
    job_vector = np.frombuffer(job_embedding, dtype=np.float32)
    resume_vector = np.frombuffer(resume_embedding, dtype=np.float32)
    
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
            
            # Update match score if application already exists
            if not created:
                application.match_score = match_score
                application.save()