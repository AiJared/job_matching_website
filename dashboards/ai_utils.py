# dashboards/ai_utils.py
import numpy as np
import pickle
import os
import traceback
import pandas as pd
import keras
import tensorflow as tf
from django.conf import settings
from .models import JobPosting, Resume, JobApplication

# Paths to saved models and preprocessors
JOB_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_model.keras')
CANDIDATE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_model.keras')
MATCHING_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'matching_model.keras')
JOB_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'job_preprocessor.pkl')
CANDIDATE_PREPROCESSOR_PATH = os.path.join(settings.BASE_DIR, 'ai_models', 'candidate_preprocessor.pkl')

def load_tf_model(path):
    """Simulated model loading - returns a placeholder"""
    print(f"Using simulated model instead of loading from {path}")
    return "simulated_model"

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
    """Generate embedding vector for a job posting using text processing."""
    print("Generating job embedding from job posting fields...")

    features = prepare_job_features(job_posting)

    text_data = ' '.join([
        features.get('title') or '',
        features.get('required_skills') or '',
        features.get('description') or '',
        features.get('requirements') or '',
        features.get('responsibilities') or '',
        str(features.get('min_experience') or ''),
        features.get('education_level') or '',
        features.get('location') or '',
        features.get('employment_type') or ''
    ]).lower().strip()

    if not text_data:
        print("Warning: No text found in job posting. Using default text.")
        text_data = "default job posting skills requirements education"

    try:
        import hashlib
        import numpy as np

        embedding = np.zeros(64)

        for word in text_data.split():
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            embedding[hash_value % 64] += 1

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        print("Job embedding generated successfully.")
        return embedding.reshape(1, -1).astype(np.float32)

    except Exception as e:
        print(f"Error generating job embedding: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def generate_resume_embedding(resume):
    """Generate embedding vector for a resume based on manually filled fields."""
    print("Generating candidate resume embedding from form fields...")
    
    candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    if not candidate_preprocessor:
        print("Failed to load candidate preprocessor.")
        return None

    # Build a clean text from actual candidate-provided fields
    features = prepare_resume_features(resume)

    # ✅ Assemble text properly
    skills = features.get('skills') or ""
    education = features.get('education') or ""
    experience = features.get('experience') or ""

    text_data = f"{skills} {education} {experience}".strip().lower()

    if not text_data:
        print("Warning: No skills, education, or experience provided by candidate.")
        text_data = "default resume candidate skills education experience"

    # ✅ Hash-based simple embedding
    try:
        import hashlib
        import numpy as np

        embedding = np.zeros(64)

        for word in text_data.split():
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            embedding[hash_value % 64] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Return in the correct format
        print("Embedding generated successfully.")
        return embedding.reshape(1, -1).astype(np.float32)

    except Exception as e:
        print(f"Error generating resume embedding: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def calculate_match_score(job_embedding, resume_embedding):
    """Calculate match score using cosine similarity"""
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
    
    return min(100, match_score)  # Ensure max is 100

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


def test_model_loading():
    """Test function to verify simulated model functionality"""
    import os
    import numpy as np
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for preprocessors in: {os.path.join(os.getcwd(), 'ai_models')}")
    
    # Check if preprocessor files exist
    print(f"Job preprocessor exists: {os.path.exists(JOB_PREPROCESSOR_PATH)}")
    print(f"Candidate preprocessor exists: {os.path.exists(CANDIDATE_PREPROCESSOR_PATH)}")
    
    # Load preprocessors
    job_preprocessor = load_preprocessor(JOB_PREPROCESSOR_PATH)
    candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    
    # Test simulated embeddings
    print("Testing simulated job embedding generation...")
    test_job_features = {
        'title': 'Software Engineer',
        'required_skills': 'Python, Django, JavaScript',
        'description': 'Join our team of developers',
        'requirements': 'Bachelor degree, 2 years experience',
        'responsibilities': 'Develop web applications',
        'min_experience': 2,
        'education_level': 'Bachelor',
        'location': 'Remote',
        'employment_type': 'Full-time',
    }
    
    # Create a mock job posting object
    class MockJobPosting:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    mock_job = MockJobPosting(
        title=test_job_features['title'],
        skills_required=test_job_features['required_skills'],
        description=test_job_features['description'],
        requirements=test_job_features['requirements'],
        responsibilities=test_job_features['responsibilities'],
        experience_required=test_job_features['min_experience'],
        education_level=test_job_features['education_level'],
        location=test_job_features['location'],
        employment_type=test_job_features['employment_type']
    )
    
    job_embedding = generate_job_embedding(mock_job)
    print(f"Job embedding shape: {job_embedding.shape if job_embedding is not None else 'None'}")
    
    # Test simulated resume embedding
    print("Testing simulated resume embedding generation...")
    class MockResume:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    mock_resume = MockResume(
        skills="Python, Django, JavaScript",
        education="Bachelor in Computer Science",
        experience="2 years as a web developer",
        extracted_text="Experienced software engineer with focus on web development"
    )
    
    resume_embedding = generate_resume_embedding(mock_resume)
    print(f"Resume embedding shape: {resume_embedding.shape if resume_embedding is not None else 'None'}")
    
    # Test matching
    if job_embedding is not None and resume_embedding is not None:
        match_score = calculate_match_score(job_embedding, resume_embedding)
        print(f"Match score: {match_score}")
    
    return {
        'job_preprocessor': job_preprocessor is not None,
        'candidate_preprocessor': candidate_preprocessor is not None,
        'simulated_job_embedding': job_embedding is not None,
        'simulated_resume_embedding': resume_embedding is not None,
        'matching_functionality': job_embedding is not None and resume_embedding is not None
    }