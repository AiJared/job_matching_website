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

# # Load TensorFlow models
# def load_tf_model(path):
#     try:
#         # Allow Lambda deserialization
#         keras.config.enable_unsafe_deserialization()
#         model = tf.keras.models.load_model(path)
#         return model
#     except Exception as e:
#         print(f"Error loading TensorFlow model from {path}: {e}")
#         return None
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

# def generate_job_embedding(job_posting):
#     """Generate embedding vector for a job posting using the job encoder model"""
#     job_model = load_tf_model(JOB_MODEL_PATH)
#     job_preprocessor = load_preprocessor(JOB_PREPROCESSOR_PATH)
    
#     if not job_model or not job_preprocessor:
#         return None
    
#     # Extract features from job posting
#     features = prepare_job_features(job_posting)
    
#     # Apply preprocessor transformation
#     try:
#         processed_features = job_preprocessor.transform(features)
        
#         # Get the embedding from the job encoder model
#         embedding = job_model.predict(processed_features)
#         return embedding
#     except Exception as e:
#         print(f"Error generating job embedding: {e}")
#         return None
def generate_job_embedding(job_posting):
    """Generate embedding vector for a job posting using text processing"""
    job_preprocessor = load_preprocessor(JOB_PREPROCESSOR_PATH)
    
    if not job_preprocessor:
        return None
    
    # Extract features from job posting
    features = prepare_job_features(job_posting)
    
    # Create a simple embedding based on text features
    try:
        # Concatenate text fields
        text_data = ' '.join([
            features['title'],
            features['required_skills'],
            features['description'],
            features['requirements'],
            features['responsibilities'],
            str(features['min_experience']),
            features['education_level'],
            features['location'],
            features['employment_type']
        ])
        
        # Convert to lowercase
        text_data = text_data.lower()
        
        # Create a basic embedding (hash-based)
        import hashlib
        import numpy as np
        
        # Create a 64-dimensional embedding using hashing trick
        embedding = np.zeros(64)
        
        # Split text into words
        words = text_data.split()
        
        # Use word hashing to create embedding
        for word in words:
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            embedding[hash_value % 64] += 1
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Reshape to match expected format
        return embedding.reshape(1, -1)
        
    except Exception as e:
        print(f"Error generating job embedding: {e}")
        # return None
        # After all your processing...

        # Final step (replace your old return)
        return embedding.reshape(1, -1).astype(np.float32)



# def generate_resume_embedding(resume):
#     """Generate embedding vector for a resume using the candidate encoder model"""
    
#     print("Loading candidate model and preprocessor...")
#     candidate_model = load_tf_model(CANDIDATE_MODEL_PATH)
#     candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    
#     if not candidate_model:
#         print(f"Failed to load candidate model from {CANDIDATE_MODEL_PATH}")
#         return None
        
#     if not candidate_preprocessor:
#         print(f"Failed to load candidate preprocessor from {CANDIDATE_PREPROCESSOR_PATH}")
#         return None
    
#     # Extract features from resume
#     print("Preparing resume features...")
#     features = prepare_resume_features(resume)
#     print(f"Features extracted: {features.keys()}")
    
#     # Apply preprocessor transformation
#     try:
#         print("Transforming features with preprocessor...")
        
#         # First check if we need to use a DataFrame
#         if hasattr(candidate_preprocessor, 'feature_names_in_'):
#             print(f"Preprocessor expects these features: {candidate_preprocessor.feature_names_in_}")
            
#         # Try different approaches based on what the preprocessor might expect
#         try:
#             # Try direct transformation first
#             processed_features = candidate_preprocessor.transform(features)
#         except Exception:
#             try:
#                 # Try with pandas DataFrame next
#                 features_df = pd.DataFrame([features])
#                 processed_features = candidate_preprocessor.transform(features_df)
#             except Exception:
#                 # Try with a dictionary containing a single list item for each feature
#                 features_dict = {k: [v] for k, v in features.items()}
#                 features_df = pd.DataFrame(features_dict)
#                 processed_features = candidate_preprocessor.transform(features_df)
                
#         print("Feature transformation successful")
        
#         # Get the embedding from the candidate encoder model
#         print("Generating prediction from model...")
#         embedding = candidate_model.predict(processed_features)
#         print(f"Embedding generated with shape: {embedding.shape}")
#         return embedding
#     except Exception as e:
#         print(f"Error in generate_resume_embedding: {e}")
#         print(traceback.format_exc())
#         return None
def generate_resume_embedding(resume):
    """Generate embedding vector for a resume using text processing"""
    print("Using simulated candidate model...")
    candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    
    if not candidate_preprocessor:
        print("Failed to load candidate preprocessor")
        return None
    
    # Extract features from resume
    features = prepare_resume_features(resume)
    
    # Create a simple embedding based on text features
    try:
        # Safely handle None values in the text fields
        text_parts = []
        for field in ['skills', 'education', 'experience', 'extracted_text']:
            if features.get(field):
                text_parts.append(str(features[field]))
            else:
                # Use empty string if field is None or empty
                text_parts.append("")
                print(f"Warning: Resume field '{field}' is empty or None")
        
        # Concatenate text fields
        text_data = ' '.join(text_parts)
        
        # If we have no text at all, use placeholder text
        if not text_data.strip():
            print("Warning: All resume text fields are empty. Using placeholder text.")
            text_data = "resume placeholder generic skills education experience"
        
        # Convert to lowercase
        text_data = text_data.lower()
        
        # Create a basic embedding (hash-based)
        import hashlib
        import numpy as np
        
        # Create a 64-dimensional embedding using hashing trick
        embedding = np.zeros(64)
        
        # Split text into words
        words = text_data.split()
        
        # Use word hashing to create embedding
        for word in words:
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            embedding[hash_value % 64] += 1
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Reshape to match expected format
        return embedding.reshape(1, -1)
        
    except Exception as e:
        print(f"Error generating resume embedding: {e}")
        import traceback
        print(traceback.format_exc())
        # return None
        # After all your processing...

        # Final step (replace your old return)
        return embedding.reshape(1, -1).astype(np.float32)



# def calculate_match_score(job_embedding, resume_embedding):
#     """Calculate match score using the matching model or cosine similarity"""
#     # Try to use the matching model if available
#     matching_model = load_tf_model(MATCHING_MODEL_PATH)
    
#     if matching_model and job_embedding is not None and resume_embedding is not None:
#         try:
#             # Prepare input for matching model
#             job_vector = np.frombuffer(job_embedding, dtype=np.float32).reshape(1, -1)
#             resume_vector = np.frombuffer(resume_embedding, dtype=np.float32).reshape(1, -1)
            
#             # Concatenate or prepare the inputs as expected by the matching model
#             model_input = [job_vector, resume_vector]  # Adjust based on how your model expects inputs
            
#             # Get predicted match score
#             match_score = float(matching_model.predict(model_input)[0][0])
            
#             # Scale to 0-100 if needed
#             if match_score <= 1.0:
#                 match_score = match_score * 100
                
#             return match_score
#         except Exception as e:
#             print(f"Error using matching model: {e}")
#             # Fall back to cosine similarity if there's an error
    
#     # Fall back to cosine similarity if matching model isn't available or there was an error
#     if job_embedding is None or resume_embedding is None:
#         return 0
    
#     # Convert binary fields to numpy arrays if needed
#     if isinstance(job_embedding, bytes):
#         job_vector = np.frombuffer(job_embedding, dtype=np.float32)
#     else:
#         job_vector = job_embedding.flatten()
        
#     if isinstance(resume_embedding, bytes):
#         resume_vector = np.frombuffer(resume_embedding, dtype=np.float32)
#     else:
#         resume_vector = resume_embedding.flatten()
    
#     # Calculate cosine similarity
#     dot_product = np.dot(job_vector, resume_vector)
#     norm_job = np.linalg.norm(job_vector)
#     norm_resume = np.linalg.norm(resume_vector)
    
#     if norm_job == 0 or norm_resume == 0:
#         return 0
    
#     similarity = dot_product / (norm_job * norm_resume)
    
#     # Convert similarity to a percentage score (0-100)
#     match_score = (similarity + 1) / 2 * 100
    
#     return match_score

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

# def test_model_loading():
#     """Test function to verify model loading"""
#     import os
#     print(f"Current working directory: {os.getcwd()}")
#     print(f"Looking for models in: {os.path.join(os.getcwd(), 'ai_models')}")
    
#     # Check if model files exist
#     print(f"Job model exists: {os.path.exists(JOB_MODEL_PATH)}")
#     print(f"Candidate model exists: {os.path.exists(CANDIDATE_MODEL_PATH)}")
#     print(f"Matching model exists: {os.path.exists(MATCHING_MODEL_PATH)}")
#     print(f"Job preprocessor exists: {os.path.exists(JOB_PREPROCESSOR_PATH)}")
#     print(f"Candidate preprocessor exists: {os.path.exists(CANDIDATE_PREPROCESSOR_PATH)}")
    
#     # Try loading the models
#     job_model = load_tf_model(JOB_MODEL_PATH)
#     candidate_model = load_tf_model(CANDIDATE_MODEL_PATH)
#     matching_model = load_tf_model(MATCHING_MODEL_PATH)
    
#     # Try loading the preprocessors
#     job_preprocessor = load_preprocessor(JOB_PREPROCESSOR_PATH)
#     candidate_preprocessor = load_preprocessor(CANDIDATE_PREPROCESSOR_PATH)
    
#     return {
#         'job_model': job_model is not None,
#         'candidate_model': candidate_model is not None,
#         'matching_model': matching_model is not None,
#         'job_preprocessor': job_preprocessor is not None,
#         'candidate_preprocessor': candidate_preprocessor is not None
#     }

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