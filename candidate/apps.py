# candidate/apps.py
from django.apps import AppConfig

class CandidateConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'candidate'
    verbose_name = 'Job Seeker Dashboard'
    
    def ready(self):
        import candidate.signals  # Import signals to register them