# dashboards/management/commands/test_ai_models.py
from django.core.management.base import BaseCommand
from dashboards.ai_utils import test_model_loading

class Command(BaseCommand):
    help = 'Test AI model loading and functionality'

    def handle(self, *args, **options):
        self.stdout.write('Testing AI model loading...')
        results = test_model_loading()
        
        for name, loaded in results.items():
            status = 'LOADED' if loaded else 'FAILED'
            self.stdout.write(f'{name}: {status}')