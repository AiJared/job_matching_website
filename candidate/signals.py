# candidate/signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from dashboards.models import Resume, JobApplication
from accounts.models import Candidate
from .utils import process_resume, update_candidate_matches
from .models import ProfileCompletionTask
from django.utils import timezone

@receiver(post_save, sender=Resume)
def resume_post_save(sender, instance, created, **kwargs):
    """Process resume when it's created or updated"""
    if created:
        # Create profile completion tasks for new candidates
        candidate = instance.candidate
        task_types = [task_type for task_type, _ in ProfileCompletionTask.TASK_TYPES]
        
        for task_type in task_types:
            ProfileCompletionTask.objects.get_or_create(
                candidate=candidate,
                task_type=task_type
            )
        
        # Mark resume upload task as completed
        task = ProfileCompletionTask.objects.get(
            candidate=candidate,
            task_type='resume_upload'
        )
        task.is_completed = True
        task.completed_at = timezone.now()
        task.save()

@receiver(post_save, sender=JobApplication)
def job_application_post_save(sender, instance, created, **kwargs):
    """Update related models when a job application is created or updated"""
    if created:
        # When a candidate applies for a job, remove it from saved jobs if it exists
        from .models import SavedJob
        SavedJob.objects.filter(
            candidate=instance.candidate,
            job=instance.job
        ).delete()