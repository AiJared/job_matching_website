# candidate/models.py
from django.db import models
from django.utils.translation import gettext as _
from accounts.models import User, Candidate
from dashboards.models import JobPosting, Resume, JobApplication

class SavedJob(models.Model):
    """Model for jobs saved by candidates"""
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='saved_jobs')
    job = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name='saved_by')
    saved_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('candidate', 'job')
        verbose_name = _('Saved Job')
        verbose_name_plural = _('Saved Jobs')
        ordering = ['-saved_at']
        
    def __str__(self):
        return f"{self.job.title} saved by {self.candidate.user.username}"

class JobSearchHistory(models.Model):
    """Model to track candidate job search history"""
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='search_history')
    query = models.CharField(_('search query'), max_length=255)
    location = models.CharField(_('location'), max_length=255, blank=True, null=True)
    category = models.CharField(_('category'), max_length=100, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('Job Search History')
        verbose_name_plural = _('Job Search Histories')
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"Search by {self.candidate.user.username}: {self.query}"

class ProfileCompletionTask(models.Model):
    """Model to track profile completion tasks"""
    TASK_TYPES = (
        ('resume_upload', 'Upload Resume'),
        ('skills_add', 'Add Skills'),
        ('education_add', 'Add Education'),
        ('experience_add', 'Add Experience'),
        ('profile_pic', 'Add Profile Picture'),
        ('contact_info', 'Complete Contact Information'),
    )
    
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='completion_tasks')
    task_type = models.CharField(_('task type'), max_length=50, choices=TASK_TYPES)
    is_completed = models.BooleanField(_('completed'), default=False)
    completed_at = models.DateTimeField(_('completed at'), null=True, blank=True)
    
    class Meta:
        unique_together = ('candidate', 'task_type')
        verbose_name = _('Profile Completion Task')
        verbose_name_plural = _('Profile Completion Tasks')
        
    def __str__(self):
        status = "Completed" if self.is_completed else "Pending"
        return f"{self.get_task_type_display()} - {status}"