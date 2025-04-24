# candidate/admin.py
from django.contrib import admin
from .models import SavedJob, JobSearchHistory, ProfileCompletionTask

@admin.register(SavedJob)
class SavedJobAdmin(admin.ModelAdmin):
    list_display = ('candidate', 'job', 'saved_at')
    list_filter = ('saved_at',)
    search_fields = ('candidate__user__username', 'job__title')
    date_hierarchy = 'saved_at'

@admin.register(JobSearchHistory)
class JobSearchHistoryAdmin(admin.ModelAdmin):
    list_display = ('candidate', 'query', 'location', 'category', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('candidate__user__username', 'query', 'location', 'category')
    date_hierarchy = 'timestamp'

@admin.register(ProfileCompletionTask)
class ProfileCompletionTaskAdmin(admin.ModelAdmin):
    list_display = ('candidate', 'task_type', 'is_completed', 'completed_at')
    list_filter = ('task_type', 'is_completed', 'completed_at')
    search_fields = ('candidate__user__username',)