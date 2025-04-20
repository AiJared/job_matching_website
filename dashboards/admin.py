# dashboards/admin.py
from django.contrib import admin
from .models import JobCategory, JobPosting, Resume, JobApplication

@admin.register(JobCategory)
class JobCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)

@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = ('title', 'company_name', 'location', 'status', 'expiry_date', 'created_at')
    list_filter = ('status', 'category', 'employment_type')
    search_fields = ('title', 'company_name', 'description', 'requirements')
    date_hierarchy = 'created_at'

@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('candidate', 'created_at', 'updated_at')
    search_fields = ('candidate__user__full_name', 'skills', 'education', 'experience')

@admin.register(JobApplication)
class JobApplicationAdmin(admin.ModelAdmin):
    list_display = ('job', 'candidate', 'status', 'match_score', 'created_at')
    list_filter = ('status',)
    search_fields = ('job__title', 'candidate__user__full_name')
    date_hierarchy = 'created_at'