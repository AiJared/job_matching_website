# dashboards/forms.py
from django import forms
from .models import JobPosting, JobApplication

class JobPostingForm(forms.ModelForm):
    expiry_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        help_text="The date when this job posting should expire"
    )
    
    class Meta:
        model = JobPosting
        exclude = ['recruiter', 'embedding_vector', 'status']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 5}),
            'requirements': forms.Textarea(attrs={'rows': 5}),
            'responsibilities': forms.Textarea(attrs={'rows': 5}),
            'skills_required': forms.Textarea(attrs={'rows': 3}),
            'category': forms.TextInput(attrs={'placeholder': 'Enter job category'}),
        }

class ApplicationStatusUpdateForm(forms.ModelForm):
    class Meta:
        model = JobApplication
        fields = ['status', 'recruiter_notes']
        widgets = {
            'recruiter_notes': forms.Textarea(attrs={'rows': 3}),
        }