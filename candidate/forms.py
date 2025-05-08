# candidate/forms.py
from django import forms
from django.utils.translation import gettext as _
from dashboards.models import Resume, JobApplication
from accounts.models import Candidate

class CandidateProfileForm(forms.ModelForm):
    """Form for candidate to update skills, education, and experience"""
    class Meta:
        model = Candidate
        fields = ['skills', 'education', 'experience', 'location']
        widgets = {
            'skills': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': _('List your skills separated by commas, e.g., Python, Django, SQL')
            }),
            'education': forms.Textarea(attrs={
                'rows': 4,
                'placeholder': _('Describe your education, degrees, institutions...')
            }),
            'years_of_experience': forms.IntegerField(attrs={
                'rows': 4,
                'placeholder': _('e.g. 5')
            }),

            'location': forms.TextInput(attrs={'placeholder': 'e.g. Nairobi, Kenya'}),
            'preference': forms.TextInput(attrs={'placeholder': 'e.g. Remote, On-site'}),
        }

class ResumeUploadForm(forms.ModelForm):
    """Form for uploading a candidate's resume"""
    class Meta:
        model = Resume
        fields = ['file']
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if file.size > 5 * 1024 * 1024:  # 5MB limit
                raise forms.ValidationError(_("File size must be no more than 5MB."))
            allowed_extensions = ['pdf', 'docx', 'doc']
            ext = file.name.split('.')[-1].lower()
            if ext not in allowed_extensions:
                raise forms.ValidationError(_(f"Only {', '.join(allowed_extensions)} files are allowed."))
        return file

class JobApplicationForm(forms.ModelForm):
    """Form for applying to a job"""
    class Meta:
        model = JobApplication
        fields = ['cover_letter']
        widgets = {
            'cover_letter': forms.Textarea(attrs={
                'rows': 5, 
                'placeholder': _('Tell the employer why you are a good fit for this position...')
            }),
        }

class JobSearchForm(forms.Form):
    """Form for searching jobs"""
    query = forms.CharField(
        required=False, 
        widget=forms.TextInput(attrs={'placeholder': _('Job title, keywords, or company')})
    )
    location = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'placeholder': _('City or country')})
    )
    category = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'placeholder': _('Job category')})
    )
