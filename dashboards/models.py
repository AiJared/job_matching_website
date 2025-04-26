# dashboards/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.translation import gettext as _
from accounts.models import User, Recruiter, Candidate
from django.utils import timezone

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class JobPosting(BaseModel):
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('filled', 'Filled'),
        ('expired', 'Expired'),
        ('draft', 'Draft'),
    )
    
    recruiter = models.ForeignKey(Recruiter, on_delete=models.CASCADE, related_name='job_postings')
    title = models.CharField(_('job title'), max_length=255)
    company_name = models.CharField(_('company name'), max_length=255)
    # Change from ForeignKey to CharField
    category = models.CharField(_('job category'), max_length=100)
    description = models.TextField(_('job description'))
    requirements = models.TextField(_('job requirements'))
    responsibilities = models.TextField(_('responsibilities'))
    location = models.CharField(_('location'), max_length=255)
    salary_min = models.DecimalField(_('minimum salary'), max_digits=12, decimal_places=2, null=True, blank=True)
    salary_max = models.DecimalField(_('maximum salary'), max_digits=12, decimal_places=2, null=True, blank=True)
    experience_required = models.PositiveIntegerField(_('experience required (years)'), default=0)
    education_level = models.CharField(_('education level'), max_length=100)
    employment_type = models.CharField(_('employment type'), max_length=50)
    status = models.CharField(_('status'), max_length=20, choices=STATUS_CHOICES, default='active')
    expiry_date = models.DateField(_('expiry date'), default=timezone.now)
    skills_required = models.TextField(_('skills required'))
    embedding_vector = models.BinaryField(_('job embedding vector'), null=True, blank=True)
    
    def __str__(self):
        return f"{self.title} at {self.company_name}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = _('Job Posting')
        verbose_name_plural = _('Job Postings')

class Resume(BaseModel):
    candidate = models.OneToOneField(Candidate, on_delete=models.CASCADE, related_name='resume')
    file = models.FileField(_('resume file'), upload_to='resumes/')
    extracted_text = models.TextField(_('extracted text'), blank=True, null=True)
    skills = models.TextField(_('skills'), blank=True, null=True)
    education = models.TextField(_('education'), blank=True, null=True)
    experience = models.TextField(_('experience'), blank=True, null=True)
    embedding_vector = models.BinaryField(_('resume embedding vector'), null=True, blank=True)
    is_processed = models.BooleanField(_('is processed'), default=False)  # Add this field
    
    def __str__(self):
        return f"Resume of {self.candidate.user.full_name}"

class JobApplication(BaseModel):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('reviewing', 'Reviewing'),
        ('shortlisted', 'Shortlisted'),
        ('rejected', 'Rejected'),
        ('interview', 'Interview'),
        ('offer', 'Offer Extended'),
        ('hired', 'Hired'),
        ('declined', 'Declined'),
    )
    
    job = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name='applications')
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='applications')
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name='applications', null=True, blank=True)
    cover_letter = models.TextField(_('cover letter'), blank=True, null=True)
    status = models.CharField(_('status'), max_length=20, choices=STATUS_CHOICES, default='pending')
    match_score = models.FloatField(_('match score'), validators=[MinValueValidator(0), MaxValueValidator(100)], default=0)
    recruiter_notes = models.TextField(_('recruiter notes'), blank=True, null=True)
    
    def __str__(self):
        return f"Application for {self.job.title} by {self.candidate.user.full_name}"
    
    class Meta:
        ordering = ['-match_score', '-created_at']
        verbose_name = _('Job Application')
        verbose_name_plural = _('Job Applications')