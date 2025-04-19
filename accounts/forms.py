from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from accounts.models import Candidate, Recruiter

User = get_user_model()

class UserRegistrationForm(UserCreationForm):
    ROLE_CHOICES = (
        ('Candidate', 'Job Seeker'),
        ('Recruiter', 'Recruiter'),
    )
    
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True, widget=forms.RadioSelect)
    email = forms.EmailField(required=True)
    full_name = forms.CharField(max_length=200, required=True)
    gender = forms.ChoiceField(choices=User.Gender_choices, required=True)
    phone = forms.CharField(max_length=15, required=False)
    country = forms.CharField(max_length=50, required=False)
    city = forms.CharField(max_length=50, required=False)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'full_name', 'gender', 'phone', 'country', 'city', 'role', 'password1', 'password2']
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.full_name = self.cleaned_data['full_name']
        user.gender = self.cleaned_data['gender']
        user.phone = self.cleaned_data['phone']
        user.country = self.cleaned_data['country']
        user.city = self.cleaned_data['city']
        user.role = self.cleaned_data['role']
        
        if commit:
            user.save()
            # Create corresponding profile based on role
            if user.role == 'Candidate':
                Candidate.objects.create(user=user)
            elif user.role == 'Recruiter':
                Recruiter.objects.create(user=user)
        
        return user

class UserLoginForm(AuthenticationForm):
    username = forms.CharField(label='Email / Username')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'autofocus': True})