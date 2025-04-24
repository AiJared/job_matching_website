from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from accounts.forms import UserRegistrationForm, UserLoginForm
from accounts.models import User

def home(request):
    return render(request, 'index.html')

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Account created successfully. You can now log in.")
            return redirect('accounts:login')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'accounts/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        # Redirect based on user role
        if request.user.role == 'Candidate':
            return redirect('dashboards:candidate_dashboard')
        elif request.user.role == 'Recruiter':
            return redirect('dashboards:recruiter_dashboard')
        else:
            return redirect('accounts:home')
    
    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            
            # Try to authenticate with email or username
            user = None
            # First try with email
            try:
                user_obj = User.objects.get(email=username)
                user = authenticate(request, email=user_obj.email, password=password)
            except User.DoesNotExist:
                # Then try with username
                user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.full_name}!")
                
                # Redirect based on user role
                if user.role == 'Candidate':
                    return redirect('candidate:candidate_dashboard')
                elif user.role == 'Recruiter':
                    return redirect('dashboards:recruiter_dashboard')
                else:
                    return redirect('accounts:login')
            else:
                messages.error(request, "Invalid email/username or password.")
        else:
            messages.error(request, "Invalid email/username or password.")
    else:
        form = UserLoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('accounts:login')
