# from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from accounts.models import User

# User = get_user_model()

class EmailOrUsernameModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            # Try to find a user matching the username
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            try:
                # Or try to find a user matching the email
                user = User.objects.get(email=username)
            except User.DoesNotExist:
                # No user was found matching either username or email
                return None
        
        # Now check the password
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None