from django.db import models
from django.core.validators import RegexValidator

# Create your models here.
from django.contrib.auth import validators
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.db import models
from django.utils.translation import gettext as _
# from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField
from django.core.validators import (
    MaxLengthValidator,
    MaxValueValidator,
    MinLengthValidator)


class TrackingModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        ordering = ["-created_at"]


class CustomManager(BaseUserManager):
    def create_user(self, email, username, password=None, is_active=True, is_admin=False, is_staff=False, role="", **others):
        if not email:
            raise ValueError("User must have an email address!")
        if not password:
            raise ValueError("User must have a password!")
        if not username:
            raise ValueError("User must have a username!")
        user_obj = self.model(
            email = self.normalize_email(email),
            username = username,
            **others
        )
        user_obj.set_password(password)
        user_obj.is_active = is_active
        user_obj.is_admin = is_admin
        user_obj.is_staff = is_staff
        user_obj.role = role
        user_obj.save(using=self._db)

        return user_obj

    def create_staff(self, email, username, password=None,**others):
        user = self.create_user(
            email, username, password=password, is_active=True,
            is_staff=False, is_admin=False, role="Administrator",**others,
        )
        return user
    
    def create_superuser(self, email, username, password=None,**others):
        user = self.create_user(
            email, username, password=password, is_active=True,
            is_staff=True, is_admin=True, role="Administrator",**others
        )
        return user


class User(AbstractBaseUser, TrackingModel):
    username_validator = UnicodeUsernameValidator()
    Role_choices = (
        ("Administrator", "Administrator"),
        ("Recruiter", "Recruiter"),
        ("Candidate", "Candidate")
    )
    Gender_choices = (
        ("Male", "Male"),
        ("Female", "Female"),
        ("Other", "Other")
    )
    username = models.CharField(
        _('username'),
        max_length=150,
        unique=True,
        validators=[username_validator],
        error_messages={
            'unique':_("A user with that username already exists."),
        },
    )
    phoneNumberRegex = RegexValidator(regex=r"^\+?1?\d{8,15}$")
    full_name = models.CharField(_('full name'),max_length=200)
    # identification = models.IntegerField(_("identification"), null=True)
    email = models.EmailField(_('email'), unique=True, error_messages={
        'unique': ('A user with email already exists.'),
    })
    gender = models.CharField(_("gender"), max_length=10, choices=Gender_choices)
    phone = PhoneNumberField(
        _('phone number'), unique=True,validators=[phoneNumberRegex],
        blank=True, null=True, max_length=27)
    country = models.CharField(_("country"), max_length=50,
                              blank=True, null=True)
    city = models.CharField(_("city"), max_length=50,
                              blank=True, null=True)
    role = models.CharField(_('Role'), max_length=17, choices=Role_choices, default="Client")

    is_active = models.BooleanField(_('active'), default=True)
    is_admin = models.BooleanField(_('admin'), default=False)
    is_staff = models.BooleanField(_("staff"), default=False)
    timestamp = models.DateTimeField(_("timestamp"), auto_now_add=True)

    def __str__(self):
        return self.username
    
    objects = CustomManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username',]

    def has_perm(self, perm, obj=None):
        return True
    
    def has_module_perms(self, app_label):
        return True
    
    @property
    def staff(self):
        return self.staff
    
    @property
    def active(self):
        return self.active

    @property
    def admin(self):
        return self.admin


class Profile(models.Model):
    user = models.OneToOneField(User,
                                on_delete=models.CASCADE, unique=True)
    bio = models.TextField(_('bio'), blank=True, null=True)
    profile_picture = models.ImageField(
        _("profile picture"), upload_to="profile",
        default="default.png")
    
    class Meta:
        abstract= True


class Administrator(Profile):
    pass
  
    def __str__(self):
        return self.user.username


class Recruiter(Profile):
    pass
    
    def __str__(self):
        return self.user.username
    
class Candidate(Profile):
    skills = models.TextField(_('skills'), blank=True, null=True)
    education = models.TextField(_('education'), blank=True, null=True)
    experience = models.TextField(_('experience'), blank=True, null=True)

    def __str__(self):
        return self.user.username