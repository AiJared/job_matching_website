from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Group
from accounts.models import User, Administrator, Recruiter, Candidate

class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'full_name', 'role', 'is_active', 'is_admin')
    list_filter = ('is_admin', 'is_active', 'role')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('username', 'full_name', 'gender', 'phone', 'country', 'city')}),
        ('Permissions', {'fields': ('is_admin', 'is_staff', 'is_active', 'role')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'full_name', 'gender', 'phone', 'country', 'city', 'role', 'password1', 'password2'),
        }),
    )
    search_fields = ('email', 'username', 'full_name')
    ordering = ('email',)
    filter_horizontal = ()

class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_email', 'get_full_name')
    search_fields = ('user__email', 'user__username', 'user__full_name')
    
    def get_email(self, obj):
        return obj.user.email
    get_email.short_description = 'Email'
    
    def get_full_name(self, obj):
        return obj.user.full_name
    get_full_name.short_description = 'Full Name'

admin.site.register(User, UserAdmin)
admin.site.register(Administrator, ProfileAdmin)
admin.site.register(Recruiter, ProfileAdmin)
admin.site.register(Candidate, ProfileAdmin)
admin.site.unregister(Group)