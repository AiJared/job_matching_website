from django.urls import path

from dashboards.views import (admin_dashboard, candidate_dashboard,
                              recruiter_dashboard)

app_name = "dashboards"

urlpatterns = [
    path("admin/", admin_dashboard, name="admin_dashboard"),
    path("recruiter/", recruiter_dashboard, name="recruiter_dashboard"),
    path("candidate/", candidate_dashboard, name="candidate_dashboard"),
]