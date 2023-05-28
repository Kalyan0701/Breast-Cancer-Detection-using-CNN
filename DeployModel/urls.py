
from django.contrib import admin
from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dashboard', views.dashboard, name="dashboard"),
    path('result/', views.result, name="result"),
    path('patient/', views.patient, name="patient"),
    path('', views.login_view, name='login'),
    path('deletePatient/<int:id>', views.deletePatient, name="deletePatient"),
    path('user_logout', views.user_logout, name="user_logout"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
