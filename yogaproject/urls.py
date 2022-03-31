"""yogaproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from yoga import views
from django.conf import settings
from django.conf.urls.static import static
from yoga.views import ProfileView

# from .views import camera

urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/',views.sign_up, name='signup'),
    path('login/', views.user_login, name='login'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('about/', views.about, name='about'),
    path('logout/',views.user_logout, name='logout'),
    path('changepassword/', views.user_changepass, name='changepassword'),
    path('advanced/', views.advanced, name='advanced'),
    path('beginner/', views.beginner, name='beginner'),
    path('intermediate/', views.intermediate, name='intermediate'),
    #path('custom/', views.custom, name='custom'),
    path('classes/', views.classes, name='classes'),
    path('forgot_password/', views.forgot_password, name='forgot_password'),
    path('send_mail_forgot_password/', views.send_mail_forgot_password, name='send_forgot_password'),
    path('reset_password_form/', views.reset_password_form, name='reset_password_form'),
    path('contact/', views.contactsendmsg, name='contactpage'),
    path('feed', views.feed, name='feed'),
    path('test/', views.test, name='test'),
    path('camera1/', views.camera1, name='camera1'),
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

