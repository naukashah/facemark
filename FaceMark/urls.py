"""FaceMark URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls.static import static
from django.urls import path
from django.views.generic import TemplateView
from fireapp import views
from django.conf import settings
from django.conf.urls.static import static

from FaceMark import settings

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', views.index, name='index'),
                  path('login.html', TemplateView.as_view(template_name='login.html')),
                  path('datasets.html', views.onCLickDatasets, name='datasets'),
                  path('embeddings.html', views.onClickEmbeddings, name='embeddings'),
                  path('train.html', views.onClickTrain, name='train'),
                  path('login.html', views.loginUser, name='login'),

                  path('temp.html', views.image_upload, name='image_upload'),

                  path('embedding', views.embedding, name='embedding'),
                  path('trainModel', views.trainModel, name='trainModel'),
                  path('attendance_list', views.attendance_list, name='attendance_list'),
                  path('add_record', views.add_record, name='add_record'),
                  path('camera', views.camera, name='camera'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
