# web_multi_agent_automl/pipeline_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('run-pipeline/', views.run_pipeline_view, name='run_pipeline'),
]
