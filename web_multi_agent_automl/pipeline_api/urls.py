from django.urls import path
from . import views
from .log_streaming import log_stream_sse_view

urlpatterns = [
    path('start-pipeline/', views.run_pipeline_start_view, name='start_pipeline'),
    path('log-stream/<str:run_id>/', log_stream_sse_view, name='log_stream_sse'),
    path('pipeline-result/<str:run_id>/', views.get_pipeline_result_view, name='get_pipeline_result'),
]
