from django.urls import path
from .views import predict_gauge_view

urlpatterns = [
    path('', predict_gauge_view, name='gauge_ui'),
]
