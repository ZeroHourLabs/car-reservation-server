from django.urls import path
from .views import chat_with_local

urlpatterns = [
    path("chat/", chat_with_local, name="chat_with_local"),
]
