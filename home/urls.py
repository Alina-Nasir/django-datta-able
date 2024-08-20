from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

urlpatterns = [
  path(''       , views.index,  name='index'),
  path('tables/', views.tables, name='tables'),
  path('invoices/', views.invoices, name='invoices'),
  path('port-invoices/', views.port_invoices, name='port-invoices'),
]
