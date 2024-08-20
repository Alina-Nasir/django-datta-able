from django.urls import re_path
from django.views.decorators.csrf import csrf_exempt

from api.views import *


urlpatterns = [

	re_path("product/((?P<pk>\d+)/)?", csrf_exempt(ProductView.as_view())),
	re_path("invoice/((?P<pk>\d+)/)?", csrf_exempt(InvoiceView.as_view())),
	re_path("port-invoice/((?P<pk>\d+)/)?", csrf_exempt(PortInvoiceView.as_view())),

]