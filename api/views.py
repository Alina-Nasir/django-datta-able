from http import HTTPStatus
from django.http import Http404
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticatedOrReadOnly


from api.serializers import *


try:

    from home.models import Product, Invoice

except:
    pass

class ProductView(APIView):

    permission_classes = (IsAuthenticatedOrReadOnly,)

    def post(self, request):
        serializer = ProductSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTPStatus.BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Record Created.',
            'success': True
        }, status=HTTPStatus.OK)

    def get(self, request, pk=None):
        if not pk:
            return Response({
                'data': [ProductSerializer(instance=obj).data for obj in Product.objects.all()],
                'success': True
            }, status=HTTPStatus.OK)
        try:
            obj = get_object_or_404(Product, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        return Response({
            'data': ProductSerializer(instance=obj).data,
            'success': True
        }, status=HTTPStatus.OK)

    def put(self, request, pk):
        try:
            obj = get_object_or_404(Product, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        serializer = ProductSerializer(instance=obj, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTPStatus.BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Record Updated.',
            'success': True
        }, status=HTTPStatus.OK)

    def delete(self, request, pk):
        try:
            obj = get_object_or_404(Product, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        obj.delete()
        return Response(data={
            'message': 'Record Deleted.',
            'success': True
        }, status=HTTPStatus.OK)

class InvoiceView(APIView):

    permission_classes = (IsAuthenticatedOrReadOnly,)

    def post(self, request):
        serializer = InvoiceSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTP_400_BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Invoice Created.',
            'success': True
        }, status=HTTP_200_OK)

    def get(self, request, pk=None):
        if not pk:
            return Response({
                'data': [InvoiceSerializer(instance=obj).data for obj in Invoice.objects.all()],
                'success': True
            }, status=HTTP_200_OK)
        try:
            obj = get_object_or_404(Invoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        return Response({
            'data': InvoiceSerializer(instance=obj).data,
            'success': True
        }, status=HTTP_200_OK)

    def put(self, request, pk):
        try:
            obj = get_object_or_404(Invoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        serializer = InvoiceSerializer(instance=obj, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTP_400_BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Invoice Updated.',
            'success': True
        }, status=HTTP_200_OK)

    def delete(self, request, pk):
        try:
            obj = get_object_or_404(Invoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        obj.delete()
        return Response(data={
            'message': 'Invoice Deleted.',
            'success': True
        }, status=HTTP_200_OK)

class PortInvoiceView(APIView):

    permission_classes = (IsAuthenticatedOrReadOnly,)

    def post(self, request):
        serializer = PortInvoiceSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTP_400_BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Port Invoice Created.',
            'success': True
        }, status=HTTP_200_OK)

    def get(self, request, pk=None):
        if not pk:
            return Response({
                'data': [PortInvoiceSerializer(instance=obj).data for obj in Invoice.objects.all()],
                'success': True
            }, status=HTTP_200_OK)
        try:
            obj = get_object_or_404(PortInvoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Port Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        return Response({
            'data': PortInvoiceSerializer(instance=obj).data,
            'success': True
        }, status=HTTP_200_OK)

    def put(self, request, pk):
        try:
            obj = get_object_or_404(PortInvoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Port Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        serializer = PortInvoiceSerializer(instance=obj, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTP_400_BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Port Invoice Updated.',
            'success': True
        }, status=HTTP_200_OK)

    def delete(self, request, pk):
        try:
            obj = get_object_or_404(PortInvoice, pk=pk)
        except Http404:
            return Response(data={
                'message': 'Port Invoice with the given id not found.',
                'success': False
            }, status=HTTP_404_NOT_FOUND)
        obj.delete()
        return Response(data={
            'message': 'Port Invoice Deleted.',
            'success': True
        }, status=HTTP_200_OK)