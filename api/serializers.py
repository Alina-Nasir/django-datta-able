from rest_framework import serializers


try:

    from home.models import Product, Invoice, PortInvoice

except:
    pass 

class ProductSerializer(serializers.ModelSerializer):
    class Meta:

        try:
            model = Product
        except:
            pass    
        fields = '__all__'

class InvoiceSerializer(serializers.ModelSerializer):
    class Meta:
        try:
            model = Invoice  # Reference the Invoice model
        except:
            pass
        fields = '__all__' 

class PortInvoiceSerializer(serializers.ModelSerializer):
    class Meta:
        try:
            model = PortInvoice  # Reference the Invoice model
        except:
            pass
        fields = '__all__' 
