from django.db import models

# Create your models here.

class Product(models.Model):
    id    = models.AutoField(primary_key=True)
    name  = models.CharField(max_length = 100) 
    info  = models.CharField(max_length = 100, default = '')
    price = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.name

class Invoice(models.Model):
    id = models.AutoField(primary_key=True)  
    invoice_no = models.CharField(max_length=100)  
    guest_name = models.CharField(max_length=100)  
    arrival_date = models.DateField()  
    departure_date = models.DateField()  
    nights = models.IntegerField()  
    rate = models.DecimalField(max_digits=10, decimal_places=2)  
    invoiced_amount = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Invoice {self.invoice_no} - {self.guest_name}"

class PortInvoice(models.Model):
    id = models.AutoField(primary_key=True)
    supplier_name = models.CharField(max_length=100)
    biller_no = models.CharField(max_length=100)
    supplier_type = models.CharField(max_length=100)
    client_name = models.CharField(max_length=100)
    vessel_name = models.CharField(max_length=100)
    invoice_no = models.CharField(max_length=100)   
    invoice_date = models.DateField()  
    invoice_due_date = models.DateField()  
    principal_amount = models.DecimalField(max_digits=10, decimal_places=2)
    vat_amount = models.DecimalField(max_digits=10, decimal_places=2)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)

    PAYMENT_STATUS_CHOICES = [
        ('paid', 'Paid'),
        ('unpaid', 'Unpaid'),
    ]
    payment_status = models.CharField(
        max_length=6,
        choices=PAYMENT_STATUS_CHOICES,
        default='unpaid'
    )
    CLAIMABLE_CHOICES = [
        ('yes', 'Yes'),
        ('no', 'No'),
    ]
    claimable = models.CharField(
        max_length=3,
        choices=CLAIMABLE_CHOICES,
        default='yes'
    )

    def __str__(self):
        return f"Port Invoice {self.invoice_no} - {self.supplier_name}" 