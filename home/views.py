from django.shortcuts import render, redirect
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout

from django.contrib.auth.decorators import login_required
import pdfquery
from datetime import datetime, timedelta


from .models import *
# from django.conf import settings
# from django.http import JsonResponse
import os
# import shutil
# import torch
# from auto_gptq import AutoGPTQForCausalLM
# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.vectorstores import Chroma
# from pdf2image import convert_from_path
# from transformers import AutoTokenizer, TextStreamer, pipeline
# import re
# import json

# DEFAULT_SYSTEM_PROMPT = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """.strip()

# def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
#     return f"""
#     [INST] <>
#     {system_prompt}
#     <>
    
#     {prompt} [/INST]
#     """.strip()

# def process_invoice(invoice_name):
#     # Define the path to the vector database
#     db_path = os.path.join(settings.BASE_DIR, 'db_store')

#     # Ensure the directory exists
#     if not os.path.exists(db_path):
#         os.makedirs(db_path)

#     # Clear existing embeddings by removing the database directory
#     if os.path.exists(db_path):
#         shutil.rmtree(db_path)

#     # Define the path to the invoice PDF
#     base_dir = os.path.join(os.getcwd(), "static", "assets", "pdfs")
#     pdf_path = os.path.join(base_dir, invoice_name)
#     print(pdf_path)
#     # Check if the invoice file exists
#     if not os.path.exists(pdf_path):
#         print("PDF path not found")
#         return JsonResponse({'error': 'Invoice not found'}, status=404)

#     # Load and process the specific PDF
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     embeddings = HuggingFaceInstructEmbeddings(
#         model_name="hkunlp/instructor-large", 
#         model_kwargs={"device": DEVICE}
#     )
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#     texts = text_splitter.split_documents(docs)

#     # Create a new vector database
#     db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)

#     # Load and setup the model
#     model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
#     model_basename = "model"
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
#     model = AutoGPTQForCausalLM.from_quantized(
#         model_name_or_path,
#         revision="gptq-4bit-128g-actorder_True",
#         model_basename=model_basename,
#         use_safetensors=True,
#         trust_remote_code=True,
#         inject_fused_attention=False,
#         device=DEVICE,
#         quantize_config=None,
#     )
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#     text_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=1024,
#         temperature=0,
#         top_p=0.95,
#         repetition_penalty=1.15,
#         streamer=streamer,
#     )
#     llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

#     SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
#     template = generate_prompt(
#         """
#         {context}
        
#         Question: {question}
#         """,
#         system_prompt=SYSTEM_PROMPT,
#     )
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={"k": 2}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )

#     # Query to retrieve specific information from the invoice
#     result = qa_chain("please retrieve the Invoice no., invoice total balance amount, arrival date, departure date, Guest Name")
#     list_items = result['result']
#     info_items = re.findall(r'\d+\.\s*(.*?)\s*:\s*(.*)', list_items)
#     info_dict = {item[0]: item[1] for item in info_items}

#     # Retrieve Accommodation / Package line item amount
#     result_1 = qa_chain("please retrieve Accommodation / Package line item amount")
#     accommodation_result = result_1['result']
#     accommodation_match = re.search(r'Accommodation / Package:\s*(\d+\.\d+ SAR)', accommodation_result)
#     if accommodation_match:
#         accommodation_amount = accommodation_match.group(1)
#         info_dict['Accommodation / Package'] = accommodation_amount
#     print("info_dict...")
#     print(info_dict)

#     return JsonResponse(info_dict)


def process_invoice(invoice_name):
  base_dir = os.path.join(os.getcwd(), "static", "assets", "pdfs")
  pdf_path = os.path.join(base_dir, invoice_name)
  
  pdf = pdfquery.PDFQuery(pdf_path)
  pdf.load()
  invoice_number = None
  label_invoice = pdf.pq('LTTextLineHorizontal:contains("Invoice No.")')
  if label_invoice:
    x0, y0, x1, y1 = map(float, [label_invoice.attr('x0'), label_invoice.attr('y0'), label_invoice.attr('x1'), label_invoice.attr('y1')])
    # Define a bounding box to the right of the label to extract the invoice number
    invoice_number_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 100}, {y1}")').text()
    # Split the text to isolate the numeric value
    parts = invoice_number_box.split()
    # Iterate through the parts to find the numeric invoice number
    for part in parts:
      if part.replace('.', '', 1).isdigit():  # Check if the part is numeric (including decimals)
        invoice_number = part
        break
  
  # Search for the label "Arrival"
  arrival_date = None
  label_arrival = pdf.pq('LTTextLineHorizontal:contains("Arrival")')
  if label_arrival:
    
    x0, y0, x1, y1 = map(float, [label_arrival.attr('x0'), label_arrival.attr('y0'), label_arrival.attr('x1'), label_arrival.attr('y1')])

    arrival_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 200}, {y1}")').text()

    parts = arrival_box.split()

    for part in parts:
      if len(part) == 2 and part.isdigit():
        # If a day is found, assume the next two parts are the month and year
        arrival_date_str = " ".join(parts[parts.index(part):parts.index(part)+3])
        break
  
  # Search for the label "Departure"
  departure_date = None
  label_depart = pdf.pq('LTTextLineHorizontal:contains("Departure")')
  if label_depart:
    x0, y0, x1, y1 = map(float, [label_depart.attr('x0'), label_depart.attr('y0'), label_depart.attr('x1'), label_depart.attr('y1')])
    departure_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 200}, {y1}")').text()
    parts = departure_box.split()
    for part in parts:
        if len(part) == 2 and part.isdigit():
            departure_date_str = " ".join(parts[parts.index(part):parts.index(part)+3])
            break

  guest_name = None
  label_guest = pdf.pq('LTTextLineHorizontal:contains("Guest Name")')
  if label_guest:
    x0, y0, x1, y1 = map(float, [label_guest.attr('x0'), label_guest.attr('y0'), label_guest.attr('x1'), label_guest.attr('y1')])
    guest_name_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 100}, {y1}")').text()
    parts = guest_name_box.split(':')
    if len(parts) > 1:
        guest_name = parts[1].strip()
  
  balance = None
  label_bal = pdf.pq('LTTextLineHorizontal:contains("Balance")')
  if label_bal:
    x0, y0, x1, y1 = map(float, [label_bal.attr('x0'), label_bal.attr('y0'), label_bal.attr('x1'), label_bal.attr('y1')])
    balance_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0 - 10}, {float(x1) + 150}, {y1 + 10}")').text()
    parts = balance_box.split()

    for part in parts:
        cleaned_part = part.replace(',', '')
        try:
            if cleaned_part.replace('.', '', 1).isdigit():
                balance = cleaned_part
                break
        except ValueError:
            continue

  # Search for the label "Accommodation / Package"
  rate = None
  label_rate = pdf.pq('LTTextLineHorizontal:contains("Accommodation / Package")')
  if label_rate:
    # Get the bounding box of the label to find the position
    x0, y0, x1, y1 = map(float, [label_rate.attr('x0'), label_rate.attr('y0'), label_rate.attr('x1'), label_rate.attr('y1')])

    # Define a bounding box to the right of the label, extending further to capture the amount
    amount_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 300}, {y1}")').text()

    # The amount might be extracted as part of the full text; let's isolate the numeric value
    parts = amount_box.split()

    # Try to find the numeric part (amount) in the extracted text
    for part in parts:
        if part.replace('.', '', 1).isdigit():
            rate = part
            break
  # Define the date format
  date_format = "%d %b %y"
  arrival_date = datetime.strptime(arrival_date_str, date_format)
  departure_date = datetime.strptime(departure_date_str, date_format)

  # Calculate the number of nights
  number_of_nights = (departure_date - arrival_date).days
  data = {
    'invoice_no': invoice_number,
    'guest_name': guest_name,
    'arrival_date': arrival_date,
    'departure_date': departure_date,
    'nights': number_of_nights,
    'invoiced_amount': balance,
    'rate': rate
  }
  # Save the data to the database
  invoice, created = Invoice.objects.get_or_create(
    invoice_no=invoice_number,
    defaults={
      'guest_name': guest_name,
      'arrival_date': arrival_date,
      'departure_date': departure_date,
      'nights': number_of_nights,
      'invoiced_amount': balance,
      'rate': rate
    }
  )

  if not created:
    invoice.guest_name = guest_name
    invoice.arrival_date = arrival_date
    invoice.departure_date = departure_date
    invoice.invoiced_amount = balance
    invoice.nights = nights
    invoice.rate = rate
    invoice.save()

def process_port_invoice(invoice_name):
  base_dir = os.path.join(os.getcwd(), "static", "assets", "pdfs")
  pdf_path = os.path.join(base_dir, invoice_name)
  
  pdf = pdfquery.PDFQuery(pdf_path)
  pdf.load()

  # Search for the label "Biller Ref.No." and get the text following it
  label_biller = pdf.pq('LTTextLineHorizontal:contains("Biller Ref.No.")')
  biller_no = None
  if label_biller:
    x0, y0, x1, y1 = map(float, [label_biller.attr('x0'), label_biller.attr('y0'), label_biller.attr('x1'), label_biller.attr('y1')])
    biller_text_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 100}, {y1}")').text()
    biller_text = biller_text_box.strip()
    biller_number_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x0 - 200}, {y1 + 5}, {x1}, {y1 + 15}")').text()
    biller_no = biller_number_box.strip()
  
  # Search for the label "Biller Name" and get the text following it
  supplier_name = None
  label_name = pdf.pq('LTTextLineHorizontal:contains("Biller Name")')
  if label_name:
    x0, y0, x1, y1 = map(float, [label_name.attr('x0'), label_name.attr('y0'), label_name.attr('x1'), label_name.attr('y1')])
    supplier_name_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x0 - 200}, {y0 + 5}, {x1}, {y1}")').text()
    supplier_text = supplier_name_box.strip()
    # Remove the "Biller Name" label from the extracted reference number
    supplier_name = supplier_text.replace("Biller Name", "").strip()
  
  # Search for the label "Vessel Name" and get the text following it
  vessel_name = None
  label = pdf.pq('LTTextLineHorizontal:contains("Vessel Name")')
  if label:
    x0, y0, x1, y1 = map(float, [label.attr('x0'), label.attr('y0'), label.attr('x1'), label.attr('y1')])

    vessel_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x0 - 300}, {y0 + 5}, {x1}, {y1}")').text()
    reference_vessel = vessel_box.strip()
    vessel_name = reference_vessel.replace(":Vessel Name", "").strip()
  
  # Search for the label "Invoice No." and get the text following it
  invoice_no = None
  label = pdf.pq('LTTextLineHorizontal:contains("Invoice Ref.No")')
  if label:
    x0, y0, x1, y1 = map(float, [label.attr('x0'), label.attr('y0'), label.attr('x1'), label.attr('y1')])
    reference_number_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x0 - 300}, {y1 + 20}, {x1}, {y1 + 20}")').text()
    invoice_no = reference_number_box.strip()
  
  # Search for the label "Invoice Issue Date" and get the text following it
  invoice_date = None
  label_date = pdf.pq('LTTextLineHorizontal:contains("Invoice Issue Date")')
  if label_date:
    # Get the bounding box of the label to find the position
    x0, y0, x1, y1 = map(float, [label_date.attr('x0'), label_date.attr('y0'), label_date.attr('x1'), label_date.attr('y1')])
    date_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x0 - 200}, {y1 + 5}, {x1}, {y1 + 15}")').text()
    invoice_issue_date_str = date_box.strip()

    # Convert the extracted date string to a datetime object
    invoice_date = datetime.strptime(invoice_issue_date_str, "%d-%m-%Y %I:%M%p")

    # Calculate the due date by adding 15 days
    invoice_due_date = invoice_date.date() + timedelta(days=15)

  # Search for the label "Total Before VAT (SR)" and get the text following it
  principal_amount = None
  label_amount = pdf.pq('LTTextLineHorizontal:contains("Total Before VAT (SR)")')
  if label_amount:
    x0, y0, x1, y1 = map(float, [label_amount.attr('x0'), label_amount.attr('y0'), label_amount.attr('x1'), label_amount.attr('y1')])
    amount_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 500}, {y1}")').text()
    parts = amount_box.split()

    for part in parts:
        cleaned_part = part.replace(',', '')
        try:
            # Try converting to float to identify numeric balance
            if cleaned_part.replace('.', '', 1).isdigit():
                principal_amount = cleaned_part
                break
        except ValueError:
            continue
  
  vat_amount = None
  # Search for the label "Total Before VAT (SR)" and get the text following it
  label_vat = pdf.pq('LTTextLineHorizontal:contains("Total VAT (SR)")')
  if label_vat:
    x0, y0, x1, y1 = map(float, [label_vat.attr('x0'), label_vat.attr('y0'), label_vat.attr('x1'), label_vat.attr('y1')])
    vat_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 600}, {y1}")').text()
    parts = vat_box.split()

    for part in parts:
        cleaned_part = part.replace(',', '')
        try:
            if cleaned_part.replace('.', '', 1).isdigit():
                vat_amount = cleaned_part
                break
        except ValueError:
            continue
  
  total_amount = None
  # Search for the label "Total Invoice Amount Due (SR)" and get the text following it
  label_total = pdf.pq('LTTextLineHorizontal:contains("Total Invoice Amount Due (SR)")')
  if label_total:
    x0, y0, x1, y1 = map(float, [label_total.attr('x0'), label_total.attr('y0'), label_total.attr('x1'), label_total.attr('y1')])
    amount_box = pdf.pq(f'LTTextLineHorizontal:overlaps_bbox("{x1}, {y0}, {float(x1) + 500}, {y1}")').text()
    parts = amount_box.split()

    for part in parts:
        cleaned_part = part.replace(',', '')
        try:
            if cleaned_part.replace('.', '', 1).isdigit():
                total_amount = cleaned_part
                break
        except ValueError:
            continue


  supplier_type = "Port Authority"
  client_name = "Zamil Offshore"
  payment_status = "unpaid"
  claimable = "yes"
  data = {
    'biller_no': biller_no,
    'supplier_name': supplier_name,
    'supplier_type': supplier_type,
    'client_name': client_name,
    'vessel_name': vessel_name,
    'invoice_no': invoice_no,
    'invoice_date': invoice_date,
    'invoice_due_date': invoice_due_date,
    'principal_amount': principal_amount,
    'vat_amount': vat_amount,
    'total_amount': total_amount,
    'payment_status': payment_status,
    'claimable': claimable,
  }
  print(data)
  # Save the data to the database
  invoice, created = PortInvoice.objects.get_or_create(
    invoice_no=invoice_no,
    defaults={
      'biller_no': biller_no,
      'supplier_name': supplier_name,
      'supplier_type': supplier_type,
      'client_name': client_name,
      'vessel_name': vessel_name,
      'invoice_date': invoice_date,
      'invoice_due_date': invoice_due_date,
      'principal_amount': principal_amount,
      'vat_amount': vat_amount,
      'total_amount': total_amount,
      'payment_status': payment_status,
      'claimable': claimable,
    }
  )

  if not created:
    invoice.supplier_name = supplier_name
    invoice.biller_no = biller_no
    invoice.supplier_type = supplier_type
    invoice.client_name = client_name
    invoice.vessel_name = vessel_name
    invoice.invoice_due_date = invoice_due_date
    invoice.invoice_date = invoice_date
    invoice.principal_amount = principal_amount
    invoice.vat_amount = vat_amount
    invoice.total_amount = total_amount
    invoice.payment_status = payment_status
    invoice.claimable = claimable
    invoice.save()
    
  

def index(request):

  context = {
    'segment'  : 'index',
    #'products' : Product.objects.all()
  }
  return render(request, "pages/index.html", context)

def tables(request):
  context = {
    'segment': 'tables'
  }
  return render(request, "pages/dynamic-tables.html", context)

def invoices(request):
  context = {
    'segment': 'invoices'
  }
  invoice_name = "37665.pdf"
  # process_invoice(invoice_name)
  return render(request, "pages/invoices.html", context)

def port_invoices(request):
  context = {
    'segment': 'port-invoices'
  }
  invoice_name = "extracted_text_from_pdf.pdf"
  # process_port_invoice(invoice_name)
  return render(request, "pages/portinvoices.html", context)
