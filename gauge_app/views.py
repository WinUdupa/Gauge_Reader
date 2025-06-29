# gauge_app/views.py
from django.shortcuts import render
from django.conf import settings
from .forms import GaugeForm
from .predict import predict_gauge
import os, base64, uuid
from django.core.files.base import ContentFile

def predict_gauge_view(request):
    
    context = {} 

    if request.method == 'POST':
        form = GaugeForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            min_value = form.cleaned_data['min_value']
            max_value = form.cleaned_data['max_value']

            filename = f"{uuid.uuid4().hex}.jpg"
            img_path = os.path.join(settings.MEDIA_ROOT, filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

            with open(img_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            with open(img_path, 'rb') as f:
                uploaded_image_base64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

            value, output_image_base64 = predict_gauge(img_path, min_value, max_value)
            if value is None:
                context['error'] = output_image_base64
            else:
                context.update({
                    'value': value,
                    'uploaded_image_base64': uploaded_image_base64,
                    'output_image_base64': output_image_base64
                })
    else:
        form = GaugeForm()

    context['form'] = form
    return render(request, 'gauge_app/form.html', context)
