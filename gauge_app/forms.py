from django import forms

class GaugeForm(forms.Form):
    image = forms.ImageField()
    min_value = forms.FloatField(label="Minimum Gauge Value")
    max_value = forms.FloatField(label="Maximum Gauge Value")
