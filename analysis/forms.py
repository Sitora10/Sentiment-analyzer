# analysis/forms.py

from django import forms

class SentimentForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={"rows": 3, "placeholder": "Matn kiriting"}), required=False)
    file = forms.FileField(required=False)
