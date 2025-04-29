# analysis/views.py

import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .forms import SentimentForm
from .sentiment_model import SentimentAnalyzer

# Инициализация анализатора (модель обучается или загружается при первом вызове)
analyzer = SentimentAnalyzer()

def index(request):
    result = None
    if request.method == "POST":
        form = SentimentForm(request.POST, request.FILES)
        if form.is_valid():
            text_input = form.cleaned_data.get("text")
            file_input = form.cleaned_data.get("file")

            if text_input:
                sentiment = analyzer.analyze_text(text_input)
                result = {
                    "type": "text",
                    "input": text_input,
                    "sentiment": sentiment
                }
            elif file_input:
                # Сохраняем временно загруженный файл на диск
                fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, "temp_files"))
                filename = fs.save(file_input.name, file_input)
                file_path = fs.path(filename)
                try:
                    full_text = analyzer.read_file(file_path)
                    analysis_result = analyzer.analyze_sentences(full_text)
                    result = {
                        "type": "file",
                        "filename": file_input.name,
                        "analysis": analysis_result
                    }
                except Exception as e:
                    result = {"error": f"Faylni o'qishda xatolik: {str(e)}"}
                finally:
                    # Удаляем временный файл
                    os.remove(file_path)
    else:
        form = SentimentForm()
    
    return render(request, "analysis/index.html", {"form": form, "result": result})
