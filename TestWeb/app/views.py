from django.shortcuts import render

# Create your views here.
from app.models import Build
from django.http import HttpResponse


async def ocr_image(request):
    if request.method == 'POST':
        # return render(request, 'post.html')
        file_name = request.POST.get('file')
        ocr_reader = Build()
        result = ocr_reader.excute(file_name, " ")
        print(result)
        return result
