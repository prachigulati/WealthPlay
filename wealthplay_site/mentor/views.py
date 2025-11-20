from django.shortcuts import render # <--- ADD THIS LINE
from django.http import JsonResponse
from mentor_engine.mentor import generate_response
import json

def mentor_respond(request):
    data = json.loads(request.body)
    reply = generate_response(data.get("message", ""))
    return JsonResponse({"reply": reply})

def home(request):
    return render(request, "home.html")