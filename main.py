from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import base64

app = FastAPI()

# Permite llamadas desde cualquier frontend (Flutter, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo solo una vez al arrancar
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)
    plot_img = results[0].plot()

    # Conteo de vainas abortadas
    class_names = results[0].names
    boxes = results[0].boxes
    count_abortadas = sum(class_names[int(cls)] == 'vaina_abortada' for cls in boxes.cls)

    # Codificar imagen de salida como base64
    result_img = Image.fromarray(plot_img)
    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse({
        "vainas_abortadas": count_abortadas,
        "image": img_base64
    })
