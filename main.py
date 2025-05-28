from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import base64

app = FastAPI()

# Permitir peticiones desde cualquier origen (útil para Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo una vez al iniciar
model = YOLO("best.pt")  # Asegúrate de que este archivo esté en la raíz o en la ruta correcta

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer imagen recibida
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)

    # Hacer predicción
    results = model(img_array)
    plot_img = results[0].plot()

    # Conteo de vainas con clase "abortada"
    class_names = results[0].names
    boxes = results[0].boxes
    count_abortadas = sum(class_names[int(cls)] == "abortada" for cls in boxes.cls)

    # Codificar imagen con resultados en base64
    result_img = Image.fromarray(plot_img)
    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse({
        "vainas_abortadas": count_abortadas,
        "image": img_base64
    })

