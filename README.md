# 🌱 API de Detección de Vainas Abortadas con YOLOv8

Este proyecto implementa una API con [FastAPI](https://fastapi.tiangolo.com/) para detectar vainas abortadas en imágenes usando un modelo de detección de objetos YOLOv8 entrenado previamente.

## 🚀 Características

- 🔍 Predicción basada en imágenes usando un modelo YOLOv8 (`best.pt`)
- 📸 Retorna la imagen con las detecciones dibujadas
- 🔢 Devuelve el conteo de vainas clasificadas como **"abortadas"**
- 🧩 Integración sencilla con aplicaciones web o móviles (ej. Flutter)
- 🌐 Soporte para CORS para facilitar pruebas desde frontend

## 🗂 Estructura del proyecto

despliegue_modelo/
│
 -  best.pt # Modelo YOLOv8 entrenado
 -  main.py # Código de la API FastAPI
 -  requirements.txt # Dependencias
 - .render.yaml # Configuración para despliegue en Render 

## 🌐 Despliegue en Render
Sube tu repositorio a GitHub.

Conéctalo con Render.

Render detectará automáticamente el archivo .render.yaml y desplegará tu API.

## 🛠 Tecnologías
- Python
- FastAPI
- YOLOv8 (Ultralytics)
- Pillow
- NumPy

