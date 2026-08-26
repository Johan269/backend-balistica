import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

app = FastAPI()

@app.post(
    "/comparar-visual",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Retorna la imagen procesada con las coincidencias balísticas."
        }
    }
)
async def comparar_visual(
    file_a: UploadFile = File(...), 
    file_b: UploadFile = File(...)
):
    # 1. Leer los bytes de cada archivo enviado
    bytes_a = await file_a.read()
    bytes_b = await file_b.read()

    # 2. Convertir bytes a arreglos de NumPy e imdecode a imágenes OpenCV
    img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

    # 3. Inicializar detector (ORB/SIFT)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    # 4. Encontrar coincidencias
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Dibujar coincidencias
    resultado = cv2.drawMatches(
        img_a, kp1, img_b, kp2, matches[:50], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 6. Codificar imagen resultante a JPEG
    _, buffer = cv2.imencode(".jpg", resultado)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
