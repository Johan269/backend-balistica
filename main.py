from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(title="API Balística - Comparación y Puntos Característicos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def redimensionar_imagen_bytes(file_bytes: bytes, max_width: int = 800) -> np.ndarray:
    """Convierte los bytes de la imagen a escala de grises y la redimensiona para no saturar memoria."""
    image = Image.open(io.BytesIO(file_bytes)).convert('L') # Convertir a grises con PIL
    
    # Calcular nueva escala manteniendo la relación de aspecto
    if image.width > max_width:
        aspect_ratio = image.height / image.width
        new_height = int(max_width * aspect_ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    return np.array(image)

def procesar_imagenes(file_a_bytes: bytes, file_b_bytes: bytes):
    try:
        img_a = redimensionar_imagen_bytes(file_a_bytes)
        img_b = redimensionar_imagen_bytes(file_b_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo el formato de imagen: {str(e)}")

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)

    if des_a is None or des_b is None:
        return 0, 0, 0, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    inliers_count = 0
    matches_mask = None

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            matches_mask = mask.ravel().tolist()
            inliers_count = int(np.sum(matches_mask))

    total_keypoints = min(len(kp_a), len(kp_b))
    coincidencia_porcentaje = round((inliers_count / total_keypoints) * 100, 2) if total_keypoints > 0 else 0.0

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    img_matches = cv2.drawMatches(img_a, kp_a, img_b, kp_b, good_matches, None, **draw_params)

    _, buffer = cv2.imencode('.jpg', img_matches)
    img_bytes = buffer.tobytes()

    return coincidencia_porcentaje, len(good_matches), inliers_count, img_bytes


@app.get("/")
def leer_raiz():
    return {"mensaje": "API Balística activa"}


@app.post("/comparar")
async def comparar_muestras(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    content_a = await file_a.read()
    content_b = await file_b.read()

    porcentaje, total_puntos, inliers, _ = procesar_imagenes(content_a, content_b)

    return {
        "coincidencia_porcentaje": porcentaje,
        "puntos_emparejados_brutos": total_puntos,
        "puntos_validos_geometricos": inliers,
        "resultado": "Coincidencia Positiva" if porcentaje >= 45.0 else "Sin Coincidencia"
    }


@app.post("/comparar-visual")
async def comparar_muestras_visual(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    content_a = await file_a.read()
    content_b = await file_b.read()

    _, _, _, img_bytes = procesar_imagenes(content_a, content_b)

    if img_bytes is None:
        raise HTTPException(status_code=400, detail="No se pudieron procesar las imágenes.")

    return Response(content=img_bytes, media_type="image/jpeg")
