import cv2
import numpy as np
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse

app = FastAPI(
    title="API de Cotejo Forense Unificado (Balística y Dactiloscopía)",
    description="Servicio integral de procesamiento de imágenes y generación de informes técnicos forenses.",
    version="2.0.0"
)

# ==========================================
# FUNCIONES AUXILIARES Y PREPROCESAMIENTO
# ==========================================

def bytes_to_cv2(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Formato de imagen inválido o corrupto.")
    return img

def cv2_to_bytes(img: np.ndarray) -> bytes:
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()

def cv2_to_base64(img: np.ndarray) -> str:
    encoded_bytes = cv2_to_bytes(img)
    return base64.b64encode(encoded_bytes).decode('utf-8')

def preprocesar_huella(img: np.ndarray) -> np.ndarray:
    """Mejora el contraste de las crestas dactilares mediante CLAHE y binarización Otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ==========================================
# SECCIÓN 1: BALÍSTICA (ORB + HAMMING)
# ==========================================

@app.post("/comparar-visual", tags=["Balística"])
async def comparar_visual_balistica(
    file_a: UploadFile = File(..., description="Imagen Muestra Evidencia (A)"),
    file_b: UploadFile = File(..., description="Imagen Muestra Patrón (B)")
):
    """Retorna la imagen JPG combinada con el trazado de coincidencias en culotes/vainas."""
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    if des1 is None or des2 is None:
        raise HTTPException(status_code=400, detail="No se identificaron patrones en una de las muestras.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    top_matches = matches[:50]

    res_img = cv2.drawMatches(img_a, kp1, img_b, kp2, top_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return Response(content=cv2_to_bytes(res_img), media_type="image/jpeg")


@app.post("/generar-informe-balistico", tags=["Balística"])
async def generar_informe_balistico(
    file_a: UploadFile = File(..., description="Imagen Muestra Evidencia (A)"),
    file_b: UploadFile = File(..., description="Imagen Muestra Patrón (B)"),
    caso_numero: Optional[str] = Form(None),
    perito: Optional[str] = Form(None)
):
    """Procesa culotes/vainas y retorna un dictamen balístico estructurado en JSON."""
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    if des1 is None or des2 is None:
        raise HTTPException(status_code=400, detail="No se pudieron extraer descriptores balísticos.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    coincidencias_alta_prioridad = [m for m in matches if m.distance < 45]

    total_kp = min(len(kp1), len(kp2))
    porcentaje = round((len(coincidencias_alta_prioridad) / total_kp * 100), 2) if total_kp > 0 else 0

    if porcentaje >= 12.0 or len(coincidencias_alta_prioridad) >= 25:
        veredicto = "CORRESPONDENCIA POSITIVA"
    elif porcentaje >= 6.0:
        veredicto = "INCONCLUYENTE"
    else:
        veredicto = "NO CORRESPONDE"

    res_img = cv2.drawMatches(img_a, kp1, img_b, kp2, matches[:40], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_b64 = cv2_to_base64(res_img)

    return JSONResponse(content={
        "dictamen": veredicto,
        "caso": caso_numero or "N/A",
        "perito": perito or "N/A",
        "metricas": {
            "puntos_evidencia": len(kp1),
            "puntos_patron": len(kp2),
            "coincidencias_alta_prioridad": len(coincidencias_alta_prioridad),
            "porcentaje_similitud": porcentaje
        },
        "anexo_grafico_base64": f"data:image/jpeg;base64,{img_b64}"
    })


# ==========================================
# SECCIÓN 2: DACTILOSCOPÍA (SIFT + FLANN)
# ==========================================

@app.post("/comparar-huellas-visual", tags=["Dactiloscopía"])
async def comparar_huellas_visual(
    file_a: UploadFile = File(..., description="Huella Dactilar Evidencia (A)"),
    file_b: UploadFile = File(..., description="Huella Dactilar Patrón (B)")
):
    """Procesa huellas dactilares y retorna la imagen JPG con el cotejo de minucias."""
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    proc_a = preprocesar_huella(img_a)
    proc_b = preprocesar_huella(img_b)

    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(proc_a, None)
    kp2, des2 = sift.detectAndCompute(proc_b, None)

    if des1 is None or des2 is None:
        raise HTTPException(status_code=400, detail="No se pudieron extraer minucias/puntos de la huella.")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]

    res_img = cv2.drawMatches(proc_a, kp1, proc_b, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return Response(content=cv2_to_bytes(res_img), media_type="image/jpeg")


@app.post("/generar-informe-dactilar", tags=["Dactiloscopía"])
async def generar_informe_dactilar(
    file_a: UploadFile = File(..., description="Huella Dactilar Evidencia (A)"),
    file_b: UploadFile = File(..., description="Huella Dactilar Patrón (B)"),
    caso_numero: Optional[str] = Form(None),
    perito: Optional[str] = Form(None)
):
    """Procesa huellas dactilares y retorna un informe dactiloscópico en JSON."""
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    proc_a = preprocesar_huella(img_a)
    proc_b = preprocesar_huella(img_b)

    sift = cv2.SIFT_create(nfeatures=1500)
    kp1, des1 = sift.detectAndCompute(proc_a, None)
    kp2, des2 = sift.detectAndCompute(proc_b, None)

    if des1 is None or des2 is None:
        raise HTTPException(status_code=400, detail="Puntos de huella insuficientes.")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match in raw_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    num_coincidencias = len(good_matches)

    if num_coincidencias >= 12:
        veredicto = "UNIDAD DE ORIGEN CONFIRMADA (COINCIDENCIA)"
    elif num_coincidencias >= 7:
        veredicto = "COINCIDENCIA PARCIAL / INCONCLUYENTE"
    else:
        veredicto = "NO COINCIDEN"

    res_img = cv2.drawMatches(proc_a, kp1, proc_b, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_b64 = cv2_to_base64(res_img)

    return JSONResponse(content={
        "dictamen": veredicto,
        "caso": caso_numero or "N/A",
        "perito": perito or "N/A",
        "metricas": {
            "puntos_detectados_evidencia": len(kp1),
            "puntos_detectados_patron": len(kp2),
            "puntos_caracteristicos_coincidentes": num_coincidencias
        },
        "anexo_grafico_base64": f"data:image/jpeg;base64,{img_b64}"
    })
