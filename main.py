import cv2
import numpy as np
import base64
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse
from typing import Optional

app = FastAPI(
    title="API de Cotejo Balístico Automático",
    description="Servicio de procesamiento de imágenes y generación de informes balísticos.",
    version="1.0.0"
)

# ---------------------------------------------------------
# ENDPOINT 1: Visualización directa de la imagen (JPG)
# ---------------------------------------------------------
@app.post(
    "/comparar-visual",
    summary="Comparación Visual Directa",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Retorna la imagen procesada en formato JPG."
        }
    }
)
async def comparar_visual(
    file_a: UploadFile = File(..., description="Imagen Muestra Evidencia (A)"),
    file_b: UploadFile = File(..., description="Imagen Muestra Patrón (B)")
):
    bytes_a = await file_a.read()
    bytes_b = await file_b.read()

    img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 45]
    else:
        good_matches = []

    resultado = cv2.drawMatches(
        img_a, kp1, img_b, kp2, good_matches[:40], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    _, buffer = cv2.imencode(".jpg", resultado)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------
# ENDPOINT 2: Generación de informe estructurado (JSON)
# ---------------------------------------------------------
@app.post(
    "/generar-informe-balistico",
    summary="Generar Informe Técnico de Cotejo",
    description="Procesa dos muestras de culotes/vainas y retorna un informe técnico estructurado en JSON."
)
async def generar_informe(
    file_a: UploadFile = File(..., description="Imagen Muestra Evidencia (A)"),
    file_b: UploadFile = File(..., description="Imagen Muestra Patrón (B)"),
    caso_numero: Optional[str] = Form("EXP-2026-001"),
    perito: Optional[str] = Form("Sistema Automatizado OpenCV")
):
    bytes_a = await file_a.read()
    bytes_b = await file_b.read()

    img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)

    num_kp1 = len(kp1) if kp1 is not None else 0
    num_kp2 = len(kp2) if kp2 is not None else 0

    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 45]
        total_matches = len(matches)
        total_good = len(good_matches)
    else:
        matches = []
        good_matches = []
        total_matches = 0
        total_good = 0

    puntos_min = max(min(num_kp1, num_kp2), 1)
    porcentaje_similitud = round((total_good / puntos_min) * 100 * 3.5, 2)
    porcentaje_similitud = min(porcentaje_similitud, 98.50)

    if total_good >= 20:
        veredicto = "CORRESPONDENCIA POSITIVA"
        conclusion = "Las muestras presentan concordancia topográfica sustancial en sus marcas de percusión y/o cierre."
    elif 8 <= total_good < 20:
        veredicto = "INCONCLUYENTE CON COINCIDENCIAS PARCIALES"
        conclusion = "Se observan huellas coincidentes, pero no alcanzan el umbral mínimo para una identificación categórica."
    else:
        veredicto = "NO CORRESPONDE (NEGATIVO)"
        conclusion = "No se observan huellas ni microrayadas concordantes suficientes entre las dos muestras."

    img_matches = cv2.drawMatches(
        img_a, kp1, img_b, kp2, good_matches[:40], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    _, buffer = cv2.imencode(".jpg", img_matches)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(
        status_code=200,
        headers={"Cache-Control": "no-transform"},
        content={
            "encabezado": {
                "institucion": "Laboratorio de Análisis Balístico Automático",
                "fecha_analisis": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "numero_expediente": caso_numero,
                "analista_responsable": perito
            },
            "datos_muestras": {
                "muestra_a_evidencia": {
                    "nombre_archivo": file_a.filename,
                    "dimensiones": f"{img_a.shape[1]}x{img_a.shape[0]} px",
                    "puntos_clave_detectados": num_kp1
                },
                "muestra_b_patron": {
                    "nombre_archivo": file_b.filename,
                    "dimensiones": f"{img_b.shape[1]}x{img_b.shape[0]} px",
                    "puntos_clave_detectados": num_kp2
                }
            },
            "resultado_cuantitativo": {
                "coincidencias_totales": total_matches,
                "coincidencias_alta_prioridad": total_good,
                "indice_similitud_estimado": f"{porcentaje_similitud}%"
            },
            "dictamen_tecnico": {
                "veredicto_preliminar": veredicto,
                "observaciones": conclusion
            },
            "anexo_grafico": {
                "formato": "image/jpeg (Base64)",
                "imagen_comparativa_b64": f"data:image/jpeg;base64,{img_b64}"
            }
        }
    )
