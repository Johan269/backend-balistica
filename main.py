import io
import gc
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")
    return img

def cv2_to_bytes(img: np.ndarray) -> bytes:
    is_success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not is_success:
        raise ValueError("Error al codificar la imagen de salida.")
    return buffer.tobytes()

def preprocesar_huella(img: np.ndarray) -> np.ndarray:
    # Escalar obligatoriamente a un máximo de 500px para no agotar la RAM de Render
    max_dim = 500
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

@app.post("/comparar-huellas-visual", tags=["Dactiloscopía"])
async def comparar_huellas_visual(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):
    try:
        # Lectura segura de bytes
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        img_a = bytes_to_cv2(content_a)
        img_b = bytes_to_cv2(content_b)

        proc_a = preprocesar_huella(img_a)
        proc_b = preprocesar_huella(img_b)

        # ORB ligero de bajo consumo (max 500 features)
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(proc_a, None)
        kp2, des2 = orb.detectAndCompute(proc_b, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return JSONResponse(
                status_code=400,
                content={"detail": "No se detectaron suficientes puntos característicos para comparar."}
            )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        res_img = cv2.drawMatches(
            proc_a, kp1, proc_b, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        img_bytes = cv2_to_bytes(res_img)

        # Forzar recolección de basura
        del img_a, img_b, proc_a, proc_b, des1, des2, res_img
        gc.collect()

        return Response(content=img_bytes, media_type="image/jpeg")

    except Exception as e:
        # En lugar de colapsar con 502, FastAPI capturará el error y responderá un JSON claro
        return JSONResponse(
            status_code=500,
            content={"error": "Fallo interno durante el procesamiento", "detalle": str(e)}
        )
