import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def procesar_y_comparar(bytes_a: bytes, bytes_b: bytes):
    nparr_a = np.frombuffer(bytes_a, np.uint8)
    nparr_b = np.frombuffer(bytes_b, np.uint8)
    
    imgA = cv2.imdecode(nparr_a, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imdecode(nparr_b, cv2.IMREAD_GRAYSCALE)

    if imgA is None or imgB is None:
        return {"coincidencia_porcentaje": 0.0, "puntos_emparejados": 0}

    imgA = cv2.resize(imgA, (512, 512))
    imgB = cv2.resize(imgB, (512, 512))

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(imgA, None)
    kp2, des2 = orb.detectAndCompute(imgB, None)

    if des1 is None or des2 is None:
        return {"coincidencia_porcentaje": 0.0, "puntos_emparejados": 0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is not None:
            imgB = cv2.warpPerspective(imgB, M, (512, 512))

    score, _ = ssim(imgA, imgB, full=True)
    porcentaje = max(0.0, float(score * 100))

    return {
        "coincidencia_porcentaje": round(porcentaje, 2),
        "puntos_emparejados": len(matches)
    }
