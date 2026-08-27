def preprocesar_huella(img: np.ndarray) -> np.ndarray:
    # 1. Redimensionar para no saturar la RAM de Render
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Escala de grises + CLAHE + Binarización
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

@app.post("/generar-informe-dactilar", tags=["Dactiloscopía"])
async def generar_informe_dactilar(
    file_a: UploadFile = File(..., description="Huella Dactilar Evidencia (A)"),
    file_b: UploadFile = File(..., description="Huella Dactilar Patrón (B)"),
    caso_numero: Optional[str] = Form(None),
    perito: Optional[str] = Form(None)
):
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    proc_a = preprocesar_huella(img_a)
    proc_b = preprocesar_huella(img_b)

    # Limitar nfeatures a 500 previene colapsos de memoria
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(proc_a, None)
    kp2, des2 = sift.detectAndCompute(proc_b, None)

    # Validar presencia de descriptores suficientes
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise HTTPException(
            status_code=400, 
            detail="No se detectaron suficientes puntos característicos en una o ambas huellas."
        )

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match in raw_matches:
        # Validación obligatoria para evitar el crash (502) por listas de tamaño < 2
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
