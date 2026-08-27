def preprocesar_huella(img: np.ndarray) -> np.ndarray:
    # Redimensionar a un máximo de 600px para bajo consumo de memoria
    max_dim = 600
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
    file_a: UploadFile = File(..., description="Huella Dactilar Evidencia (A)"),
    file_b: UploadFile = File(..., description="Huella Dactilar Patrón (B)")
):
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    proc_a = preprocesar_huella(img_a)
    proc_b = preprocesar_huella(img_b)

    # ORB ligero (limitado a 800 puntos)
    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(proc_a, None)
    kp2, des2 = orb.detectAndCompute(proc_b, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise HTTPException(status_code=400, detail="No se identificaron minucias suficientes.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_tuple in matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

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
    img_a = bytes_to_cv2(await file_a.read())
    img_b = bytes_to_cv2(await file_b.read())

    proc_a = preprocesar_huella(img_a)
    proc_b = preprocesar_huella(img_b)

    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(proc_a, None)
    kp2, des2 = orb.detectAndCompute(proc_b, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise HTTPException(status_code=400, detail="Puntos de huella insuficientes.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_tuple in matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    num_coincidencias = len(good_matches)

    if num_coincidencias >= 12:
        veredicto = "UNIDAD DE ORIGEN CONFIRMADA (COINCIDENCIA)"
    elif num_coincidencias >= 6:
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
