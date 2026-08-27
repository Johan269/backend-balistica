from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional

# ... tu código de procesamiento ...

    # Al final de la función, retorna directamente usando JSONResponse:
    return JSONResponse(
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
