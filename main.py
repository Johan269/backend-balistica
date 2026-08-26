from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

app = FastAPI()

@app.post(
    "/comparar-visual",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Retorna la imagen comparativa codificada en JPG."
        }
    }
)
async def comparar_visual(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    # ... tu código de lectura y procesamiento con OpenCV / ORB / SIFT ...
    
    # Ejemplo de codificación final en buffer
    # _, img_encoded = cv2.imencode('.jpg', imagen_resultado)
    # return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    pass
