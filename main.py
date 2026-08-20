from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from comparador import procesar_y_comparar

app = FastAPI(title="API Balística Forense")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"mensaje": "Servidor de Balística Forense Activo"}

@app.post("/comparar")
async def comparar_muestras(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    bytes_a = await file_a.read()
    bytes_b = await file_b.read()
    
    resultado = procesar_y_comparar(bytes_a, bytes_b)
    return resultado
