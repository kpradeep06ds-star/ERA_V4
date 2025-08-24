#main.py
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

# Serve files under ./static at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# (Optional) build a dynamic list of images
IMAGE_DIR = Path("static/animals")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    images = [p.name for ext in ("*.jpg", "*.png", "*.jpeg") for p in IMAGE_DIR.glob(ext)]
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Mimic your Flask checks
    if file is None:
        return JSONResponse({"error": "No file part"})
    if (file.filename or "").strip() == "":
        return JSONResponse({"error": "No selected file"})

    contents = await file.read()  # bytes
    file_info = {
        "name": file.filename,
        "size": len(contents),
        "type": file.content_type or "application/octet-stream",
    }
    # If you'll need to re-read later, uncomment:
    # await file.seek(0)
    return file_info
