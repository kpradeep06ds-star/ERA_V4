
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pyfiglet
from typing import Tuple
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

# Comment these two lines out if you choose the "no OpenCV" variant below
import cv2

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pyfiglet

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def generate_colorful_ascii_art(ascii_text: str) -> str:
    # Keep your existing implementation here
    return "<br>".join(ascii_text.splitlines())

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    available_fonts = pyfiglet.FigletFont.getFonts()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "colorful_ascii_art": "",
            "text_input": "",
            "selected_font": "standard",
            "available_fonts": available_fonts,
            "mode": "text",
        },
    )

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, text: str = Form(...), font: str = Form("standard")):
    try:
        raw_ascii = pyfiglet.figlet_format(text, font=font)
    except pyfiglet.FontNotFound:
        raw_ascii = pyfiglet.figlet_format(text)  # fallback to default
        font = "standard"
    colorful_html = generate_colorful_ascii_art(raw_ascii)
    available_fonts = pyfiglet.FigletFont.getFonts()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "colorful_ascii_art": colorful_html,
            "text_input": text,
            "selected_font": font,
            "available_fonts": available_fonts,
            "mode": "text",
        },
    )


def generate_colorful_ascii_art(ascii_text: str) -> str:
    lines = ascii_text.split('\n')
    max_rows = len(lines)
    max_cols = max(len(line) for line in lines) if lines else 1

    colorful_html_lines = []
    for r_idx, line in enumerate(lines):
        colorful_line = []
        for c_idx, char in enumerate(line):
            if char == ' ':
                colorful_line.append('&nbsp;')
            else:
                # Calculate a normalized 'C' value from 0 to 1 based on position
                c_normalized = (r_idx + c_idx) / (max_rows + max_cols + 1e-6)

                # Map c_normalized to Hue (0-360) for a gradient from Red to Blue
                # 0 (Red) -> 60 (Yellow) -> 120 (Green) -> 240 (Blue)
                # We'll use a range from 0 to 240 for a smooth transition through these colors.
                hue = round(c_normalized * 240) # Max hue of 240 (blue)

                # Keep saturation and lightness consistent and high enough for dark background
                saturation = 90 # High saturation for vibrant colors
                lightness = 60  # Mid-range lightness for good visibility

                color = f"hsl({hue}, {saturation}%, {lightness}%)"
                colorful_line.append(f'<span style="color: {color};">{char}</span>')
        colorful_html_lines.append("".join(colorful_line))
    return "<br>".join(colorful_html_lines)


CHARSETS = {
    # Dense to sparse: tweak as you like
    "standard": "@%#*+=-:. ",
    "blocks": "█▓▒░  ",
    "sharp":  "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
}
DEFAULT_CHARSET = "standard"

def _aspect_correct_size(w: int, h: int, max_cols: int = 120) -> Tuple[int, int]:
    """
    Convert image pixel aspect to character cell aspect.
    Terminal characters are taller than wide (roughly 2:1 height:width).
    We compensate by scaling height down.
    """
    if w == 0 or h == 0:
        return (1, 1)
    scale = max(1, w / max_cols)
    new_w = int(round(w / scale))
    new_h = int(round((h / scale) * 0.5))  # 0.5 ~ char aspect fudge
    return max(new_w,1), max(new_h,1)

def _pil_to_gray_arr(img: Image.Image) -> np.ndarray:
    # Convert to grayscale with mild contrast normalization
    img = ImageOps.exif_transpose(img.convert("L"))  # correct orientation, grayscale
    arr = np.asarray(img, dtype=np.uint8)
    # Stretch contrast (simple)
    arr = np.interp(arr, (arr.min(), arr.max()+1e-6), (0, 255)).astype(np.uint8)
    return arr

def _image_to_ascii_lines(gray: np.ndarray, charset: str = DEFAULT_CHARSET, max_cols: int = 120) -> str:
    h, w = gray.shape[:2]
    new_w, new_h = _aspect_correct_size(w, h, max_cols)
    # resize with PIL for decent quality
    img_small = Image.fromarray(gray).resize((new_w, new_h), Image.BICUBIC)
    small = np.asarray(img_small, dtype=np.uint8)

    chars = CHARSETS.get(charset, CHARSETS[DEFAULT_CHARSET])
    n = len(chars) - 1
    lines = []
    # optional edge boost: combine intensity with edges
    # sobel edges
    sobelx = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)) if 'cv2' in globals() else 0
    sobely = np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)) if 'cv2' in globals() else 0
    edges = sobelx + sobely if isinstance(sobelx, np.ndarray) else 0

    # normalize signals
    small_f = small.astype(np.float32) / 255.0
    if isinstance(edges, np.ndarray):
        e = edges / (edges.max() + 1e-6)
        # blend edges to preserve outlines; tweak 0.25–0.5
        small_f = np.clip(0.85*small_f + 0.35*e, 0, 1)

    idx = (small_f * n).astype(np.int32)
    for r in idx:
        line = "".join(chars[i] for i in r)
        lines.append(line)
    return "\n".join(lines)

def _detect_face_region(gray3c: np.ndarray) -> Tuple[int,int,int,int]:
    """
    Return (x,y,w,h) for the largest detected face; fallback to full image if none.
    """
    try:
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray3c, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        if len(faces) == 0:
            return (0, 0, gray3c.shape[1], gray3c.shape[0])
        # pick largest
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        return (x,y,w,h)
    except Exception:
        # On any error, just return full image
        return (0, 0, gray3c.shape[1], gray3c.shape[0])

def image_bytes_to_ascii(
    data: bytes,
    charset: str = DEFAULT_CHARSET,
    max_cols: int = 120,
    crop_face: bool = True
) -> str:
    img = Image.open(BytesIO(data)).convert("RGB")
    gray_full = _pil_to_gray_arr(img)

    if crop_face and 'cv2' in globals():
        # Use OpenCV for face detection on a 3-channel gray-ish input
        gray3c = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        x,y,w,h = _detect_face_region(gray3c)
        gray_cropped = gray_full[y:y+h, x:x+w]
    else:
        gray_cropped = gray_full

    ascii_text = _image_to_ascii_lines(gray_cropped, charset=charset, max_cols=max_cols)
    return ascii_text


@app.get("/face", response_class=HTMLResponse)
async def face_form(request: Request):
    available_fonts = pyfiglet.FigletFont.getFonts()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "colorful_ascii_art": "",
            "text_input": "",
            "selected_font": "standard",
            "available_fonts": available_fonts,
            "mode": "face",
            "max_cols": 120,
            "charset": "standard",
            "crop_face_checked": True,
        },
    )

@app.post("/face", response_class=HTMLResponse)
async def face_ascii(
    request: Request,
    file: UploadFile = File(...),
    max_cols: int = Form(120),
    charset: str = Form("standard"),
    crop_face: str = Form("on"),
):
    data = await file.read()
    ascii_text = image_bytes_to_ascii(
        data,
        charset=charset,
        max_cols=max_cols,
        crop_face=(crop_face == "on"),
    )
    colorful_html = generate_colorful_ascii_art(ascii_text)

    available_fonts = pyfiglet.FigletFont.getFonts()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "colorful_ascii_art": colorful_html,
            "text_input": "",
            "selected_font": "standard",
            "available_fonts": available_fonts,
            "mode": "face",
            "max_cols": max_cols,
            "charset": charset,
            "crop_face_checked": (crop_face == "on"),
        },
    )