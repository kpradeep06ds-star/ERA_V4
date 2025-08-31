# ASCII Art Generator

A simple web-based tool that converts input text into colorful ASCII art using various fonts. This application is built with FastAPI and Python, providing a fast and modern API for generating stylized text.

## Features

*   **Text to ASCII Art Conversion**: Transform any given text into an ASCII art representation.
*   **Font Selection**: Choose from a wide range of available ASCII art fonts provided by the `pyfiglet` library.
*   **Colorful Gradient**: Generated ASCII art features a vibrant color gradient that transitions from red (top-left) through yellow and green, ending with blue (bottom-right).
*   **Dark Mode Friendly**: The web interface is designed with a dark background for comfortable viewing.

## Technologies Used

*   **Python**: The core programming language.
*   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
*   **Uvicorn**: An ASGI server to run the FastAPI application.
*   **Pyfiglet**: A Python library for rendering text in various ASCII art fonts.
*   **Jinja2**: A modern and designer-friendly templating language for Python, used for rendering HTML.

## Setup

Follow these steps to get the project up and running on your local machine.

### 1. Clone the repository (if applicable)

If your project is in a Git repository, clone it:
```bash
git clone <your-repository-url>
cd <your-project-directory>
```
If you're creating this locally, ensure you are in the project's root directory.

### 2. Create `requirements.txt`

Make sure you have a `requirements.txt` file in your project's root directory with the following content:

```
pyfiglet
fastapi
uvicorn
jinja2
Pillow
numpy
opencv-python-headless
```

### 3. Create `app.py`

Create an `app.py` file in your project's root directory with the FastAPI application logic. (Content provided in previous interactions).

### 4. Create `templates` directory and `index.html`

Create a directory named `templates` in your project's root. Inside the `templates` directory, create an `index.html` file with the web interface content. (Content provided in previous interactions).

### 5. Install dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

## Face Image → ASCII (New)

Turn a **human or animal face photo** into ASCII art, with optional auto-crop around the face and a light edge-enhancement to preserve outlines.

### How to use (UI)


To start the FastAPI application, navigate to your project's root directory in your terminal and run:

```bash
uvicorn app:app --reload
```

*   `app:app` refers to the FastAPI `app` instance inside `app.py`.
*   `--reload` enables auto-reloading of the server when code changes are detected, which is useful for development.

## Usage

Once the server is running, open your web browser and navigate to `http://127.0.0.1:8000`.

*   Enter the text you want to convert into ASCII art in the "Enter text:" field.
*   Select your desired font from the "Choose font:" dropdown menu.
*   Click the "Generate" button to see your colorful ASCII art appear below.
---

### API (endpoints)

* **GET `/face`** – Renders the upload form.
* **POST `/face`** – Generates ASCII from an uploaded image.

  * `multipart/form-data` fields:

    * `file` *(required)* – image file (`.png`, `.jpg`, …)
    * `max_cols` *(int, default: 120)*
    * `charset` *(default: `standard`)* – one of `standard|blocks|sharp`
    * `crop_face` *(checkbox)* – `on` to enable face auto-crop
* Example (HTML response):

  ```bash
  curl -F "file=@face.jpg" \
       -F "max_cols=120" \
       -F "charset=standard" \
       -F "crop_face=on" \
       http://127.0.0.1:8000/face
  ```

### Installation notes

* Extra dependencies are already listed in `requirements.txt`:

  * `Pillow`, `numpy`, `opencv-python-headless`&#x20;
* **Face detection model:** place `haarcascade_frontalface_default.xml` in the project root (same level as `app.py`).
  If the cascade isn’t present or no face is detected, the app **falls back** to converting the whole image.

### Tips

* **Width**: 100–160 gives good detail without scrolling.
* **Charsets**:

  * `standard` → balanced detail
  * `blocks` → bold “poster” look
  * `sharp` → maximum detail (more characters)
* **Performance**: OpenCV “headless” build is used. If you want to skip face detection, uncheck **Auto-detect & crop face**.

---

Enjoy creating vibrant ASCII art!