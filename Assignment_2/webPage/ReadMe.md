# FastAPI File Upload + Animal Images Demo

This project is a simple **FastAPI** web app that:

- Serves a homepage (`index.html`) using Jinja2 templates.
- Handles file uploads (`/upload`) and returns file info as JSON.
- Serves static files (animal photos/icons) from the `static/` directory.

---

## 📂 Project Structure


---

## 🚀 Setup & Run

1. **Install dependencies**
   ```bash
   pip install fastapi uvicorn jinja2 python-multipart
   uvicorn main:app --reload
   http://127.0.0.1:8000


   ```

2. **Serving Animal Images**

    ```
    app.mount("/static", StaticFiles(directory="static"), name="static")

    <img src="/static/animals/cat-icon.png" alt="Cat">
<img src="/static/animals/dog.jpg" alt="Dog">

    img.src = `/static/animals/${selectedAnimal}.jpg`;



    ```
3. **Upload the End Point**

    ```
    curl -X POST "http://127.0.0.1:8000/upload" \
     -F "file=@cat.jpg"

    ```
