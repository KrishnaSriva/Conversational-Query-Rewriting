import os
import uvicorn
from fastapi.staticfiles import StaticFiles
from api.index import app

# Mount frontend files locally
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    print("Serving on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
