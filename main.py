from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from predict import Predict_image
import asyncio
import shutil
import os


app = FastAPI()

template = Jinja2Templates(directory='templates')

@app.get("/", status_code=status.HTTP_200_OK)
async def home(request: Request):
    try:
        return template.TemplateResponse("index.html", {"request": request})
    except:
        raise HTTPException(status_code=404, detail="Page not found")


@app.post("/", status_code=200)
async def classify_image(request: Request, file: UploadFile = File(...)):
    file_path = f"static/uploads/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        model = Predict_image(file_path)

        prediction_class = await asyncio.to_thread(model.predict)

        print(prediction_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"prediction_class": prediction_class, "accuracy": 86.0})
