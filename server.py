import os, io
from script import MODEL, ageGenderClassifier
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


model  = MODEL()
app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/file", StaticFiles(directory="file"), name="file")
# templates = Jinja2Templates(directory="templates")

# @app.get("/")
# async def read_item(request: Request):
#     return templates.TemplateResponse("home.html", {"request": request})



# @app.post('/uploaddata/')
# async def upload_file(request: Request, file:UploadFile=File(...)):
#     print(request)
#     content = file.file.read()
#     saved_filepath = f'file/{file.filename}'
#     with open(saved_filepath, 'wb') as f:
#         f.write(content)
#     output = model.predict_from_path(saved_filepath)
#     payload = {'request': request, "filename": file.filename,
#     'output': output}
#     return templates.TemplateResponse("home.html", payload)

@app.post("/predict")
def predict(request: Request, file:UploadFile=File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert('L')
    output = model.predict(image)
    return output