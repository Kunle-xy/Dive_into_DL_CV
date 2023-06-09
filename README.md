## Deploying trained  DL model using Fast-API

## Instructions

This model uses transfer learning architecture, VGG16, to identify and predict gender and age of a person in an image.

## Intall all dependencies
- `pip install -r requirements.txt`

## Training

click on the link to Google colab
[Link: ](https://colab.research.google.com/drive/1KmYFLo6YWWOVfHowPvayFFCYdSvMCwzZ)









## Fast-API (Deployment)

1. Ensure all dependencies are installed
`pip install fastapi uvicorn aiofiles jinja2 `
- `uvicorn` is a minimal low-level serve/application interface for setting up APIs
- `aiofiles` enables server to work asynchronously with requests

2. cd to the folder that contains script.py and server.py
run on terminal:
-`uvicorn server:app--reload`

3. To fetch predictions:
run on terminal:
- `curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path to your image.png;type=image/png"`


## References:
- Modern Computer Vision with Pytorch ( Ayyadevara & Reddy 2020)
