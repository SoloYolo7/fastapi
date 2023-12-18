import PIL
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.model_func import class_id_to_label, load_model, transform_image

model = None 
app = FastAPI()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Create class of answer: only class name 
class ImageClass(BaseModel):
    prediction: str

# Load model at startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get('/')
def return_info():
    return 'Hello FastAPI'


@app.post('/classify')
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    adapted_image = transform_image(image)
    pred_index = model(adapted_image.unsqueeze(0)).detach().cpu().numpy().argmax()
    imagenet_class = class_id_to_label(pred_index)
    response = ImageClass(
        prediction=imagenet_class
    )

    return response