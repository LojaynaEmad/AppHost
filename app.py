from flask import Flask, request
from flask_restful import Resource, Api
import keras
from PIL import Image, ImageOps 
import numpy as np
from dotenv import load_dotenv
from  pyngrok import ngrok
import os

load_dotenv()


class_names = open("labels.txt", "r").readlines()

model = keras.models.load_model("keras_model.h5", compile=False)

def predict(img):
    img = img.resize((224),224)
    img_np = np.array(img)/255
    img_np = img_np.reshape(1,224,224,3)
    pred = model.predict(img_np , verbose=0)
    return class_names[np.argmax(pred)]

class Predict(Resource):
    def post(self):
        file = request.files["image"]
        img = Image.open(file)
        return{"result": predict(img)}
    
 
app = Flask(__name__)
api =Api(app)

api.add_resource(predict, "/classify")

NGROK_AUTH = os.getenv("NGROK_AUTH")
port = 5000
ngrok.set_auth_token(NGROK_AUTH)
tunnel = ngrok.connect(port, domain="wired-rightly-worm.ngrok-free.app")
print("Public URL:", tunnel.public_url)
app.run(host ="0,0,0,0", post=5000)    