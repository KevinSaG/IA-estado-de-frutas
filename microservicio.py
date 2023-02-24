from flask import Flask, jsonify, request
import requests
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
import cv2 as cv2
import numpy as np

##Cargar imagen: Escalado y Redimensionar
def cargarImg(path):
    dsize=(128,128)
    ruta=path
    src = cv2.imread(ruta)[:,:,::-1] #BGR (cv2) a RGB (matplotlib )
    imgcv=cv2.resize(src, dsize)
    imgcv=imgcv*(1/255)
    return imgcv

def etiquetar(index): 
    if(index==0):
        et='fresh_apple'
    if(index==1):
        et='fresh_banana'
    if(index==2):
        et='fresh_bitter_gourd'
    if(index==3):
        et='fresh_capsicum'
    if(index==4):
        et='fresh_orange'
    if(index==5):
        et='fresh_tomato'
    if(index==6):
        et='stale_apple'
    if(index==7):
        et='stale_banana'
    if(index==8):
        et='stale_bitter_gourd'
    if(index==9):
        et='stale_capsicum'
    if(index==10):
        et='stale_orange'
    if(index==11):
        et='stale_tomato'  
    return et

def etiquetarTF(index): 
    if(index==0):
        et='fresh_grape'
    if(index==1):
        et='fresh_mango'
    if(index==2):
        et='fresh_peach'
    if(index==3):
        et='stale_grape'
    if(index==4):
        et='stale_mango'
    if(index==5):
        et='stale_peach' 
    return et

#def predecirF(img, modelo, et):
    #pred=modelo.predict(np.array([img]))
    #id_pred=pred[0].tolist().index(max(pred[0].tolist()))
    #return et(id_pred) 

def predecirF(img, modelo, et):
    pred_dc={}
    pred=modelo.predict(np.array([img]))
    pred_sf=np.sort(pred[0])[::-1]
    for i in range(3):
        pred_dc[et(pred[0].tolist().index(pred_sf[i]))] = str(pred_sf[i]) 
    return pred_dc



app = Flask(__name__)

@app.route("/api/ML/", methods=['GET', 'POST'])
def frutas():
    #data = request.get_data()
    #data = request.form.data()
    data = request.get_json()
    print(data['Path_img'])
   
    # ...model.h5
    modelFrutas=keras.models.load_model('frutas.h5')
    modelFrutasTF=keras.models.load_model('frutasTF.h5')

    img=cargarImg(data['Path_img'])
    #res=predecirF(img, modelFrutas, etiquetar)
    res=predecirF(img, modelFrutasTF, etiquetarTF)
       
    return jsonify(
             pred=res
         )




if __name__ == "__main__":
    app.run()
