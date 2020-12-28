# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:59:04 2020

@author: Sainath
"""
from __future__ import division, print_function
import sys
from flask import Flask,url_for,request,render_template
#request to request the image and render to render the page
import os
from werkzeug.utils import secure_filename
#this was imported only for secure_filename
app = Flask(__name__)
from tensorflow.keras.models import load_model
from keras import backend
from tensorflow.keras import backend
from tensorflow.keras.preprocessing import image
import numpy as np
#import tensorflow as tf is not working so we use
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
global graph
graph=tf.get_default_graph()
model=load_model('mri.h5')



@app.route('/',methods=['GET'])
def index():
    return render_template("base.html")
@app.route('/predict',methods=['GET','POST']) 
def upload():
    if request.method == 'POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        #gives path of folder in which app.py is present
        print('current path: ',basepath)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        #we are saving the image we uploaded in the uploads folder ,next we save it
        #u get file path as result by addng base path ,uploads and filename 
        f.save(file_path)
        print('joined path: ',file_path)
        img=image.load_img(file_path,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        with graph.as_default():
            preds=model.predict_classes(x)
            print(preds)
        index=['No','Yes']
        text='Presence of Tumor : '+index[preds[0]]
        return text
        
    
        
if __name__=="__main__":
    app.run(debug=False,threaded=False)
    
