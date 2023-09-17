from flask import Flask,render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
#define flask app
app=Flask(__name__)
model_path='model/apple1.h5'
model=load_model(model_path)
def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_img)
    test_image=test_image/225
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return  result
@app.route('/',methods=['GET'])
def index():
    return  render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        #get the file host
        f=request.files['file']
        #save the file to upload folder
#         basepath=os.path.dirname(os.path.realpath('__file__'))
#         file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
#         f.save(file_path)
        result=model_predict(f,model)
        categories=['Healthy','Multiple disease','Rust','Scab']
        # process for human understanding Expand Down
    
    
  
        pre_class=result.argmax()
        output=categories[pre_class]
        return output
    return None
