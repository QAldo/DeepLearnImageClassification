#!/usr/bin/env python
# coding: utf-8

# # Install dependency

# In[1]:


pip install fastapi tensorflow


# In[ ]:


from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf


# In[ ]:


#Load trained model
model = tf.keras.models.load_model('models', 'furnituremodel.h5')
classes = ['bed', 'chair', 'sofa']
app = FastAPI()


# In[ ]:


#Create API endpoint
async def predict(image:UploadFile = File(...)):
    img = Image.open(image.file)
    
    #Preprocess
    img = img.resize((224,224))
    img = np.array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    
    #Make prediction
    pred = model.predict(img)
    
    predicted_class = classes[np.argmax(prediction)]
    
    return {'class': predicted_class}


# In[ ]:


#Start FastAPI App
if __name == '__main__':
    import uvicorn
    uvicorn.run(app.host='0.0.0.0', port = 8000)


# In[1]:


#Test API
import requests

url = 'http://localhost:8000/predict'
files = {'image': open('testimage.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

