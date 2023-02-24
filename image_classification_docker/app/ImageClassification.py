#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Load Data

# In[ ]:


import tensorflow as tf 


# In[1]:


import matplotlib.pyplot as plt


# In[ ]:


import os


# In[ ]:


import cv2
import imghdr


# In[ ]:


data_dir='Data for test'


# In[ ]:


os.listdir(data_dir)


# In[ ]:


image_exts = ['jpeg', 'jpg', 'bmp', 'png']


# In[ ]:


image_exts[2]


# In[ ]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)


# In[ ]:


img = cv2.imread(os.path.join('Data for test','Bed','90cm Bed with underbed drawer.jpg'))


# In[ ]:


type(img)


# In[ ]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue exists with image {}'.format(image_path))
            # os.remove(image_path)


# In[ ]:


#Load Data


# In[ ]:


tf.data.Dataset


# In[ ]:


import numpy as np


# In[ ]:


# Pre-processing - build data pipeline
data = tf.kera.utils.image_dataset_from_directory('data')


# In[ ]:


# Access
data_iterator = data.as_numpy_iterator()


# In[ ]:


batch = data_iterator.next()


# In[ ]:


# Images as numpy arrays
batch[0].shape


# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# # 2. Preprocess Data
# 

# In[ ]:


# Scale data
#perform the transformation in pipeline
data = data.map(lambda x, y: (x/255,y))


# In[ ]:


scale_iterator = data.as_numpy_iterator()


# In[ ]:


batch = scaled_iterator.next()


# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# In[ ]:


# Split Data
train_size = int(len(data)*.7)
val_size= int(len(data)*.2)+1
test_size = int(len(data)*.1)+1


# In[ ]:


train_size + val_size + test_size


# In[ ]:


train = data.take(train_size)
val = data.skip(train_size).take(val.size)
test = data.skip(train_size+val_size).take(test.size)


# In[ ]:


len(test)


# # 3. Build Deep Learning Model

# In[ ]:


# 1. Build the model


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[ ]:


# Create Sequential class
model = Sequential()


# In[ ]:


#
model.add(Conv2D(16,(3,3), 1, activation='relu', input_shape=(256,256,3)))  # 16 filters, 1 stride, 3 channels
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu')) #256 neurons
model.add(Dense(1, activation='sigmoid')) #single dense layer


# In[ ]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


#2. Train the model


# In[ ]:


logdir = 'logs'


# In[ ]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[ ]:


hist = model.fit(train, epochs=20 validation_data=val, callbacks=[tensorboard_callback])


# In[ ]:


hist.history


# In[ ]:


#3. Refine Plot Performance


# In[ ]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[ ]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# # 4. Evaluate Performance

# In[ ]:


# 4.1 Evaluate


# In[ ]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[ ]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[ ]:


len(test)


# In[ ]:


for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, ythat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[ ]:


print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# In[ ]:


# 4.2 Test


# In[ ]:


import cv2


# In[ ]:


img = cv2.imread('bedtest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[ ]:


yhat = model.predict(np.expand_dims(resize/255, 0))
yhat


# In[ ]:


# Working on data
if yhat > 0.66:
    print(f'Predicted class is bed')
else if yhat > 0.33 AND yhat < 0.65:
    print(f'Predicted class is chair')
else:
    print(f'Predicted class is sofa')


# # 5. Save model

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save(os.path.join('models', 'furnituremodel.h5'))


# In[ ]:


#reload the model
new_model = load_model(os.path.join('models', 'furnituremodel.h5'))


# In[ ]:


yhatnew = new_model.predict(np.expand_dims(resize/255,0))


# In[ ]:




