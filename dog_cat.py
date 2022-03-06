#https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/
# Cats vs Dogs Classification (with 98.7% Accuracy) using CNN Keras – Deep Learning Project for Beginners
# Cats vs Dogs classification is a fundamental Deep Learning project for beginners. If you want to start your Deep Learning Journey with Python Keras, you must work on this elementary project.
# 
# In this Keras project, we will discover how to build and train a convolution neural network for classifying images of Cats and Dogs.
# 
# The Asirra (Dogs VS Cats) dataset:
# The Asirra (animal species image recognition for restricting access) dataset was introduced in 2013 for a machine learning competition. The dataset includes 25,000 images with equal numbers of labels for cats and dogs.

# Deep Learning Project for Beginners – Cats and Dogs Classification
# 
# Steps to build Cats vs Dogs classifier:

# 1. Import the libraries:
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os


# 2. Define image properties:
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

# 3. Prepare dataset for training model:
filenames=os.listdir("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-dog-cat\\train")
categories=[]
for f_name in filenames:
    category=f_name.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)
df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})


# 4. Create the neural net model:

from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D,\
#      Dropout,Flatten,Dense,Activation,\
#      BatchNormalization

from tensorflow.keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Conv2D

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(Image_Width,Image_Height,Image_Channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])
  
  
# 5. Analyzing model:
model.summary()

# 6. Define callbacks and learning rate:
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
callbacks = [earlystop,learning_rate_reduction]


# 7. Manage data:
df["category"] = df["category"].replace({0:'cat',1:'dog'})
train_df,validate_df = train_test_split(df,test_size=0.20,
  random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15


# 8. Training and validation data generator:
train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "C://Users//Antonio//Desktop//Master//python//python-dog-cat//train//",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
                                                 
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "C://Users//Antonio//Desktop//Master//python//python-dog-cat//train//", 
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)
# test_datagen = ImageDataGenerator(rotation_range=15,
#                                 rescale=1./255,
#                                 shear_range=0.1,
#                                 zoom_range=0.2,
#                                 horizontal_flip=True,
#                                 width_shift_range=0.1,
#                                 height_shift_range=0.1)
# test_generator = test_datagen.flow_from_dataframe(train_df,
#                                                  "C://Users//Antonio//Desktop//Master//python//python-dog-cat//train//",x_col='filename',y_col='category',
#                                                  target_size=Image_Size,
#                                                  class_mode='categorical',
#                                                  batch_size=batch_size)
  
# #9. Model Training:
epochs=10
# epochs=2
# history = model.fit_generator(
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks)
    
    
# 10. Save the model:
model.save("model1_catsVSdogs_10epoch.h5")
# from tensorflow.keras.models import load_model
# model = load_model("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-dog-cat\\model1_catsVSdogs_10epoch.h5")



# 11. Test data preparation:
# test_filenames = os.listdir("./dogs-vs-cats/test1")
test_filenames = os.listdir("C:\\Users\\Antonio\\Desktop\\Master\\\python\python-dog-cat\\test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
# nb_samples = test_df.shape[0]

test_df['category'] = '1'
nb_samples = test_df.shape[0]
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
test_generator = train_datagen.flow_from_dataframe(test_df,
                                                 "C://Users//Antonio//Desktop//Master//python//python-dog-cat//test1//",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)

# 12. Make categorical prediction:

# predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

# 13. Convert labels to categories:

test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })



# test = np.argmax(predict, axis=-1)
# label_map = dict((v,k) for k,v in train_generator.class_indices.items())
# test= test.replace({ 'dog': 1, 'cat': 0 })


# 14. Visualize the prediction results:

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    # img = load_img("./dogs-vs-cats/test1/"+filename, target_size=Image_Size)
    img = load_img("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-dog-cat\\test1\\"+filename, target_size=Image_Size)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
    plt.show()
    
# 15. Test your model performance on custom data:
results={
    0:'cat',
    1:'dog'
}
from PIL import Image
import numpy as np
# im=Image.open("__image_path_TO_custom_image")
im=Image.open("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-dog-cat\\test1\\5.jpg")
im=im.resize(Image_Size)
im=np.expand_dims(im,axis=0)
im=np.array(im)
im=im/255
#pred=model.predict_classes([im])[0]
pred=model.predict([im])[0][0]
# pred2= np.round_(pred, decimals = 0)
# pred2=np.int16(pred2)[0]
# print(pred2,results[pred2])
if pred-0.5>=0 :
  pred2=1
else :
  pred2=0
print(pred2,results[pred2])



plt.tight_layout()
plt.show()




