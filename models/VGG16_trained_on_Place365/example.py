import numpy as np
import vgg16_place365_keras

def decode(pred, labels, nth=10):
    '''
    decode prediction
    '''
    indice=np.argsort(pred)[-nth:][::-1]
    return {labels[i]:pred[i] for i in indice}
    
# build model
# the zip file can be downloaded via
# https://drive.google.com/file/d/1ir9f6s693e9g9ZnjvB14RnSgBkYYW_96/view?usp=sharing

# or parsed from the original caffemodel at
# https://github.com/CSAILVision/places365
model=vgg16_place365_keras.build('vgg16_places365_weights.zip')
model.summary()

# load label tags
labels=vgg16_place365_keras.load_labels('labels.txt')

# load mean
mean_img=np.load('places365CNN_mean.npy')
mean_img=mean_img[:,16:-16,16:-16] # clip from 256x256 to 224x224
mean_img.shape

# load test image
import cv2
img=cv2.imread('test.jpg')
img=cv2.resize(img,(224,224))

# channel first format, BGR
input_data=(np.transpose(img,[2,0,1]).astype(np.float32)-mean_img)[None,...]
input_data.shape

# prediction
pred=model.predict(input_data)

print(decode(pred[0],labels,20))