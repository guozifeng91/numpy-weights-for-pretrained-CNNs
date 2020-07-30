import keras

# build VGG16 Place365 net based on the weight value and net architecture accessed from https://github.com/CSAILVision/places365

# the weight value is for caffe model and are parsed to npy files, the sequence are kept unchanged (channel first, BGR)

def _build_net():
    model = keras.Sequential()
#     model.add(keras.Input(shape=(3,224,224)))
    
    # group 1 224>112
    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv1_1',
                                  input_shape=(3,224,224)
                                 ))
    model.add(keras.layers.ReLU(name='relu1_1'))
    
    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv1_2'
                                 ))
    model.add(keras.layers.ReLU(name='relu1_2'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='pool1',data_format='channels_first'))
    
    # group 2 112>56
    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv2_1'
                                 ))
    model.add(keras.layers.ReLU(name='relu2_1'))
    
    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv2_2'
                                 ))
    model.add(keras.layers.ReLU(name='relu2_2'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='pool2',data_format='channels_first'))
    
    # group 3 56>28
    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv3_1'
                                 ))
    model.add(keras.layers.ReLU(name='relu3_1'))
    
    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv3_2'
                                 ))
    model.add(keras.layers.ReLU(name='relu3_2'))
    
    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv3_3'
                                 ))
    model.add(keras.layers.ReLU(name='relu3_3'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='pool3',data_format='channels_first'))
    
    # group 4 28>14
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv4_1'
                                 ))
    model.add(keras.layers.ReLU(name='relu4_1'))
    
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv4_2'
                                 ))
    model.add(keras.layers.ReLU(name='relu4_2'))
    
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv4_3'
                                 ))
    model.add(keras.layers.ReLU(name='relu4_3'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='pool4',data_format='channels_first'))
    
    # group 5 14>7
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv5_1'
                                 ))
              
    model.add(keras.layers.ReLU(name='relu5_1'))
    
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv5_2'
                                 ))
    model.add(keras.layers.ReLU(name='relu5_2'))
    
    model.add(keras.layers.Conv2D(filters=512,
                                  kernel_size=[3,3],
                                  strides=(1, 1),
                                  padding="same",
                                  data_format='channels_first',
                                  name='conv5_3'
                                 ))
    model.add(keras.layers.ReLU(name='relu5_3'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='pool5',data_format='channels_first'))
    
    # fc6
    model.add(keras.layers.Flatten(data_format='channels_last', name='f6'))
    # here i assume both caffe flatten the tensor using row major oder
    # see https://en.wikipedia.org/wiki/Row-_and_column-major_order
    #
    # in other word, using the indices of the last axis as the indicator, the flattened sequence is
    #
    # 0,1,2,...,n,0,1,2,...,n
    #
    # NOT
    #
    # 0,0,0,...,0,1,1,1,...,n
    #
    # the test run of this model seems confirmed this assumption
    
    model.add(keras.layers.Dense(4096, name='fc6'))
    model.add(keras.layers.ReLU(name='relu6'))
    model.add(keras.layers.Dropout(0.5, name='drop6'))

    model.add(keras.layers.Dense(4096, name='fc7'))
    model.add(keras.layers.ReLU(name='relu7'))
    model.add(keras.layers.Dropout(0.5, name='drop7'))
    
    model.add(keras.layers.Dense(365, name='fc8a'))
    model.add(keras.layers.Softmax(axis=-1, name='prob'))
    
    return model

import zipfile
import io
import numpy as np

def _to_np_array(bstr):
    bio=io.BytesIO(bstr)
    arr=np.load(bio)
    bio.close()
    return arr

def _set_weights_conv(f, model, name):
    weights=[]
    with f.open('%s-0.npy'%name) as ff:
        # caffe save weights as (output_channel, input_channel, height, width)
        # while keras need (height, width, input_channel, output_channel) regardless the data_format option
        # thus transpose [2,3,1,0] is used
        # future keras may change this?
        weights.append(np.transpose(_to_np_array(ff.read()),[2,3,1,0]))

    with f.open('%s-1.npy'%name) as ff:
        weights.append(_to_np_array(ff.read()))
    model.get_layer(name).set_weights(weights)

def _set_weights_dense(f, model, name):
    weights=[]
    with f.open('%s-0.npy'%name) as ff:
        # caffe save weights as (output_channel, input_channel)
        # while keras need (input_channel, output_channel)
        # thus transpose is used
        weights.append(_to_np_array(ff.read()).T)

    with f.open('%s-1.npy'%name) as ff:
        weights.append(_to_np_array(ff.read()))
    model.get_layer(name).set_weights(weights)
    
def _load_weights(filename, model):
    with zipfile.ZipFile(filename) as f:
        for name in ['conv1_1',
                     'conv1_2',
                     'conv2_1',
                     'conv2_2',
                     'conv3_1',
                     'conv3_2',
                     'conv3_3',
                     'conv4_1',
                     'conv4_2',
                     'conv4_3',
                     'conv5_1',
                     'conv5_2',
                     'conv5_3']:
            _set_weights_conv(f,model,name)
            
        for name in ['fc6','fc7','fc8a']:
            _set_weights_dense(f,model,name)

def build(weightfile=None):
    print('building nets')
    model=_build_net()
    
    print('loading weights')
    if weightfile is not None:
        _load_weights(weightfile,model)
    return model

def load_labels(filename):
    with open(filename) as f:
        labels=f.readlines()
    labels=[l.strip().split('/')[-1] for l in labels]
    
    return labels