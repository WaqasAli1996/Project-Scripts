import os
import math
import numpy as np
import glob
from tqdm import tqdm
import scipy

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding,Input, merge,ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import sklearn.metrics as metrics

from PIL import image, ImageDraw

import matplotlib.pyplot as plt
plt.ion()

datadir='crowdai'
num_classes = 38
classes = [datadir + '/c_' + str(i) for i in range(num_classes)]

d0= glob.glob(classes[0] + '/*')
i0=scipy.misc.imread(d0[0],mode='RGB')
I = np.array([scipy.misc.imread(d, mode='RGB') for d in tqdm(d0)])
plt.imshow(I[0])

sizes = np.array([i.shape[:2] for i in I])
plt.plot(sizes[:,0],sizes[:,1],'_')

def new_dims(dims):
    '''
    >>>new_dims((256,300))
    ((0,256),(62,318))
    '''
    smaller = min(dims)
    w_min =  int((dims[0]-smaller) / 2)
    w_max =  int(dims[0] - w_min)
    h_min =  int((dims[1] - smaller) / 2)
    h_max =  int(dims[1] - h_min)
    return ((w_min,w_max),(h_min,h_max))
nd=[new_dims(s) for s in sizes]
Ip= np.array([scipy.misc.imresize(ii[nn[0][0]:nn[0][1],nn[1][0]:nn[1][1]],(224,224),'cubic','RGB') for ii,nn in zip(I,nd)])
plt.imshow(Ip[1])

num_data= len(glob.glob(datadir+'/*/*'))
all_imgs= np.zeros((num_data,224,224,3),dtype=np.uint8)
labels = np.zeros((num_data,num_classes),dtype=np.float16)
cnt =0
for i,c in enumerate(classes):
    images = glob.glob(c + '/*')
    for im in tqdm(images):
        img_tmp = scipy.misc.imread(im,mode='RGB')
        s= img_tmp.shape[:2]
        nn=new_dims(s)
        nd = new_dims(s)
        all_imgs[cnt]=scipy.misc.imresize(img_tmp[nn[0][0]:nn[0][1], nn[1][0]:nn[1][1]], (224, 224), 'cubic', 'RGB')
        labels[cnt][i]=1
        cnt +=1

all_imgs=all_imgs.transpose((0,3,1,2))

np.savez_compressed('crop_img_labs.npz',all_imgs,labels)
#np.savez_compressed('crop_labels_onehot.npz',labels)

model = Sequential()
nrows=224
ncols=224

model.add(Convolution2D(8,5,5,input_shape=(3,nrows,ncols),border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

sgd= SGD(0.001)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
h =model.fit(all_imgs,labels,batch_size=32,nb_epoch=1,shuffle=True)
