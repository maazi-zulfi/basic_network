
import cPickle
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPool2D, Input, ZeroPadding2D, Dense, Flatten, Activation

with open('/home/maaz/dev/projects/data/cifar-10-batches-py/data_batch_1', 'rb') as fo:
    dict = cPickle.load(fo)

data = dict['data']
label = dict['labels']
train_data = data/255.0
train_data = np.reshape(train_data,(-1,3,32,32))
train_data = np.transpose(train_data,[0,2,3,1])

x1 = Input(batch_shape=(None, 32, 32, 3))

x = ZeroPadding2D(padding=(1,1))(x1)# 34,34,3
x = Conv2D(filters= 64,kernel_size=(3,3),strides=(1,1), activation='relu')(x)#32,32,64
x = MaxPool2D(pool_size=(2, 2))(x)#16,16,64

x = ZeroPadding2D(padding=(1,1))(x)# 18,18,64
x = Conv2D(filters= 128,kernel_size=(3,3),strides=(1,1), activation='relu')(x)#16,16,64
x = MaxPool2D(pool_size=(2, 2))(x)#8,8,64

x = ZeroPadding2D(padding=(1,1))(x)# 10,10,3
x = Conv2D(filters= 256,kernel_size=(3,3),strides=(1,1), activation='relu')(x)#8,8,64
x = MaxPool2D(pool_size=(2, 2))(x)#4,4,64

x = ZeroPadding2D(padding=(1,1))(x)# 6,6,3
x = Conv2D(filters= 256,kernel_size=(3,3),strides=(1,1), activation='relu')(x)#4,4,64
x = MaxPool2D(pool_size=(2, 2))(x)#2,2,256

x = Flatten()(x) #1024
x = Dense(units=1)(x)
x = Activation(activation='sigmoid')(x)

model = Model(x1,x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print model.summary()
print "maaz is stupid"
model.fit(x=train_data,y=label,batch_size=32,epochs=50)
#model = Sequential()
#model.add(Conv2D())