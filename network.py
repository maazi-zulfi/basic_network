
import cPickle
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPool2D, Input, ZeroPadding2D, Dense, Flatten, Activation

data = []
label = []
i = 0
for i in range(5):
    with open('/home/maaz/dev/projects/data/cifar-10-batches-py/data_batch_{}'.format(i),'rb') as fo:
        dict0 = cPickle.load(fo)
    data.append(dict0['data'])
    label.extend(dict0['labels'])

np.concatenate([data])

train_data = np.array(data)/255.0
train_data = train_data.reshape((-1,3,32,32))
train_data = np.transpose(train_data,[0,2,3,1])
label = np.array(label)
label = np.array([label==0,label==1,label==2,label==3,label==4,label==5,label==6,label==7,label==8,label==9],dtype=np.int)
label = np.transpose(label)
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
x = Dense(units=10)(x)
x = Activation(activation='softmax')(x)

model = Model(x1,x)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print model.summary()
# print "maaz is stupid"
model.fit(x=train_data,y=label,batch_size=32,epochs=50)
#model = Sequential()
#model.add(Conv2D())
model.save('/home/maaz/PycharmProjects/saved models/basic_network.h5')