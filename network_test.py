import cPickle
import numpy as np
from keras.models import load_model
model = load_model('/home/maaz/dev/saved_models/basic_network.h5')

#with open('/home/maaz/dev/projects/data/cifar-10-batches-py/test_batch','rb') as fo:
   # dict = cPickle.load(fo)
