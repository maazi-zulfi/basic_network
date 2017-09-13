import cPickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

model = load_model('/home/maaz/dev/saved_models/basic_network.h5')

with open('/home/maaz/dev/projects/data/cifar-10-batches-py/test_batch','rb') as fo:
    dict_t = cPickle.load(fo)
    t_data = dict_t['data']
    t_label = dict_t['labels']

test_data = np.array(t_data)/255.0
test_data = test_data.reshape((-1,3,32,32))
test_data = np.transpose(test_data,[0,2,3,1])

result = model.predict(test_data)
d_label = np.argmax(result, axis=1)

print classification_report(t_label,d_label)