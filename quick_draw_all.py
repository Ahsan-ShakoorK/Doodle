class_names = ['airplane','book','basket','bat',
               'brain','bread','apple','bridge',
               'bicycle','banana',
               'The%20Eiffel%20Tower','bus','cake', 
               'hourglass','camel',  'house', 
               'ice%20cream','candle','jacket','car','sheep',
               'key','shoe', 'castle',
               'cat','knife','ladder', 'cell%20phone', 'chair', 'leaf',
               'smiley%20face', 'clock', 'cloud',  
               'coffee%20cup'  , 'lion',  'snowman',
               'lollipop','spoon',  
               'crown','star', 'cup', 'monkey','diamond', 'motorbike','dog',
               'mountain', 'mouse', 'strawberry','mushroom','drums','sun', 
               'sword',
               'envelope','table','eyeglasses',
               'eye','face','tiger','finger','fish', 'pear','flower','tree','fork','pizza', 
               'umbrella','popsicle', 'grapes','rabbit','hand',
               'rainbow'   ,'zigzag'
]



import urllib.request
import os 
import numpy as np
from sklearn.manifold import TSNE


# Random state.
RS = 42



def download_and_load(test_split = 0.2, max_items_per_class = 2000):
  root = 'data'
  os.mkdir('data')
  base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for c in class_names:
    path = base+c+'.npy'
    print(path)
    urllib.request.urlretrieve(path, f'{root}/{c}.npy')
  
  #initialize variables 
  x = np.empty([0, 784])
  y = np.empty([0])

  #load each data file 
  for idx, file in enumerate(class_names):
      data = np.load(f'{root}/{file}.npy')
      data = data[0: max_items_per_class, :]
      labels = np.full(data.shape[0], idx)

      x = np.concatenate((x, data), axis=0)
      y = np.append(y, labels)

  data = None
  labels = None

  #randomize the dataset 
  permutation = np.random.permutation(y.shape[0])
  x = x[permutation, :]
  y = y[permutation]

  #reshape and inverse the colors 
  x = 255 - np.reshape(x, (x.shape[0], 28, 28))

  #separate into training and testing 
  test_size  = int(x.shape[0]/100*(test_split*100))

  x_test = x[0:test_size, :]
  y_test = y[0:test_size]

  x_train = x[test_size:x.shape[0], :]
  y_train = y[test_size:y.shape[0]]
  
  return x_train, y_train, x_test, y_test, class_names