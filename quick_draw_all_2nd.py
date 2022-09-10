class_names = ['marker','spreadsheet', 'crab', 'matches', 'square', 'crayon',  
               'megaphone','squiggle', 'crocodile',  'mermaid',  'squirrel', 'crown',   'microphone',  'stairs', 
               'cruise%20ship' , 'microwave',   'star', 'cup',  'monkey',  'steak', 'diamond',  'moon', 'stereo',
               'dishwasher',  'mosquito',  'stethoscope','diving%20board' , 'motorbike',  'stitches', 'dog',
               'mountain', 'stop%20sign', 'dolphin', 'mouse', 'strawberry', 'door', 'moustache', 'streetlight', 
               'stove',  'donut',   'mouth', 'dragon', 'mug', 'string%20bean', 'dresser', 'mushroom',  'submarine',
               'drill', 'nail', 'suitcase', 'drums', 'necklace', 'sun', 'duck', 'nose' , 'swan', 'dumbbell',
               'ocean',  'sweater', 'ear', 'octagon', 'swing%20set', 'elbow', 'octopus', 'sword', 'elephant',
               'onion', 'syringe', 'envelope',  'oven', 'table', 'eraser', 'owl', 'teapot', 'eyeglasses',
               'paintbrush',  'teddy-bear', 'eye', 'paint%20can'  ,  'telephone', 'face', 'palm%20tree' , 
               'television', 'fan',  'panda',  'tennis%20racquet', 'feather', 'pants',  'tent', 'fence', 
               'paper%20clip',  'tiger','finger', 'parachute', 'toaster', 'fire%20hydrant',  'parrot', 'toe', 
               'fireplace',  'passport','toilet', 'firetruck',  'peanut', 'toothbrush', 'fish', 'pear', 'tooth',
               'flamingo', 'peas','toothpaste', 'flashlight', 'pencil', 'tornado', 'flip%20flops',  'penguin', 
               'tractor','floor%20lamp',  'piano', 'traffic%20light', 'flower', 'pickup%20truck', 'train',
               'flying%20saucer','picture%20frame', 'tree', 'foot', 'pig', 'triangle', 'fork', 'pillow', 'trombone', 
               'frog','pineapple', 'truck', 'frying%20pan',   'pizza',  'trumpet', 'garden%20hose', 'pliers',
               't-shirt','garden', 'police%20car',  'umbrella',  'giraffe',  'pond',  'underwear', 'goatee',  'pool',
               'van','golf%20club',   'popsicle', 'vase', 'grapes', 'postcard',  'violin', 'grass', 'potato',
               'washing%20machine', 'guitar', 'power%20outlet', 'watermelon', 'hamburger', 'purse', 'waterslide',
               'hammer' , 'rabbit',  'whale', 'hand', 'raccoon', 'wheel', 'harp',  'radio', 'windmill', 'hat',
               'rainbow'   ,   'wine%20bottle','headphones' , 'rain' ,'wine%20glass', 'hedgehog',  'rake',  
               'wristwatch', 'helicopter' ,'remote%20control',   'yoga',  'helmet',  'rhinoceros' , 'zebra',  
               'hexagon',  'rifle',  'zigzag', 'hockey%20puck', 'river'

]



import urllib.request
import os 
import numpy as np
from sklearn.manifold import TSNE


# Random state.
RS = 42



def download_and_load(test_split = 0.2, max_items_per_class = 10000):
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