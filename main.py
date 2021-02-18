import os, zipfile, cv2, csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

def prepare_image(img):
  
  # convert the color from BGR to RGB then convert to PIL array
  cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  # resize the array (image) then PIL image
  img_array = image.img_to_array(cvt_image)
  
  return keras.applications.mobilenet.preprocess_input(img_array)

def get_label(labels):
  bigger = max(labels)
  
  for i in range(4):
    if labels[i] == bigger:
      if i == 0:
        return 'upright'
      elif i == 1:
        return 'rotated_left'
      elif i == 2:
        return 'rotated_right' 
      elif i == 3:
        return 'upside_down'

batch_size = 32
num_classes = 4
epochs = 100
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

for item in os.listdir("./"): # loop through items in dir
    if item.endswith(".zip"): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall("./") # extract file to dir
        zip_ref.close() # close file

train_truth = []
y=[]
with open("train.truth.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
      if i > 0:
        train_truth.append(line[0])
        if line[1] == 'upright':
          
          y.append(0)
        elif line[1] == 'rotated_left':
          
          y.append(1)
        elif line[1] == 'rotated_right':
          
          y.append(2)
        elif line[1] == 'upside_down':
          
          y.append(3)

X = []

path_train_images = './train'
for filename in train_truth:
  img = cv2.imread(os.path.join(path_train_images, filename))
  
  X.append(prepare_image(img))

X = np.array(X)
y = np.array(y)

x_train,  x_test, y_train, y_test = train_test_split(X, y, test_size=0.24)

y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),input_shape=(64, 64, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

test = []
images_test = []
for filename in glob.glob(os.path.join("test", "*.jpg")):
  img = cv2.imread(os.path.join(path_train_images, filename))
  
  test.append(prepare_image(img))
  images_test.append(filename)
  
test = np.array(test)
test = test.astype('float32')

test /= 255

with open('test.preds.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dn", "label"])
    predictions = model.predict(test)
    i=0
    for i in range(len(images_test)):
      
      writer.writerow([images_test[i], get_label(predictions[i])])



