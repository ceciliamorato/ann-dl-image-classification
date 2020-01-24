from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import tensorflow as tf
import numpy as np

SEED = 1234
tf.random.set_seed(SEED)  

cwd = os.getcwd()
print(cwd)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = True

if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=10,
                                        height_shift_range=10,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1./255,
                                        validation_split=0.25)
else:
    train_data_gen = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.25)
    
test_data_gen = ImageDataGenerator(rescale=1./255)

dataset_dir = os.path.join('../input/ann-and-dl-image-classification/', 'Classification_Dataset')
bs = 8

img_h = 256
img_w = 256

num_classes = 20

classList = ['owl', 
             'galaxy', 
             'lightning', 
             'wine-bottle',
             't-shirt',
             'waterfall',
             'sword',
             'school-bus',
             'calculator',
             'sheet-music',
             'airplanes',
             'lightbulb',
             'skyscraper',
             'mountain-bike',
             'fireworks',
             'computer-monitor',
             'bear',
             'grand-piano',
             'kangaroo',
             'laptop']

training_dir = os.path.join(dataset_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               classes=classList,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='training') # set as training data

valid_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               classes=classList,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='validation') # set as validation data

test_dir = os.path.join(dataset_dir, 'test')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             batch_size=bs, 
                                             classes=classList,
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)


train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

valid_dataset = valid_dataset.repeat()

test_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                              output_types=(tf.float32, tf.float32),
                                                output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

test_dataset = test_dataset.repeat()
train_gen.class_indices


class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=num_filters,
                                             kernel_size=(3, 3),
                                             strides=(1, 1), 
                                             padding='same')
        self.activation = tf.keras.layers.ReLU()  # we can specify the activation function directly in Conv2D
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.activation(x)
        x = self.pooling(x)

        return x


depth = 5
start_f = 8
num_classes = 20

class CNNClassifier(tf.keras.Model):
    def __init__(self, depth, start_f, num_classes):
        super(CNNClassifier, self).__init__()
        
        self.feature_extractor = tf.keras.Sequential()
    
        for i in range(depth):
            self.feature_extractor.add(ConvBlock(num_filters=start_f))
            start_f *= 2
            
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Dense(units=num_classes, activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
        
    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
# Create Model instance
model = CNNClassifier(depth=depth,
                      start_f=start_f,
                      num_classes=num_classes)
# Build Model (Required)
model.build(input_shape=(None, img_h, img_w, 3))


# Visualize created model as a table
model.feature_extractor.summary()

# Visualize initialized weights
model.weights

loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


import os
from datetime import datetime

def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')
            
# Early Stopping
callbacks = []

early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(es_callback)

model.fit(x=train_dataset,
          epochs=5,  #### set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_gen))


from PIL import Image
# ....
path = '../input/ann-and-dl-image-classification/Classification_Dataset/'
image_filenames = next(os.walk(path + 'test'))[2]

results = {}
for image_name in image_filenames:
    img = Image.open(path + 'test/' + image_name).convert('RGB')
    img_array = np.array(img) 
    
    img_array = np.resize(img_array, [img_h, img_w, 3])
    img_array = np.expand_dims(img_array, 0) 
   
    img_array = img_array/255;
   

    prediction = model.predict(img_array)
    prediction = np.argmax(prediction, axis=-1)
    
    print(prediction)
    results[image_name] = prediction
    

create_csv(results)