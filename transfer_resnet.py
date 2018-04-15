# Inspired from https://gist.github.com/Prakashvanapalli/221571cf036179ed5343432eca16e72b
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# Values taken from RASTA paper
def imagenet_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

# Check that img_width and img_height correspond to the value expected to be used with the network used
img_width, img_height = 224, 224
train_data_dir = "data/wikipaintings_train"
validation_data_dir = "data/wikipaintings_val"
nb_train_samples = 66549
nb_validation_samples = 7383
batch_size = 64
epochs = 50

# Change this line to use another network
model = applications.resnet50.ResNet50(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')

# Freeze the layers which you don't want to train
for layer in model.layers[:-1*(len(model.layers)//5)]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(25, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
preprocessing_function=imagenet_preprocess_input,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.,
width_shift_range = 0.,
height_shift_range=0.,
rotation_range=0)

test_datagen = ImageDataGenerator(
preprocessing_function=imagenet_preprocess_input,
horizontal_flip = False,
fill_mode = "nearest",
zoom_range = 0.,
width_shift_range = 0.,
height_shift_range=0.,
rotation_range=0)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("resnet50.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Save data using Tensorboard
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# If you want to load saved weights
#model_final.load_weights('resnet50.h5')

model_final.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint, early, tbCallBack], validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size)
