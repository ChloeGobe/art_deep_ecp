from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

def imagenet_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

img_width, img_height = 299, 299
train_data_dir = "data/wikipaintings_train"
validation_data_dir = "data/wikipaintings_val"
nb_train_samples = 66549
nb_validation_samples = 7383
batch_size = 64
epochs = 50

#model = applications.resnet50.ResNet50(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
#model = applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_width, img_height, 3), pooling=None, classes=1000)

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:-1]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
#x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
#x = Dense(128, activation="elu")(x)
#x = Dropout(0.25)(x)
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
checkpoint = ModelCheckpoint("inceptionV3.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# A decommenter pour reprendre une epoch
#model_final.load_weights('inceptionV3.h5')

# Train the model 
#model_final.fit_generator(
#train_generator,
#samples_per_epoch = nb_train_samples,
#epochs = epochs,
#validation_data = validation_generator,
#nb_val_samples = nb_validation_samples,
#callbacks = [checkpoint, early])

model_final.fit_generator(train_generator, steps_per_epoch=1040, epochs=20, verbose=1, callbacks=[checkpoint, early, tbCallBack], validation_data=validation_generator, validation_steps=116)
