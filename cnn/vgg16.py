'''pre trained VGG16 implentation'''
from tensorflow.keras.applications.vgg16 import VGG16
# also import all packages from cnn code file

# creating training and testing set
# image data generator for training - transforming images to help # the model learn better
train_datagen_vgg = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
# validation split for the training process, not the true testing set
        validation_split = 0.2
       )
# creating training set
training_set_vgg = train_datagen_vgg.flow_from_dataframe(
    dataframe = train,
    x_col = "abs_path",
    y_col = "label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset = 'training'
)
# creating validation set
validation_vgg = train_datagen_vgg.flow_from_dataframe(
    dataframe = train,
    x_col = "abs_path",
    y_col = "label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset = 'validation'
)
# Creating VGG16 model
base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = True, 
weights = 'imagenet') # tried both imagenet and the provided weights for celeb dataset

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[27]:


input_shape = (224, 224, 3)
optim_1 = Adam(lr=0.001)
n_classes=105 # 105 celebrities

# play with the following
n_steps = 10
n_val_steps = 10 
n_epochs = 10

# First we'll train the model without Fine-tuning
vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0) # play with fine_tune

# Run when custom weights are used instead of default: 
# vgg_model.load_weights('vgg_face_weights.h5', by_name = True)


# Fitting model
vgg_history = vgg_model.fit_generator(training_set_vgg,
                            epochs=10, steps_per_epoch = 10,
                            validation_data=validation_vgg, validation_steps = 10,       
                            verbose=1)
