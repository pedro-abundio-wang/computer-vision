from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class LeNet5:
    
    def build(self, width=32, height=32, depth=1, classes=10):
        
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # initialize the model
        model = Sequential()
        
        # CONV => RELU => POOL layers
        model.add(Conv2D(6, (5, 5), padding="valid", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # CONV => RELU => POOL layers
        model.add(Conv2D(16, (5, 5), padding="valid"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        
        model.add(Dense(84))
        model.add(Activation("relu"))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model