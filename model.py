import numpy as np

from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ConvLSTM2D
from keras.layers import Input
from keras.layers import concatenate, merge
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import np_utils
from keras.optimizers import Adam


class UNet:

    def init(self):
        self.create_unet()

    def train(self, dicom_list, ground_list):
        self.create_unet_lstm()
        self.model.fit(dicom_list, ground_list, batch_size=32, epochs=2, verbose=1)
        self.model.save('my_model.h5')

    def predict(self, input):
        self.model = load_model('my_model.h5')
        return self.model.predict(input, batch_size=None, verbose=0, steps=None)

    def create_unet(self, input_size = (512,512,1)):
        self.inputs = Input(input_size)

        # convolution and downsampling
        self.conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(self.inputs)
        self.conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(self.conv1)
        
        self.pool1 = MaxPooling2D(pool_size=(2, 2))(self.conv1)
        self.conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(self.pool1)
        self.conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(self.conv2)
        
        self.pool2 = MaxPooling2D(pool_size=(2, 2))(self.conv2)
        self.conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(self.pool2)
        self.conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(self.conv3)
        
        self.pool3 = MaxPooling2D(pool_size=(2, 2))(self.conv3)
        self.conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(self.pool3)
        self.conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(self.conv4)
        self.drop4 = Dropout(0.5)(self.conv4)
        
        self.pool4 = MaxPooling2D(pool_size=(2, 2))(self.drop4)
        self.conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(self.pool4)
        self.conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(self.conv5)
        self.drop5 = Dropout(0.5)(self.conv5)

        # deconv with skip connections (upscaling)
        self.up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(self.drop5))
        self.merge6 = concatenate([self.drop4, self.up6], axis = 3)
        self.conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(self.merge6)
        self.conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(self.conv6)

        self.up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(self.conv6))
        self.merge7 = concatenate([self.conv3, self.up7], axis = 3)
        self.conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(self.merge7)
        self.conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(self.conv7)

        self.up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(self.conv7))
        self.merge8 = concatenate([self.conv2, self.up8], axis = 3)
        self.conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(self.merge8)
        self.conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(self.conv8)

        self.up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(self.conv8))
        self.merge9 = concatenate([self.conv1, self.up9], axis = 3)
        self.conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(self.merge9)
        self.conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(self.conv9)
        self.conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(self.conv9)

        # final convolution to generate output
        self.conv10 = Conv2D(1, 1, activation = 'sigmoid')(self.conv9)

        self.model = Model(input = self.inputs, output = self.conv10)
        self.model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy')
        self.model.summary()

    def create_unet_lstm(self, input_size=(3, 512, 512, 1)):
        self.inputs = Input(input_size, name = 'input')

        # convolution and downsampling
        self.conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same'))(self.inputs)
        self.conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same'))(self.conv1)
        
        self.pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(self.conv1)
        self.conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same'))(self.pool1)
        self.conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same'))(self.conv2)
        
        self.pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(self.conv2)
        self.conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same'))(self.pool2)
        self.conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same'))(self.conv3)
        
        self.pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(self.conv3)

        # bidirectional conv LSTM layer
        self.bidir1 = Bidirectional(ConvLSTM2D(256, 3, border_mode='same', return_sequences=True))(self.pool3)

        
        # deconv with skip connections (upscaling)
        self.up1 = TimeDistributed(UpSampling2D((2,2)))(self.bidir1)
        self.concat1 = concatenate([self.conv3, self.up1])
        self.conv4 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same'))(self.concat1)
        self.conv4 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same'))(self.conv4)
        
        self.up2 = TimeDistributed(UpSampling2D((2,2)))(self.conv4)
        self.concat2 = concatenate([self.conv2, self.up2])
        self.conv5 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same'))(self.concat2)
        self.conv5 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same'))(self.conv5)
        
        self.up3 = TimeDistributed(UpSampling2D((2,2)))(self.conv5)
        self.concat3 = concatenate([self.conv1, self.up3])
        self.conv6 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same'))(self.concat3)

        # bidirectional conv LSTM layer
        self.bidir2 = Bidirectional(ConvLSTM2D(64, 3,  padding='same'))(self.conv6)

        # final convolution to generate output
        self.conv7 = Conv2D(1, 1, activation = 'sigmoid')(self.bidir2)

        self.model = Model(input = self.inputs, output = self.conv7)
        self.model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy')
        self.model.summary()