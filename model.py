import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input
from keras.layers import concatenate
from keras.utils import np_utils
from keras.optimizers import Adam


class UNet:

    def init(self):
        self.create_unet()

    def train(self, dicom_list, ground_list):
        self.create_unet()
        self.model.fit(dicom_list, ground_list, batch_size=32, epochs=2, verbose=1)

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