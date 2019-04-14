import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, Cropping2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib.pyplot import imread
from math import ceil

class DataUtil:

    def __init__(self, data_path='./data/', split_ratio=0.2, correction_factor=0.2):

        self.data_path = data_path
        self.split_ratio = split_ratio
        self.correction_factor = correction_factor

        self.training_samples, self.validation_samples = self.split_data()

    """
    @param: sample
    sample is the csv file line contains the path of the center, left, and right
    image with steering angle

    return: center, left, and right image with there corresponding steering angle 
            with correction
    """
    def process_sample(self, sample):

        images, measurements = [], []
        for i in range(3):
            image_filename = sample[i].split('/')[-1]
            image = imread(self.data_path + 'IMG/' + image_filename)
            images.append(image)

        measurement = float(sample[3])

        #Add correction to the left and right steering angle
        left_measurement = measurement + self.correction_factor
        right_measurement = measurement - self.correction_factor
        
        measurements.append(measurement)
        measurements.append(left_measurement)
        measurements.append(right_measurement)

        return images, measurements

    """
    Import csv file and return the list of the lines 
    """
    def import_data(self):

        with open(self.data_path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            lines = []
            next(reader)
            for line in reader:
                lines.append(line)

        return lines

    """
    Split data into training and validation set with size of validation set equal to self.split_ratio
    """
    def split_data (self):
        training_samples, validation_samples = train_test_split(self.import_data(), test_size=self.split_ratio)

        return training_samples, validation_samples

    """
    Data generator function to load and preprocess the data on fly
    """
    def data_generator (self, samples, batch_size=128):
        n_samples = len(samples)

        while True:
            shuffle(samples)

            for offset in range(0, n_samples, batch_size):
                #fetch batch from samples
                batch_samples = samples[offset: offset+batch_size]

                images, measurements = [], []

                for sample in batch_samples:
                    sample_images, sample_measurements = self.process_sample(sample)

                    images.extend(sample_images)
                    measurements.extend(sample_measurements)

                #Augment the data by flipping the image horizontally
                augmented_images, augmented_measurements = [], []
                for image, measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_images.append(np.fliplr(image))

                    augmented_measurements.append(measurement)
                    augmented_measurements.append(measurement*-1.0)

                X_train, y_train = np.array(augmented_images), np.array(augmented_measurements)

                yield X_train, y_train

    def validation_generator (self, batch_size=128):
        # divide the batch size by six because for one line of input (i.e. csv file line)
        # we create 6 data sample using data augmentation
        batch_size = batch_size // 6

        return self.data_generator(samples=self.validation_samples, batch_size=batch_size), ceil(len(self.validation_samples)/batch_size)

    def training_generator (self, batch_size=128):
        # divide the batch size by six because for one line of input (i.e. csv file line)
        # we create 6 data sample using data augmentation
        batch_size = batch_size // 6

        return self.data_generator(samples=self.training_samples, batch_size=batch_size), ceil(len(self.training_samples)/batch_size)                

class ModelUtil:

    def __init__(self, input_shape=(160, 320, 3)):
        self.input_shape = input_shape

    def getModel(self):
        model = Sequential()
        #Zero mean normalization of data
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=self.input_shape))

        # Crop the image to remove the things that are not relevent like trees etc.
        model.add(Cropping2D(cropping=((70, 20), (0, 0))))

        #Model Architecture
        model.add(Conv2D(24, (5, 5)))
        model.add(BatchNormalization()) #Add batch normalization for regularization to prevent overfitting
        model.add(MaxPooling2D())
        model.add(Activation('relu')) #Add relu actionvation for non-linearity

        model.add(Conv2D(36, (5, 5)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Activation('relu'))

        model.add(Conv2D(48, (5, 5)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(1))

        print('Printing Network Layers.....')
        for layer in model.layers:
            print(layer.output_shape)

        model.compile(loss='mse', optimizer='adam')
        
        return model

class PipeLine:

    def __init__(self):
        self.dataUtil = DataUtil()
        self.modelUtil = ModelUtil()

    def run(self, batch_size, n_epoch):
        training_generator, training_step_size = self.dataUtil.training_generator(batch_size)
        validation_generator, validation_step_size = self.dataUtil.validation_generator(batch_size)
        print(training_step_size, validation_step_size)
        model = self.modelUtil.getModel()
        
        #Fit model using keras fit generator
        model.fit_generator(training_generator, 
                steps_per_epoch=training_step_size, 
                validation_data=validation_generator, 
                validation_steps=validation_step_size, 
                epochs=n_epoch, verbose=1)

        model.save('model.h5')

if __name__ == "__main__":
    pipeline = PipeLine()
    pipeline.run(128, 5)