# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/recovery1.jpg "Recovery Image"
[image2]: ./examples/recovery2.jpg "Recovery Image"
[image3]: ./examples/recovery3.jpg "Recovery Image"
[image4]: ./examples/image.png "Normal Image"
[image5]: ./examples/flipped.png "Flipped Image"
[image6]: ./examples/center_lane.png "Center Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 133-154) 

The model includes RELU layers to introduce nonlinearity (e.g. code line 136), and the data is normalized in the model using a Keras lambda layer (code line 127). 

#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting (e.g. model.py lines 134).

To prevent overfitting I also used data augmentation like flipping images horizontally as well as using left and right images to help the model generalize.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 176).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The recovering from the left and right sides of the road help my vehicle to come in center whenever it get deflected from it.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia architecture since it has been proven to be very successful in self-driving car tasks.

My first step was to use a convolution neural network model similar to the Nvidia architecture I thought this model might be appropriate because it has been proven to be very successful in self-driving car tasks. And a lot of other students are also find it very useful, this architecture is also recommended in the lessons.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added batch normalization layer in each layer of the model instead of dropout layer because dropout is generally less effective at regularizing convolutional layers.

The final step was to run the simulator to see how well the car was driving around track one. Results were pretty good i.e. the car was running fine on the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 127-170) consisted of a convolution neural network of five convolutional layers and four fully connected layers. Following are the layers and the layers sizes

Input (160x320x3)
|   Layer                                                   |  Output Size  |
|-----------------------------------------------------------|---------------|
| Cropping Layer                                            |   (70x320x3)  |
| First Convolutional Layer (with batch normalization)      |   (66x316x24) |
| MaxPooling Layer                                          |   (33x158x24) |
| Relu Activation Layer                                     |   (33x158x24) |
| Second Convolutional layer (with batch normalization)     |   (29x154x36) |
| MaxPooling Layer                                          |   (14x77x36)  |
| Relu Activation Layer                                     |   (14x77x36)  |
| Third Convolutional layer (with batch normalization)      |   (10x73x48)  |
| MaxPooling Layer                                          |   (5x36x48)   |
| Relu Activation Layer                                     |   (5x36x48)   |
| Fourth Convolutional layer (with batch normalization)     |   (3x34x64)   |
| Relu Activation Layer                                     |   (3x34x64)   |
| Fifth Convolutional layer (with batch normalization)      |   (1x32x64)   |
| Relu Activation Layer                                     |   (1x32x64)   |
| First Fully Connected Layer (with batch normalization)    |   100         |
| Relu Activation Layer                                     |   100         |
| Second Fully Connected Layer (with batch normalization)   |   50          |
| Relu Activation Layer                                     |   50          |
| Third Fully Connected Layer (with batch normalization)    |   10          |
| Relu Activation Layer                                     |   10          |
| Fourth Fully Connected Layer (i.e. output layer)          |   1           |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one and half laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come to center of the lane when it get deflected from it. These images show what a recovery looks like starting from left of the lane to the center of the lane:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I also recorded the data by driving the vehicle in the opposite direction of the track.

To augment the data set, I also flipped images and angles thinking that this would help the model to make it unbiased. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 51882 number of data points. I then preprocessed this data by applying the zero mean normalization, and cropping the part of image which is unrelevent for the training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the training and validation loss start increasing after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
