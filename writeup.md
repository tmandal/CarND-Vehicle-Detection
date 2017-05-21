# Vehicle Detection Project

The goals / steps of this project are the following:

* Train a classifier with a labeled training set of given images after normalizing image features.
* Use this classifier to identify vehicles in test images.
* Implement heatmap technique to reject false positives.
* Implement moving average of vehicle identifications from consecutive video frames to smooth out their bounding boxes and follow the vehicles in video.
* Run the above pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and ensure that vehicles are detected in bounding boxes.

[//]: # (Image References)
[image1]: ./output_images/training_samples.png
[image2]: ./output_images/detected_bboxes.png
[image3]: ./output_images/heatmap.png
[image4]: ./output_images/vehicle_detection.png
[video1]: ./project_video.mp4
[video2]: ./project_video_w_vehicles.mp4

## Discussion on choice of implementation

As suggested in project rubric and in the lectures, I initially implemented a linear SVM based classifier after extracting hog features along with color histogram and spatial binning. SVM classifier resulted in excellent accuracy in validation images. After implementing sliding window and heatmap technique to detect vehicles and reject false positives, I found out that there were still way too many false positives detected on the road with strong signals and could not get rid of these without sacrificing positively detected vehicles. Then, I found out on slack channel that many people had successes with applying convolutional neural network for classifying vehicle images. So, I attempted to adopt a CNN based implementation instead of a linear SVM with hog feature extraction.

## Implementation details

### 1. Convolutional neural network as a classifier 

CNNs have been proven to detect and classify objects in images. For this vehicle detection project, a CNN can be employed to identify vehicles with the help of labeled data set from udacity. The data set has about 8000 vehicle images and another 8000 non-vehicle images (road patches, railings etc.) of 64x64 size. A couple of sample training images are shown below.

![alt text][image1]

The CNN is architected with following layers. First, the values from input images are normalized in [-1, 1]. Then first three convolutional layers work as feature extractors. The extracted features pass through a pooling layer to down-size activations before using a fully connected layer for estimating probability of vehicle detection. There are a few specialities in the last layer. A dropout layer is used before the last layer in order to regularize the network during training. The fully connected layer is implemented as a convolutional layer with 1x1x1 output. The reason behind using convolution for the last layer is that larger input images can be fed to this CNN to generate their batch outputs.

Relu activations are used in first 3 convolutional layers for adding non-linearities in the network. Last convolutional layer acting as a fully connected layer uses sigmoid activation for generating binary probabilistic distribution.

| Layer                | Description       | Output Shape        | Param #   
|:--------------------:|:-----------------:|:-------------------:|:------------------:| 
| Input                | 64x64x3 RGB Image |   (None, 64, 64, 3)  | 0                 |
| Normalization        | [-1, 1] Normalization Layer |   (None, 64, 64, 3)   |      0 |        
| Convolution 3x3      | 16 filters, 1x1 stride, same padding    |  (None, 64, 64, 16)   |     448|
| Relu                 |                   |  (None, 64, 64, 16) | 0                  |
| Convolution 3x3      | 32 filters, 1x1 stride, same padding | (None, 64, 64, 32)  |      4640   |  
| Relu                 |                   |  (None, 64, 64, 32) | 0                  |   
| Convolution 3x3      | 64 filters, 1x1 stride, same padding | (None, 64, 64, 64)  |      18496  |  
| Relu                 |                   |  (None, 64, 64, 64) | 0                  |   
| MaxPooling 8x8       | 8x8 stride        |  (None, 8, 8, 64)   |      0             |         
| Dropout              | keep_prob=0.5     |   (None, 8, 8, 64)  |      0             |
| Convolution 8x8      | 1 filter, 1x1 stride, valid padding | (None, 1, 1, 1)   |        4097  |    
| Sigmoid              |                   |  (None, 1, 1, 1)    | 0                  |

This network is implemented is Keras and is trained with binary cross-entropy loss using an adam optimizer for 10 epochs. The data set provided by udacity is split into training and validation sets with 20% of data set aside for validation. After the network is trained, validation and training accuracies are above 99%.

### 2. Vehicle detection in images using CNN classifier

This CNN classifier is a binary predictor and predicts a probability close to 1.0 for vehicles and 0.0 for non-vehicles. A probability threshold of 0.85 is used to separate vehicle and non-vehicle prediction.

The images typically do not have vehicles all over the place. So, a sub-image is selected with a patch of ((0, 400), (1280, 720)) from 1280x720 images as region of interest. This selection avoids all false positives that may have been detected as vehicles in upper half of the images.

Next, this image patch is inferred using a model of above network with trained weights and the output of last CNN layer is looked at for predictions. Because the input image is now 1280x260, the above network will now output 159x31 prediction matrix. Each element of this prediction matrix will now predict whether or not a vehicle is detected in a 64x64 patch in the input image. The location of 64x64 patch in the input image can be found by scaling the location in prediction matrix by the factor compunded by sizes of pooling layers used in the network. Thus, this network can do sliding windows of vehicle detection in a single test of an image. Further, for multi-scaling sliding windows, the input image can be scaled back and forth to detect near and far vehicles in the images by running the network on scaled versions of the images.

### 3. Removing false positive detection and combining multiple nearby detections

After a test image is tested using the classifier, it is observed that a couple of false positives are classified in some images and many many bounding boxes are identified around the actual vehicles of reasonable sizes in the images. This means that there are very strong signals for the real vehicles with few weak signals for non-vehicles. So, the vanilla classifier is working as expected. It's just that weak signals need to be filtered out. 

The filtering is done via heatmap technique as suggested in the project. Basically, each bounding box as detected by the classifier heats up its region in an empty image of one color channel. As the stimulations of detected bounding boxes add up, overlapping regions of bounding boxes separate out strong regions from weak regions in this heatmap. Applying a threshold on this stimulated heatmap helps isolate bounding boxes of vehicles which tend to have strong signals from classifier. Then, strong pixels are collected from this heatmap and grouped them into rectangles that are expected to be wrapping around detected vehicles. Thus, weaker false positive bounding boxes are rejected and stronger bounding boxes are coalesced to generate bounding boxes of detected vehicles.

Below are the images captured from vehicle detection pipeline for a test image.

![alt text][image2]
![alt text][image3]
![alt text][image4]

### 4. Smoothing vehicle detection boxes in video

When the above pipeline (CNN classifier + heatmap filter + grouping) is applied in each frame of project video, there are two things that look odd. First, there are still a few false positives that are observed in one frame or two. Secondly, the bounding boxes around detected vehicles abruptly jump from one frame to next frame. In order to solve both of these issues, a history of detected bounding boxes are kept from last few frames and these bounding boxes are further grouped using cv2.groupRectangle with some group threshold. This thresholding rejcts spurious bounding boxes that might have been detected only in a frame or two. And the rectangle grouping from cv2.groupRectangle using a history of bounding boxes enables smoothness in final bounding boxes.

![alt text][video4]

## Submitted Files

Vehicle_Detection_CNN.ipynb : ipython notebook for vehicle detection pipeline
output_images/result?.jpg : test images annotated with detected vehicles
project_video_w_vehicles.mp4 : project video output with detected vehicles

## Conclusions


