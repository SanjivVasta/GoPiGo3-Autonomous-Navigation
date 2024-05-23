import numpy as np
import os
import cv2
from sklearn.svm import SVC
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from easygopigo3 import EasyGoPiGo3
import time

# define path to your dataset of images
data_dir = '/home/pi/Project/Data/feature_data'
categories = ['CurveLeft','CurveRight','Left','Right','Straight'] 
img_size = 224  # Image size to resize to

# load the ResNet50 model and remove the top layer
base_model = ResNet50(weights='imagenet', include_top=False)

# extract features from the input image using the pre-trained model and flatten them
def extract_features(img):
    # resize image
    img = cv2.resize(img, (img_size, img_size))
    # extract features from the pretrained model
    feature = base_model.predict(np.expand_dims(img, axis=0))
    feature = feature.flatten()
    return feature

# load and train the SVM classifier and fit it to the extracted features
start_time = time.time()  # Start measuring the time
svm = SVC(kernel="linear", probability=True)
features = []
labels = []
for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)
    feature = extract_features(img)
    features.append(feature)
    label = img_name.split('_')[0]  # assumes the label is the first part of the filename before an underscore
    labels.append(categories.index(label))
svm.fit(features, labels)
end_time = time.time()  # stop measuring the time
elapsed_time = end_time - start_time  # calculate the elapsed time
print(f"Time taken to train the dataset: {elapsed_time:.2f} seconds")  # print the elapsed time to the command line

# initialize the camera object
cam = cv2.VideoCapture(0)

# initialize the GoPiGo3 object
gpg = EasyGoPiGo3()

# initialize the ultrasonic sensor
us = gpg.init_distance_sensor()

# loop over frames from the camera
while True:
    # read a frame from the camera
    ret, frame = cam.read()

    # crop the input frame to the bottom half
    height, width, _ = frame.shape
    crop_img = frame[height // 2:height, :]

    # extract cropped frame features
    feature = extract_features(crop_img)

    # predict the category of the frame using the SVM classifier
    probability = svm.predict_proba([feature])[0]
    label = svm.predict([feature])[0]
    percentage = probability.max() * 100

    if percentage <= 50:
        category = 'Unknown'
        accuracy_text = "" # no percentage shown for Unknown category
    else:
        category = categories[label]
        accuracy_text = f"{percentage:.2f}%" # percentage to 2 decimal places

    # display the recognised category with accuracy percentage
    cv2.putText(frame, category, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, accuracy_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    # print the predicted category and accuracy to the command line
    print("Predicted category:", category)
    print("Accuracy:", accuracy_text)

    # stop the robot if an object is detected within 5cm using Ultrasonic sensor
    if us.read_mm() < 50: # 50mm distance
        gpg.stop() # stop robot
        print("Object Detected!")
        
    # declare speed and radius for turning movements
    speed = 100   
        
    # move the GoPiGo3 based on the predicted category
    if category == 'Straight':
        gpg.forward() # move robot forward
        time.sleep(1) # action time duration
        gpg.stop() # stop robot
    elif category == 'Left':
        gpg.forward() # move robot forward
        time.sleep(0.5) # action time duration
        gpg.left()
        gpg.turn_degrees(-90) # angle of rotation -90 degree turn (anti-clockwise)
        gpg.stop()
    elif category == 'Right':
        gpg.forward() # move robot forward
        time.sleep(0.5) # action time duration
        gpg.right()
        gpg.turn_degrees(90) # angle of rotation 90 degree turn (clockwise)
        gpg.stop() # stop robot
    elif category == 'CurveLeft': 
        for i in range(4): # loop 4 times
             gpg.set_speed(speed)
             gpg.turn_degrees(-22.5) # angle of rotation -22.5 degree turn (anti-clockwise)
             gpg.set_motor_limits(gpg.MOTOR_LEFT, dps=500)
             gpg.set_motor_limits(gpg.MOTOR_RIGHT, dps=500)
             gpg.forward() # move robot forward
             time.sleep(0.5)
             gpg.stop() # stop robot
    elif category == 'CurveRight': 
        for i in range(4):
             gpg.set_speed(speed)
             gpg.turn_degrees(22.5) # angle of rotation 22.5 degree turn (clockwise)
             gpg.set_motor_limits(gpg.MOTOR_LEFT, dps=500)
             gpg.set_motor_limits(gpg.MOTOR_RIGHT, dps=500)
             gpg.forward() # move robot forward
             time.sleep(0.5)
             gpg.stop() # stop robot


    # exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # breaks out of loop
    
# release the camera and close all windows
cam.release() # releases camera
cv2.destroyAllWindows() # closes all windows
