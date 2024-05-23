# GoPiGo3-Autonomous-Navigation
## Project Overview

This repository contains the code and documentation for the development and implementation of an autonomous autopilot system for the GoPiGo3 robot. Utilizing the Raspberry Pi and deep learning techniques, this project aims to emulate the real-world functionality of an autonomous vehicle, following a path and using sensors to enhance road safety.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Experiments and Results](#experiments-and-results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


### Detailed Explanation
The autonomous navigation system leverages several advanced technologies and methodologies to achieve its goals:

- Image Classification: The core of the navigation system involves classifying images captured by the Raspberry Pi Camera. The system uses these images to understand the path the robot must follow.

- ResNet50: A powerful Convolutional Neural Network (CNN) architecture known as ResNet50 is employed for image classification tasks. ResNet50 is chosen for its efficiency and accuracy in deep learning applications, particularly in recognizing complex patterns within images.

- Raspberry Pi Camera: This camera module is used to capture real-time images of the environment. These images are essential for the CNN to process and determine the path ahead.

- TensorFlow: The entire deep learning model, including the ResNet50 architecture, is implemented using TensorFlow, a popular open-source deep learning framework. TensorFlow provides the necessary tools and libraries to build and train the CNN effectively.

### Training the Model
To enable the robot to accurately recognize and follow a path, a dataset must be collected and used to train the model. The training process involves several key steps:

1. Data Collection:

- Capture images of the path the robot will follow using the Raspberry Pi Camera.
- Ensure a variety of images under different lighting conditions and angles to create a robust dataset.

2. Dataset Preparation:

- Label the collected images appropriately to indicate the correct path or direction.
- Split the dataset into training and validation sets to evaluate the model's performance.

3. Model Training:

- Utilize the ResNet50 architecture within TensorFlow to create a CNN.
- Train the CNN using the prepared dataset. During training, the model learns to recognize features of the path by adjusting its internal parameters.
- Implement techniques such as data augmentation and regularization to improve the model's generalization capabilities.

4. Model Evaluation:

- After training, evaluate the model's performance on the validation set to ensure it accurately recognizes the path.
- Fine-tune the model if necessary to achieve optimal performance.

### System Integration
Once the model is trained, it is integrated into the autonomous navigation system:

- Real-Time Processing: The Raspberry Pi Camera continuously captures images as the robot moves. These images are fed into the trained CNN to classify the path in real-time.
- Decision Making: Based on the classification results, the system decides the direction and speed for the GoPiGo3 robot to follow the path accurately.
- Sensor Integration: An ultrasonic sensor is also used to detect obstacles, ensuring the robot navigates safely without collisions.

This project demonstrates the potential of combining hardware components, artificial intelligence techniques, and deep learning models to develop an advanced autopilot system. The system was tested for recognition accuracy, processing speed, and turning efficiency, showing promising potential despite some performance limitations. Future work could focus on refining the robot's performance and enhancing its navigational capabilities to reach destinations using the shortest and most efficient route.


## Features

- Convolutional Neural Networks (CNNs): For image classification and feature extraction.
- Ultrasonic Sensor Integration: For object detection and environment awareness.
- Efficient Processing: Optimized for improved runtime speeds and navigation efficiency.

## Installation

### Prerequisites
- Raspberry Pi with Raspbian installed
- GoPiGo3 robot
- Python 3.x
- Tensorflow
- Keras
- EasyGoPiGo3
- PiCamera
- wpa.supplicant file

## Steps
1. Clone the Repository:
git clone https://github.com/yourusername/GoPiGo3-Autonomous-Navigation.git
cd GoPiGo3-Autonomous-Navigation

2. Install Dependencies:
pip install -r requirements.txt

3. Hardware Setup:
- Assemble the GoPiGo3 robot.
- Attach the Raspberry Pi to the GoPiGo3.
- Connect the ultrasonic sensor and camera module.

## Usage
### Running the Autonomous Navigation System
1. Start the System:
python main.py

2. Monitor Output:
The robot will start navigating and displaying sensor data in the terminal.
Ensure the path and environment are safe for autonomous testing.
Processing delay may occur.

## Configuration
Modify config.json to change parameters such as speed, sensor thresholds, and neural network settings.

## Architecture
### System Components
- Raspberry Pi: Central processing unit for the system.
- GoPiGo3 Robot: Mobile platform for navigation.
- CNN Model: For real-time image classification.
- Ultrasonic Sensor: For detecting obstacles and measuring distances.

## Workflow
1. Image Acquisition: Camera captures images.
2. Image Processing: CNN processes images to classify path.
3. Decision Making: Algorithm decides movements based on processed data.
4. Actuation: Commands sent to GoPiGo3 motors to navigate.

## Experiments and Results
### Performance Metrics
- Recognition Accuracy: Percentage of correctly identified paths.
- Processing Speed: Time taken to process each frame.
- Turning Efficiency: Accuracy and responsiveness of turns.

## Results
During the development and implementation process, I was able to program an autonomous driving autopilot system for the GoPiGo 3 robot using the Raspberry Pi. As shown in the testing above, using feature extraction the pre-trained CNN model can recognise the path ahead in the Raspberry Pi camera view. Once the path is recognised at a high enough accuracy the GoPiGo3 robot is able to move accordingly along the path. The GoPiGo can follow a straight line, make turns to the left and right, follow a curve in the path and stop if the path is no longer recognised. 
 
Additional functionality that I added was the ability for the robot to detect if an object is in the GoPiGo’s path and can stop when it is detected using ultrasonic sensors. Once the object is removed the robot will continue along the path it recognises. As well as the object detection, there is also a live video stream containing the output category label and the recognition accuracy to make the system more user friendly. 

## Future Work
- Performance Improvements: Optimize the CNN model and processing algorithms.
- Enhanced Navigation: Implement advanced pathfinding algorithms.
- Additional Sensors: Integrate more sensors for better environmental awareness.

## Contributing
We welcome contributions to enhance this project. Please follow these steps:

1. Fork the Repository.
2. Create a Branch:
git checkout -b feature-branch
3. Commit Your Changes:
git commit -m 'Add new feature'
4. Push to the Branch:
5. git push origin feature-branch
5. Open a Pull Request.

## License

This project is licensed under the Apache 2.0 License.

## Contact
For any inquiries or further information, please contact:

### Sanjiv Vasta
- Email: sanjivvasta13@gmail.com
- LinkedIn: https://www.linkedin.com/in/sanjiv-vasta-9980251b9/
