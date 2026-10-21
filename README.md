Face Detection and Recognition

A Python-based face detection and recognition project using OpenCV and a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

Features

Face detection using OpenCV Haar Cascade.

Face data collection through a webcam.

Dataset preprocessing and consolidation.

CNN model training for face recognition.

Real-time face recognition through a webcam.

Technologies Used

Python 3.11

TensorFlow / Keras

OpenCV

NumPy

Pickle

Project Structure

Face Detection/
├── models/
│   ├── final_model.h5
│   ├── class_names.p
│   └── haarcascade_frontalface_default.xml
├── camera_test.py
├── collect_data.py
├── consolidated_data.py
├── train_model.py
├── recognize.py
├── requirements.txt
├── .gitignore
└── README.md

Setup

Install the required dependencies:

py -3.11 -m pip install -r requirements.txt

How to Run

Run the scripts from the project root directory in the following order:

Test the webcam:

py -3.11 camera_test.py

Collect face images:

py -3.11 collect_data.py

Prepare the processed dataset:

py -3.11 consolidated_data.py

Train the CNN model:

py -3.11 train_model.py

Run real-time face recognition:

py -3.11 recognize.py
