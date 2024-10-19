Tumor Classification Model
This project aims to classify breast tumor images as either malignant or benign using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset contains mammogram images of tumors, with labels indicating whether the tumor is benign (0) or malignant (1).

Project Overview
Dataset: The dataset consists of mammogram images with associated labels. The train_data.csv file contains the training data with image filenames and corresponding labels, while test_data.csv contains the filenames for the test images.
Image Preprocessing: All images are resized to 224x224 pixels and normalized for use in the neural network.
Model: The model is a CNN with multiple convolutional and pooling layers, followed by fully connected layers for classification.
Output: The model predicts whether the tumor is malignant or benign based on the input image.
Requirements
Python 3.x
TensorFlow 2.x
Keras
Pandas
NumPy
Model Overview
Preprocessing:

Images are loaded and resized to 224x224 pixels.
Pixel values are normalized by dividing by 255.
Training images are labeled based on the folder they belong to (0 for benign, 1 for malignant).
CNN Model:

The model consists of two Conv2D layers followed by MaxPooling2D layers for down-sampling.
A Flatten layer converts the 2D output into a 1D vector.
Dense layers are used for classification, with a sigmoid activation function for binary classification.
Training:

The model is trained using the binary_crossentropy loss function, with the Adam optimizer and accuracy as the evaluation metric.
Training is performed for 10 epochs with an 80-20 training-validation split.
Usage
1- Preprocess the data: The script reads the CSV files and preprocesses the images by resizing them and normalizing pixel values.

2- Train the model: The model is trained on the preprocessed images for 10 epochs. Adjust the number of epochs as necessary.
3- Make predictions: After training, the model makes predictions on the test set. The results are stored in a CSV file, final_submission.csv.
Results
The model predicts tumor classifications on the test data, and the results are stored in the final_submission.csv file, containing the filename and predicted label (0 for benign and 1 for malignant).
