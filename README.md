# Handwritten Digit Recognition using CNN 📝🤖
## 📌 Project Description
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained to classify digits (0-9) and can predict user-uploaded images.

Key Features:
- Uses the MNIST dataset for training and testing.
- Implements a CNN for high accuracy in digit classification.
- Allows users to upload custom handwritten digit images for prediction.
- Uses TensorFlow and OpenCV for model implementation and image preprocessing.
## 📂 Dataset Used
- **MNIST Dataset**: Contains 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size 28x28 pixels.
## 🔧 Installation & Setup

### Step 1: Install Dependencies
```bash
pip install tensorflow numpy matplotlib opencv-python google-colab
```
### Step 2: Clone or Download the Project
```bash
git clone https://github.com/ayushfande2003/handwritten-digit-recognition.git

cd handwritten-digit-recognition
```
### Step 3: Run the Project in Google Colab or Jupyter Notebook
Make sure you have the necessary dependencies and run the notebook in Google Colab for easy execution.
## Step-by-Step Implementation
### Step 1: Import Necessary Libraries
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from google.colab.patches import cv2_imshow
from google.colab import files
```
Explanation: This step imports the required libraries for deep learning (TensorFlow), image processing (OpenCV), visualization (Matplotlib), and file handling.
### Step 2: Load and Preprocess the MNIST Dataset
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Reshape data to fit into the CNN model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```
Explanation: The dataset is loaded, normalized, and reshaped for the CNN input. We also convert the labels to a one-hot encoding format, which is crucial for classification tasks. The one-hot encoded format will represent each digit with a 10-element vector (e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for digit 3).
### Step 3: Build the CNN Model
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
Explanation: The CNN architecture consists of convolutional layers (for feature extraction), max-pooling layers (to reduce the size of the feature maps), and fully connected layers (for classification). The model ends with a softmax output layer to predict the probability of each digit (0-9).
### Step 4: Compile the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
Explanation: We compile the model using the Adam optimizer and categorical cross-entropy loss function, which is appropriate for multi-class classification problems.
### Step 5: Train the Model
```python
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```
Explanation: We train the model for 5 epochs. We also pass the test dataset for validation, so the model's performance is evaluated after every epoch.
### Step 6: Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
```
Explanation: After training, we evaluate the model’s accuracy on the test dataset to see how well it performs on unseen data.
### Step 7: Save the Trained Model
```python
model.save('mnist_cnn_model.h5')
```
Explanation: We save the trained model so it can be reused later for making predictions without needing to retrain it.
### Step 8: Function to Predict User-Uploaded Image
```python
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()
    return digit
```
Explanation: This function allows users to upload their own handwritten digit images. The image is processed, resized, and normalized before being passed to the model for prediction.
### Step 9: Upload and Predict
```python
uploaded = files.upload()
for file_name in uploaded.keys():
    print(f"Processing file: {file_name}")
    predicted_digit = predict_digit(file_name)
    print(f"Predicted Digit: {predicted_digit}")
```
Explanation: This step allows users to upload their handwritten digit image and get predictions. The ``` files.upload() ``` function will prompt you to upload a file, and the model will predict the digit.
## 🎯 Results & Accuracy
The CNN model achieves an accuracy of **~98%** on the MNIST test dataset.

![Model Test Accuracy](images/test_accuracy.png)

