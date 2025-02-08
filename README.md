# 📝 Handwritten Digit Recognition using CNN

## 📌 Project Description
This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits from the **MNIST dataset**. The model is trained to classify digits (0-9) and can predict user-uploaded images.

## 🚀 Features
- Trained on the **MNIST dataset** for high accuracy in digit classification.
- Allows users to upload custom handwritten digit images for prediction.
- Uses **TensorFlow** and **OpenCV** for model implementation and image preprocessing.

## 📂 Dataset Used
- **MNIST Dataset**: 60,000 training images and 10,000 test images of handwritten digits (0-9), each of size **28x28 pixels**.

---

## 🔧 Setup Instructions

### 1. **Clone the Repository**
First, clone this repository to your local machine or into Google Colab:
```bash
git clone https://github.com/ayushfande2003/handwritten-digit-recognition.git
cd handwritten-digit-recognition
2. Install Dependencies
To install required libraries, run:

bash
Copy
Edit
pip install tensorflow numpy matplotlib opencv-python
🏗 Step-by-Step Implementation
Step 1: Import Necessary Libraries
python
Copy
Edit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from google.colab.patches import cv2_imshow
from google.colab import files
Explanation: Import essential libraries for deep learning (TensorFlow), image processing (OpenCV), and visualization (Matplotlib).
Step 2: Load and Preprocess the MNIST Dataset
python
Copy
Edit
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
Explanation: Load the MNIST dataset, normalize the pixel values to be between 0 and 1, and reshape the images for CNN input. Labels are one-hot encoded to represent each digit class.
Output Example:

A sample image would look like this (a 28x28 grayscale image of a digit):

Step 3: Build the CNN Model
python
Copy
Edit
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
Explanation: Build a simple CNN model:
Conv2D layers: Extract features from the images.
MaxPooling2D layers: Reduce dimensionality.
Dense layers: Fully connected layers for classification.
Dropout: Regularization to reduce overfitting.
Step 4: Compile the Model
python
Copy
Edit
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Explanation: Compile the model using the Adam optimizer and categorical cross-entropy loss function for multi-class classification.
Step 5: Train the Model
python
Copy
Edit
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
Explanation: Train the model for 5 epochs using the training data, and validate it using the test data.
Step 6: Evaluate the Model Performance
python
Copy
Edit
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
Explanation: Evaluate the trained model's accuracy on the test dataset. It will print the test accuracy percentage.
Sample Output:

yaml
Copy
Edit
Test Accuracy: 98.72%
Step 7: Save the Trained Model
python
Copy
Edit
model.save('mnist_cnn_model.h5')
Explanation: Save the trained model for future use, so you can load it later without retraining.
Step 8: Load the Saved Model
python
Copy
Edit
loaded_model = tf.keras.models.load_model('mnist_cnn_model.h5')
Explanation: Load the saved model back into memory to make predictions.
Step 9: Function for Predicting User-Uploaded Digit Image
python
Copy
Edit
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = loaded_model.predict(img)
    digit = np.argmax(prediction)

    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()
    return digit
Explanation: This function takes a user-uploaded image, preprocesses it, and uses the trained model to predict the digit.
Step 10: Upload and Predict User-Drawn Digit
python
Copy
Edit
uploaded = files.upload()
for file_name in uploaded.keys():
    print(f"Processing file: {file_name}")
    predicted_digit = predict_digit(file_name)
    print(f"Predicted Digit: {predicted_digit}")
Explanation: Allow users to upload handwritten digit images, predict the digit using the model, and display the result.
🎯 Results & Accuracy
The CNN model achieves an accuracy of ~98% on the MNIST test dataset.

🚀 Deployment
You can deploy this model using:

Streamlit or a simple HTML-JS frontend for real-time digit recognition.
🛠 Future Enhancements
Improve accuracy with more complex CNN architectures.
Deploy the model on a web or mobile application.
Implement real-time digit recognition using OpenCV and a webcam.
📜 Acknowledgments
MNIST Dataset
TensorFlow Documentation
