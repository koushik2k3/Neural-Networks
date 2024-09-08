# Flask Image Classification and Drawing App

## Overview
![Output](https://github.com/user-attachments/assets/6f9363ec-c44c-440b-840c-0b2c2f95d72b)

![Output2 ](https://github.com/user-attachments/assets/858e6d14-02db-450e-b4b8-ee44cf02f658)

This project is a Flask web application that enables users to upload images for classification of the MNIST Fashion dataset using a custom neural network. Additionally, users can draw numbers which are processed and displayed. The neural network is built from scratch without using high-level libraries like Keras, relying instead on custom implementations of neural network layers.

## Features

- **Image Classification:** Upload an image to classify it using a custom-built neural network.
- **Number Drawing:** Draw a number and view the generated image.
- **Custom Neural Network:** Implemented using custom classes for convolutional layers, dense layers, activation functions, and more.

## Requirements

- Python 3.x
- Flask
- Flask-WTF
- Pandas
- NumPy
- Pillow
- Matplotlib

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/koushik2k3/Neural-Networks.git
   cd Neural-Networks
   ```

2. **Install the required packages:**

   It is recommended to use a virtual environment. Install the dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content if it does not already exist:

   ```txt
   Flask
   Flask-WTF
   pandas
   numpy
   pillow
   matplotlib
   ```

3. **Prepare the dataset and weights:**

   Ensure you have the following files in the project directory:
   - `weights1.xlsx`
   - `weights2.xlsx`
   - `bias1.xlsx`
   - `bias2.xlsx`
   - `weights.xlsx`

   These files are used to initialize the neural network's weights and biases.

4. **Run the application:**

   Start the Flask development server:

   ```bash
   python app.py
   ```

   By default, the server will run on `http://127.0.0.1:4000/`.
## Drawing Feature

The drawing feature allows users to create images of numbers and view them. Here’s how it works:

1. **Number Drawing:**

   - When a number is entered on the `/draw` route, the application processes it to generate an image.
   - The `draw` function uses a custom neural network to produce an image representing the given number. The network performs a forward pass with the number encoded as a one-hot vector.
   - The resulting output is a 28x28 image, which is saved as a PNG file. This file is stored in the `static/drawings` directory.

2. **Multiple Digits:**

   - For numbers with multiple digits, the `drawmorethan1digit` function processes each digit separately and concatenates the images to form a composite image.
   - Each digit is converted into a 28x28 image, which is then placed side by side to create a single image representing the entire number.

3. **Displaying the Image:**

   - The generated image is saved and can be accessed through the `/static/drawings/<filename>` route.
   - The filename is generated dynamically based on the number drawn.

## Custom Convolutional Neural Network (CNN)

This project demonstrates how to build a Convolutional Neural Network (CNN) from scratch. Here's a breakdown of how the custom CNN works:

1. **Network Architecture:**

   The custom CNN is composed of the following layers:
   - **Convolutional Layer:** Applies convolution operations to extract features from input images.
   - **Activation Layer:** Applies activation functions (e.g., ReLU, sigmoid) to introduce non-linearity.
   - **Reshape Layer:** Reshapes the output of the convolutional layer to fit the input requirements of fully connected layers.
   - **Dense Layer:** Implements fully connected layers to process the features extracted by the convolutional layers.
   - **Softmax Layer:** Provides a probability distribution over classes for classification tasks.

2. **Layer Details:**

   - **Convolutional Layer:** 
     - Custom `Convolutional` class performs 2D convolution operations using kernels and biases.
     - Implemented with methods for forward and backward propagation, weight updating, and saving/loading weights.

   - **Dense Layer:**
     - Custom `Dense` class implements fully connected layers with weight and bias parameters.
     - Includes methods for forward and backward propagation and weight updates.

   - **Activation Functions:**
     - Implementations for various activation functions such as ReLU and sigmoid are included.
     - Custom classes for activation functions handle both forward and backward propagation.

   - **Softmax Layer:**
     - Custom `Softmax` class applies the softmax function to produce class probabilities from the network output.

3. **Training and Prediction:**

   - **Training:**
     - The `train` function performs training using backpropagation and gradient descent.
     - Loss functions such as mean squared error and binary cross-entropy are used for calculating the error and gradients.

   - **Prediction:**
     - The `predict` function applies the trained network to input data and provides output predictions.


## Usage

1. **Upload Image for Classification:**

   Visit the root URL to access the image upload form. Upload an image from the fashion_mnist folder, and it will be classified using the custom neural network. The classification result will be displayed.

2. **Draw a Number:**

   Navigate to the `/draw` route. Enter a number to draw and submit the form. The generated image of the number will be displayed.

## Directory Structure

- `app.py` - The main Flask application file.
- `libv2.py` - Contains custom neural network implementations, including layers and activation functions.
- `libv3.py` - Additional custom functions or utilities (if applicable).
- `static/` - Directory for static files, including uploaded images and generated drawings.
- `templates/` - Directory for HTML templates.
- `weights1.xlsx`, `weights2.xlsx`, `bias1.xlsx`, `bias2.xlsx`, `weights.xlsx` - Files with pre-trained model weights and biases.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

## Acknowledgments

- This project utilizes custom implementations of neural network components.
- Inspired by foundational concepts in neural networks and machine learning.
