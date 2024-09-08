
# Flask Image Classification and Drawing App

## Overview

This project is a Flask web application that enables users to upload images for classification using a custom neural network. Additionally, users can draw numbers which are processed and displayed. The neural network is built from scratch without using high-level libraries like Keras, relying instead on custom implementations of neural network layers.

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
   cd your-repo
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

## Usage

1. **Upload Image for Classification:**

   Visit the root URL to access the image upload form. Upload an image, and it will be classified using the custom neural network. The classification result will be displayed.

2. **Draw a Number:**

   Navigate to the `/draw` route. Enter a number to draw and submit the form. The generated image of the number will be displayed.

## Directory Structure

- `app.py` - The main Flask application file.
- `libv2.py` - Contains custom neural network implementations, including layers and activation functions.
- `libv3.py` - Additional custom functions or utilities (if applicable).
- `static/` - Directory for static files, including uploaded images and generated drawings.
- `templates/` - Directory for HTML templates.
- `weights1.xlsx`, `weights2.xlsx`, `bias1.xlsx`, `bias2.xlsx`, `weights.xlsx` - Files with pre-trained model weights and biases.

## Building the Neural Network

This project demonstrates constructing neural networks from scratch, avoiding high-level libraries such as Keras. The implementation includes:

- **Convolutional Layers:** Custom `Convolutional` class for applying convolution operations.
- **Dense Layers:** Custom `Dense` class for fully connected layers.
- **Activation Functions:** Custom activation functions such as ReLU and sigmoid.
- **Training and Prediction:** Functions for training the network and making predictions.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

## Acknowledgments

- This project utilizes custom implementations of neural network components.
- Inspired by foundational concepts in neural networks and machine learning.


