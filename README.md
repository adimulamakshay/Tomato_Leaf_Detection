# **Leaf Disease Detection**

This project is designed to identify and classify potential diseases in tomato plant leaves from images. It utilizes a pre-trained InceptionV3 model for image classification and outputs the most likely disease along with its confidence score. Additionally, it provides information on recommended fertilizers and techniques to manage the disease. This Jupyter Notebook implements a deep learning model for classifying plant diseases from images. It leverages transfer learning with a pre-trained InceptionV3 model and fine-tunes it on the PlantVillage dataset.


**Project Structure:**

* `PlantVillage/`: Directory containing the training and validation image datasets (assuming this directory structure).
    * `train/`: Subdirectory containing images of plant leaves with various diseases (organized into class folders).
    * `val/`: Subdirectory containing validation images for model evaluation.
* `leaf_detection.ipynb`: Jupyter Notebook containing the code for training and evaluating the model.
* `model_inception_leaf.h5`: The saved TensorFlow model file (trained InceptionV3 model for leaf disease classification).
* `leaf_detect.py`: Main Python script containing the graphical user interface (GUI) and image classification logic.

**Dependencies:**

* TensorFlow
* Keras
* Pillow (PIL Fork)
* tkinter
* numpy

**Running the Project:**
1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install Dependencies:**
   ```bash
   !pip install tensorflow keras Pillow matplotlib numpy
   ```

3. **Navigate to the Project Directory:**
   ```python
   import os
   os.chdir('/content/drive/MyDrive/YourProjectPath') # Replace with your actual project directory
   ```

* Make sure to replace `'/content/drive/MyDrive/YourProjectPath'` with the actual path to your project directory in Google Colab.

4. **Run the Jupyter Notebook:**
   * Open the `leaf_disease_detection.ipynb` notebook in Google Colab or a local Jupyter environment.
   * Execute the cells in the notebook sequentially.

**Code Overview:**

* **Import Libraries:** Imports necessary libraries for TensorFlow, Keras, image processing, and data manipulation.
* **Set GPU Memory Allocation:** Configures GPU memory usage for training (optional, adjust as needed).
* **Load Pre-trained Model:** Loads the InceptionV3 model pre-trained on ImageNet, excluding the top classification layers.
* **Freeze Pre-trained Layers:** Freezes the pre-trained layers of InceptionV3 to prevent them from being updated during training.
* **Add Custom Layers:** Adds a new classification layer with the number of outputs matching the number of disease classes in your dataset.
* **Compile Model:** Defines the loss function (categorical cross-entropy), optimizer (Adam), and metrics (accuracy) for model training.
* **Data Augmentation:** Creates ImageDataGenerator objects for training and validation sets, applying random transformations (shear, zoom, horizontal flip) for data augmentation.
* **Load Data:** Loads training and validation image datasets using flow_from_directory with appropriate target sizes and batch sizes.
* **Train Model:** Trains the model on the training dataset with validation on the validation dataset for a specified number of epochs.
* **Visualize Training Results:** Plots training and validation loss and accuracy curves to monitor model performance.
* **Save Model:** Saves the trained model as `model_inception_leaf.h5`.
* **Download Model (Optional):** The code snippet using `from google.colab import files` allows users to download the trained model (`model_inception_leaf.h5`) directly from Colab to their local machine.
![image](https://github.com/user-attachments/assets/df4ed8ef-fe62-4cdf-8310-0a22c3f6c648)
![image](https://github.com/user-attachments/assets/ee1426c5-85a8-419c-bc80-f0bb9af658c7)

1. **Install Dependencies:**
   ```bash
   pip install tensorflow keras Pillow tkinter numpy
   ```

2. **Download the Pre-trained Model:**
   * You'll need to download the pre-trained InceptionV3 model weights and place them in the `model_inception_leaf.h5` file. Pre-trained models can be found online from various sources.

3. **Run the Script:**
   ```bash
   python leaf_detect.py
   ```

**Using the GUI:**

1. Click the "Browse Image" button to select an image file containing a tomato leaf.
2. The selected image will be displayed in the window.
3. Click the "Select Disease" button to open a dialog where you can choose a disease category from a list.
   * Selecting a disease will display a pop-up window with detailed information about control methods in the form of recommended fertilizers and techniques.
4. The "Top Predicted Disease" section will display the model's prediction for the uploaded image, including the most likely disease and its confidence score.

**Note:**

This is a basic implementation and can be further enhanced with features like:
* Training with a larger and more balanced dataset.
* Experimenting with different hyperparameters (learning rate, epochs, etc.) for potentially better performance.
* Implementing a custom image pre-processing pipeline for improved image normalization or noise reduction.
* Integrating the model into a GUI application for user interaction and disease prediction on new images.
* Support for different image classification models.
* Training a custom model on a larger tomato leaf disease image dataset.
* Expanding the disease information database to include more diseases and treatment options.
* Incorporating image pre-processing techniques to improve model accuracy.


