# **Leaf Disease Detection**

This project is designed to identify and classify potential diseases in tomato plant leaves from images. It utilizes a pre-trained InceptionV3 model for image classification and outputs the most likely disease along with its confidence score. Additionally, it provides information on recommended fertilizers and techniques to manage the disease.

**Project Structure:**

* `leaf_detect.py`: Main Python script containing the graphical user interface (GUI) and image classification logic.
* `model_inception_leaf.h5`: The saved TensorFlow model file (trained InceptionV3 model for leaf disease classification).

**Dependencies:**

* TensorFlow
* Keras
* Pillow (PIL Fork)
* tkinter
* numpy

**Running the Project:**

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
