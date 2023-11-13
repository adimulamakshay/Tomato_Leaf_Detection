import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the leaf disease detection model
model = tf.keras.models.load_model('C:\\Users\\aksha\\OneDrive\Desktop\\Leaf_Detection\\model_inception_leaf.h5')

# Define a list of labels for leaf diseases
labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Create a dictionary of disease-related information
disease_info = {
    'Tomato___Bacterial_spot': 'Fertilizer: Use copper-based fungicides.\nTechnique: Remove and destroy infected plants.',
    'Tomato___Early_blight': 'Fertilizer: Use balanced fertilizers with higher potassium.\nTechnique: Remove infected leaves.',
    'Tomato___Late_blight': 'Fertilizer: Use fungicides containing copper or chlorothalonil.\nTechnique: Remove infected leaves.',
    'Tomato___Leaf_Mold': 'Fertilizer: Use organic fertilizers rich in phosphorus and potassium.\nTechnique: Provide good ventilation.',
    'Tomato___Septoria_leaf_spot': 'Fertilizer: Use balanced fertilizers with higher potassium.\nTechnique: Remove infected leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Fertilizer: Use nitrogen-rich fertilizers.\nTechnique: Apply miticides.',
    'Tomato___Target_Spot': 'Fertilizer: Use balanced fertilizers with higher potassium.\nTechnique: Remove infected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Fertilizer: Use potassium-rich fertilizers.\nTechnique: Remove infected plants and control whiteflies.',
    'Tomato___Tomato_mosaic_virus': 'Fertilizer: Use well-balanced fertilizers.\nTechnique: Remove infected plants and control aphids.',
    'Tomato___healthy': 'No specific fertilizer recommendations for healthy plants.'
}

# Create the main window
window = tk.Tk()
window.title("Leaf Disease Detection")
window.geometry("600x600")
window.configure(bg="#b2dfdb")  # Set background color to a teal shade

# Create a function to browse and display an image
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        classify(file_path)

# Create a label to display the selected image
image_label = tk.Label(window, bg="#ffffff")
image_label.pack(pady=20)

# Create a "Browse" button with a different color
browse_button = tk.Button(window, text="Browse Image", command=browse_image, bg="#00897b", fg="white")
browse_button.pack()

# ... (previous code)

# Function to show detailed disease information in a dialog box
def show_disease_info(disease_choice):
    disease_info_text = disease_info[disease_choice]
    messagebox.showinfo("Disease Information", disease_info_text)

# ... (remaining code)


# Create a function to select a disease and display information about it in a detailed dialog box
def select_disease():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    disease_choice = simpledialog.askstring("Leaf Disease Detection", "Select a disease from the list:", initialvalue="Tomato___Bacterial_spot", parent=root)
    if disease_choice:
        if disease_choice in disease_info:
            show_disease_info(disease_choice)
        else:
            messagebox.showinfo("Leaf Disease Detection", "No information available for the selected disease.")
    root.destroy()

# Create a "Select Disease" button with a different color
select_disease_button = tk.Button(window, text="Select Disease", command=select_disease, bg="#00897b", fg="white")
select_disease_button.pack()

# Create a label for predictions
predictions_label = tk.Label(window, text="Top Predicted Disease:", bg="#b2dfdb", fg="#004d40", font=("Arial", 14))
predictions_label.pack()

# Create a text widget to display predictions and percentages
predictions_text = tk.Text(window, height=4, width=40, bg="#ffffff")
predictions_text.pack()

# Define a function to classify the image
def classify(image_path):
    img = tf.image.decode_image(tf.io.read_file(image_path))
    img = tf.image.resize(img, [236, 236])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = img[None, ...]
    prediction = model.predict(img).flatten()
    class_index = np.argmax(prediction)
    predicted_disease = labels[class_index]
    prediction_percentage = prediction[class_index] * 100

    # Display predictions and percentages
    predictions_text.delete(1.0, tk.END)  # Clear the previous predictions
    predictions_text.insert(tk.END, f"Disease: {predicted_disease}\n")
    predictions_text.insert(tk.END, f"Confidence: {prediction_percentage:.2f}%\n")

# Create the main loop
window.mainloop()
