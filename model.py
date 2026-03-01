import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# ------------------ CONFIGURATION ------------------
# IMPORTANT: Create a folder named 'dataset' in the same directory as this script.
# Inside 'dataset', create subfolders for each plant disease class
# (e.g., 'tomato_healthy', 'tomato_late_blight', etc.) and place your images there.
DATA_DIR = "dataset" 
MODEL_PATH = "plant_disease_model.keras"  # Modern Keras format
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10 # You can increase this for better accuracy if you have a good GPU

# ------------------ SOLUTIONS DICTIONARY ------------------
SOLUTIONS = {
    "bell_pepper_bacterial_spot": "Use copper-based fungicide sprays and avoid overhead watering.",
    "bell_pepper_healthy": "No disease detected. Maintain proper watering and fertilization.",
    "chilli_healthy": "Plant is healthy. Ensure balanced fertilizers and pest monitoring.",
    "chilli_leaf_curl": "Control whiteflies/aphids with neem oil or insecticides.",
    "chilli_leaf_spot": "Apply fungicides and remove infected leaves.",
    "chilli_whitefly": "Spray neem oil or imidacloprid for whitefly control.",
    "chilli_yellowish": "Use nitrogen-rich fertilizers and improve soil health.",
    "lemon_anthracnose": "Remove infected leaves and apply copper-based fungicides.",
    "lemon_citrus_canker": "Use resistant varieties, apply copper sprays, and prune affected areas.",
    "lemon_healthy": "No issues found! Keep monitoring your plant regularly.",
    "lemon_sooty_mould": "Wash leaves with mild soap solution and control sap-sucking insects.",
    "tomato_bacterial_spot": "Use disease-free seeds, apply copper sprays, and rotate crops.",
    "tomato_early_blight": "Apply fungicides like chlorothalonil or mancozeb.",
    "tomato_healthy": "Tomato plant is healthy. Continue good practices.",
    "tomato_late_blight": "Spray with metalaxyl fungicide and destroy infected plants.",
    "tomato_leaf_mold": "Improve air circulation, avoid leaf wetness, apply fungicide.",
    "tomato_mosaic_virus": "Remove infected plants, disinfect tools, and control aphids.",
    "tomato_septoria_leaf_spot": "Remove infected leaves, use fungicide sprays.",
    "tomato_spider_mites": "Spray with neem oil or miticide to control mites.",
    "tomato_target_spot": "Remove affected leaves, use fungicide treatment.",
    "tomato_yellow_leaf_curl_virus": "Control whiteflies and use resistant varieties."
}

# ------------------ MODEL TRAINING FUNCTION ------------------
def train_model_threaded(status_label):
    """
    Trains the model in a separate thread to keep the GUI responsive.
    Saves the trained model and the class indices to files.
    """
    try:
        # Step 1: Set up data generators
        status_label.config(text="Status: Preparing data...")
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255)

        train_data = datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset="training"
        )

        val_data = datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset="validation"
        )

        num_classes = len(train_data.class_indices)
        if num_classes == 0:
            raise ValueError("No image classes found. Check 'DATA_DIR' path and subdirectories.")

        # Step 2: Define the model architecture
        status_label.config(text="Status: Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Step 3: Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Step 4: Train the model
        status_label.config(text="Training in progress... Please wait ⏳")
        model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

        # Step 5: Save the model and the class indices "decoder ring"
        status_label.config(text="Status: Saving model and class data...")
        model.save(MODEL_PATH) 

        class_indices = train_data.class_indices
        with open(CLASS_INDICES_PATH, "w") as f:
            json.dump(class_indices, f)

        status_label.config(text="✅ Training complete! Model and classes saved.")
        messagebox.showinfo("Training", "Model trained and saved successfully!")

    except Exception as e:
        status_label.config(text=f"❌ Error: {str(e)}")
        messagebox.showerror("Error", str(e))

# ------------------ PREDICTION FUNCTION ------------------
def predict_image(img_path):
    """
    Loads the trained model and class indices to predict the class of a single image.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
        return "Model not found.", "Please train the model first."

    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load class indices
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)

        # Create a reverse mapping from index to class name
        labels = {v: k for k, v in class_indices.items()}

        # Prepare the image
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        # Get the class name and solution
        predicted_class = labels.get(class_idx, "Unknown")
        solution = SOLUTIONS.get(predicted_class, "No solution available.")

        return predicted_class, solution

    except Exception as e:
        return f"Prediction Error: {e}", "Could not process the image."

# ------------------ GUI APPLICATION CLASS ------------------
class PlantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Trainer & Predictor")
        self.root.geometry("600x600")

        tk.Label(root, text="Plant Disease Detection", font=("Arial", 18, "bold")).pack(pady=10)

        tk.Button(root, text="Train Model", command=self.start_training, width=25, height=2).pack(pady=5)
        tk.Button(root, text="Predict from Image File", command=self.predict_from_file, width=25, height=2).pack(pady=5)
        tk.Button(root, text="Capture from Camera", command=self.capture_from_camera, width=25, height=2).pack(pady=5)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=5)

        self.solution_label = tk.Label(root, text="", font=("Arial", 12), wraplength=550, justify="center")
        self.solution_label.pack(pady=10)

        self.status_label = tk.Label(root, text="Ready", font=("Arial", 10), fg="blue")
        self.status_label.pack(side="bottom", fill="x", ipady=5)

    def start_training(self):
        # Run training in a separate thread to avoid freezing the GUI
        threading.Thread(target=train_model_threaded, args=(self.status_label,), daemon=True).start()

    def predict_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.process_and_display_prediction(file_path)

    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        ret, frame = cap.read()
        cap.release()

        if ret:
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.process_and_display_prediction(img_path)
        else:
            messagebox.showerror("Capture Error", "Could not capture frame.")

    def process_and_display_prediction(self, img_path):
        self.show_image(img_path)
        disease, solution = predict_image(img_path)

        self.result_label.config(text=f"Prediction: {disease.replace('_', ' ').title()}")

        color = "green" if "healthy" in disease.lower() else "red"
        self.result_label.config(fg=color)
        self.solution_label.config(text=f"Suggestion: {solution}", fg="black")

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((250, 250)) # Resize while maintaining aspect ratio
        img = ImageTk.PhotoImage(img)
        self.img_label.config(image=img)
        self.img_label.image = img

# ------------------ MAIN EXECUTION BLOCK ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantApp(root)
    root.mainloop()
