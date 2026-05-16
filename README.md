# Plant Disease Detection using CNN (Deep Learning)

A modern, web-based plant disease detection platform powered by a Convolutional Neural Network (CNN).

## 🚀 Features
- **CNN-Powered Diagnosis**: Accurately detect diseases from leaf images.
- **Modern UI**: Sleek, glassmorphic design with nature-inspired aesthetics.
- **Detection History**: Automatically saves predictions to a local SQLite database for future reference.
- **Real-time Feedback**: Instant results with confidence scores.
- **Responsive Design**: Works on mobile and desktop.

## 🛠️ Project Structure
```
Plant_Disease_Detection_Project/
│
├── app.py                # Flask Web Application
├── train_model.py        # CNN Training Script
├── requirements.txt      # Python Dependencies
├── plant_history.db      # SQLite Database (Auto-created)
│
├── model/
│    ├── plant_disease_model.h5   # Trained CNN Model
│    └── class_indices.json       # Class mapping for inference
│
├── dataset/              # Dataset for training/testing
│    ├── train/
│    │     ├── Apple___Black_rot/
│    │     ├── Apple___healthy/
│    │     ├── Tomato___Late_blight/
│    │     ├── Tomato___Early_blight/
│    │     └── ...
│    └── test/
│          ├── Apple___Black_rot/
│          ├── Apple___healthy/
│          ├── Tomato___Late_blight/
│          ├── Tomato___Early_blight/
│          └── ...
│
├── templates/            # HTML Templates
│    ├── index.html
│    └── history.html
│
├── static/               # Static Files
│    ├── style.css
│    └── uploads/         # Saved User Uploads
│
└── README.md
```

## 📦 Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Place your training images in `dataset/train/` organized by class name folders (e.g., `Apple___Black_rot`, `Apple___healthy`).

3. **Train the Model**:
   ```bash
   python train_model.py
   ```
   *Note: This will save the trained model to `model/plant_disease_model.h5`.*

4. **Run the Web Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧪 Technology Stack
- **Backend**: Python, Flask, SQLAlchemy
- **Deep Learning**: TensorFlow, Keras
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript
- **Database**: SQLite

## 📝 Note
If the `model/plant_disease_model.h5` file is not found, the application will run in **Demo Mode** with mock predictions to showcase the UI.
