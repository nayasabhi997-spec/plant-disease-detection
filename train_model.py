import os
import json

# Robust import check
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("Error: TensorFlow not found. Please run: pip install tensorflow==2.15.0")
    exit(1)

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10
DATASET_PATH = 'dataset/train'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.h5')

def train():
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: {DATASET_PATH} is empty. Please add images to category folders.")
        return

    print("Loading dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f)

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model.save(MODEL_PATH)
    print(f"Success! Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train()
