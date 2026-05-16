import os
import numpy as np
from PIL import Image

def generate_expanded_dataset():
    base_path = 'dataset'
    categories = ['train', 'test']
    diseases = [
        'Apple___Black_rot', 'Apple___healthy', 'Tomato___Late_blight', 'Tomato___Early_blight',
        'Cherry___healthy', 'Corn___common_rust', 'Blueberry___healthy', 'Strawberry___healthy',
        'Strawberry___leaf_scorch', 'Grape___healthy', 'Peach___healthy', 'Peach___bacterial_spot',
        'Pepper_bell___bacterial_spot', 'Pepper_bell___healthy', 'Orange___healthy', 
        'Orange___huanglongbing', 'Potato___early_blight', 'Potato___late_blight', 
        'Potato___healthy', 'Grape___black_rot'
    ]
    
    print(f"Generating synthetic images for {len(diseases)} classes...")
    
    for category in categories:
        for disease in diseases:
            dir_path = os.path.join(base_path, category, disease)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            num_images = 5 # Small number for testing
            for i in range(num_images):
                if 'healthy' in disease.lower():
                    color_bias = [0, 200, 0] # Green
                elif 'blight' in disease.lower() or 'spot' in disease.lower() or 'rust' in disease.lower():
                    color_bias = [150, 100, 50] # Brown/Rusty
                else:
                    color_bias = [100, 100, 100] # Grayish
                
                data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                data = (data // 2) + np.array(color_bias, dtype=np.uint8)
                img = Image.fromarray(data)
                img.save(os.path.join(dir_path, f'dummy_{i}.jpg'))
                
    print("Dataset expansion complete.")

if __name__ == '__main__':
    generate_expanded_dataset()
