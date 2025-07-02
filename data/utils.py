import os
from PIL import Image
import matplotlib.pyplot as plt

def show_first_5_images(folder_path):
    image_files = os.listdir(folder_path)[:5]

    plt.figure(figsize=(15, 5))
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)

        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"{image_file}")
        plt.axis('off')
    plt.show()

def check_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except Exception:
                print(f"Corrupted: {path}")
