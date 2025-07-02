
# Driver Drowsiness Detection

This project uses **ResNet-50**, a powerful convolutional neural network (CNN), to classify whether a driver is **drowsy or alert** based on facial images. It is built with PyTorch and designed for scalability, modularity, and ease of training and inference.

---

## Why ResNet-50?

ResNet-50 belongs to the **Residual Networks** family developed by Microsoft Research. It solves the **degradation problem** in deep neural networks, where adding more layers leads to worse performance due to vanishing gradients.

### Key Innovation: Residual Blocks

Residual blocks introduce **skip connections**, allowing gradients to flow more effectively and enabling deep networks like ResNet-50 to train efficiently.

ResNet-50 specifically uses the **Bottleneck Residual Block**:

![Bottleneck Residual Block](https://github.com/user-attachments/assets/ee28ec41-7763-41b0-b06b-b3afe131dfe9)

> _Source: [Roboflow ResNet-50 Overview](https://blog.roboflow.com/what-is-resnet-50)_

---

## Dataset

The model is trained on a labeled image dataset stored on Google Drive:
```
/content/drive/MyDrive/kaggle_datasets/driver-drowsiness-dataset-ddd/
```

- Class 0: `Drowsy`
- Class 1: `Non Drowsy`

You can visualize and verify dataset integrity using built-in utilities:
```python
from data import show_first_5_images, check_images
check_images(dataset_path)
show_first_5_images("Drowsy")
```

---

## Preprocessing & Transforms

Images are resized and normalized to match the ResNet-50 input requirements:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

These values are aligned with the **ImageNet** pretrained weights:
- `mean = [0.485, 0.456, 0.406]`
- `std = [0.229, 0.224, 0.225]`

---

## Project Structure

```
driver-drowsiness-detection/
├── data/
│   ├── dataset.py          # Transforms & dataset loading
│   └── utils.py            # Image checking & visualization
├── models/
│   └── resnet.py           # Transfer learning model setup
├── training/
│   └── train.py            # Training and validation logic
├── main.py                 # Entry point for training
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Training

Run the training script:

```bash
python main.py
```

Model checkpoints and best weights are saved to:
```
/content/drive/MyDrive/kaggle_datasets/driver-drowsiness-dataset-ddd/checkpoints/
```

Training uses **transfer learning**, freezing the base ResNet layers and training only the classifier head.

---

##  Evaluation

The `train_fn()` function includes validation accuracy tracking. Evaluation metrics and loss are printed at the end of each epoch.

Future improvements may include:
- Confusion matrix visualization
- Precision/recall scoring
- Model export to ONNX or TorchScript

---

##  Requirements

```
torch
torchvision
matplotlib
Pillow
scikit-learn
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  References

- [ResNet Paper (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [ResNet50 PyTorch Docs](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [Roboflow ResNet Blog](https://blog.roboflow.com/what-is-resnet-50)

---

