import torch
from torch.utils.data import DataLoader
from data import get_datasets, show_first_5_images, check_images
from models import build_model
from training import train_fn


dataset_path = '/content/drive/MyDrive/kaggle_datasets/driver-drowsiness-dataset-ddd/Driver Drowsiness Dataset (DDD)/'

check_images(dataset_path)
show_first_5_images(f"{dataset_path}/Drowsy")
show_first_5_images(f"{dataset_path}/Non Drowsy")

train_ds, val_ds, _ = get_datasets(dataset_path)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model = model.to(device)

save_dir = "/content/drive/MyDrive/kaggle_datasets/driver-drowsiness-dataset-ddd/checkpoints"
train_fn(index=0, model=model, train_loader=train_loader, val_loader=val_loader, save_dir=save_dir, device=device)
