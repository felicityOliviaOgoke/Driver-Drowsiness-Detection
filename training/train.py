import torch
import os
import torch.nn as nn
import torch.optim as optim

def train_fn(index, model, train_loader, val_loader, save_dir, device, start_epoch=0, num_epochs=2):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"resnet50_gpu_{index}_checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"[gpu:{index}] Resuming from epoch {start_epoch} with best acc {best_acc:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if i % 20 == 0:
                print(f"[gpu:{index}] Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"[gpu:{index}] Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}")

        # Validation
        val_acc = validate_fn(model, val_loader, device, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_gpu_{index}.pth"))
            print(f"[gpu:{index}] Best model saved with acc {best_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

def validate_fn(model, val_loader, device, criterion):
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(f"[Validation] Loss: {val_loss/val_total:.4f}, Accuracy: {val_acc:.4f}")
    return val_acc
