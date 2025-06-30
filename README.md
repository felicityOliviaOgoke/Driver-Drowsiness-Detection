# Driver-Drowsiness-Detection
ResNet-50 is CNN architecture that belongs to the ResNet (Residual Networks) family, a series of models designed to address the challenges associated with training deep neural networks.
Developed by researchers at Microsoft Research Asia, ResNet-50 is renowned for its depth and efficiency in image classification tasks.

The primary problem ResNet solved was the degradation problem in deep neural networks. As networks become deeper, their accuracy saturates and then degrades rapidly. This degradation is not caused by overfitting, but rather the difficulty of optimizing the training process.

ResNet solved this problem using Residual Blocks that allow for the direct flow of information through the skip connections, mitigating the vanishing gradient problem.

The residual block used in ResNet-50 is called the Bottleneck Residual Block. This block it has the following architecture:

![image](https://github.com/user-attachments/assets/ee28ec41-7763-41b0-b06b-b3afe131dfe9)


source : https://blog.roboflow.com/what-is-resnet-50/#:~:text=ResNet%2D50%20is%20CNN%20architecture,efficiency%20in%20image%20classification%20tasks.

+-------------------+           +------------------+
|      CPU          |           |   Device (tpu)   |
|-------------------|           |------------------|
|  Load image       |  ----->   | Run model.forward|
|  Resize, normalize|           | Backprop         |
+-------------------+           +------------------+
          ↑                              |
          |         optimizer.step()     ↓
          +-----------------------------+



from : https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])   
    ])
    
The inference transforms are available at ResNet50_Weights.IMAGENET1K_V1.transforms and perform the following preprocessing operations: 
Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

  
