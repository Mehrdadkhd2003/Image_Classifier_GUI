# Image Classification GUI (ResNet-50 + PyQt5)

## Overview
This project presents a desktop-based image classification application built upon a pretrained ResNet-50 convolutional neural network provided by PyTorch’s `torchvision` library.  
The system incorporates a graphical user interface (GUI) developed using PyQt5, enabling image classification through either real-time webcam capture or loading images from local storage.

Image inference is performed on the ImageNet-1K dataset. For each processed image, the predicted class label and the associated confidence score are rendered directly onto the image and displayed within the GUI.

## Technical Details

### Model
- **Architecture:** ResNet-50  
- **Pretrained Weights:** ImageNet-1K pretrained parameters  
- **Dataset:** ImageNet-1K (1,000 classes)  
- **Framework:** PyTorch (`torch`, `torchvision`)  

The network produces an output tensor of shape `(1, 1000)`, representing a probability distribution across all ImageNet classes. The top-1 classification result is obtained via `torch.topk`.

### Preprocessing Pipeline
All input images undergo a standardized preprocessing procedure prior to inference, consisting of the following stages:

- Resize operation applied to the shorter image side (256 pixels)
- Center cropping to a spatial resolution of 224 × 224
- Conversion to tensor representation
- Normalization using ImageNet statistical parameters:
  - **Mean:** `[0.485, 0.456, 0.406]`
  - **Standard Deviation:** `[0.229, 0.224, 0.225]`

These transformations are implemented using the `torchvision.transforms` module.

## GUI Design
The graphical user interface is implemented using PyQt5 and is organized into two primary components.

### Main Window
- Displays a real-time webcam stream acquired through OpenCV
- Provides controls for capturing images from the webcam
- Allows loading images from the local file system

### Capture Window
- Displays the selected image
- Executes the classification process
- Overlays the predicted class label and confidence score directly onto the image using OpenCV and Qt painting utilities

## External Dependencies
- **PyTorch** – neural network inference  
- **Torchvision** – pretrained ResNet-50 architecture and preprocessing transforms  
- **OpenCV** – webcam acquisition and image processing  
- **PyQt5** – graphical user interface framework  
- **Pillow (PIL)** – image loading and format conversion  
- **Requests / JSON** – retrieval of ImageNet class labels

## Running the Application
```bash
python Image_Classifier_GUI.py
