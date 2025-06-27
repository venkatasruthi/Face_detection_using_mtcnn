# Face Recognition using MTCNN & PyTorch

This project implements a face recognition pipeline using **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection and a simple **CNN classifier** built using **PyTorch** for identity classification. It also includes visualization utilities and proper data preprocessing techniques for face cropping and annotation-based labeling.

---

## Key Features

- Face Detection**: Detects and crops faces using `facenet-pytorch` MTCNN.
- Face Classification**: A lightweight CNN to classify faces based on labeled annotations.
- Custom Dataset Loader**: Reads images and JSON-based annotations, automatically detects faces, and prepares training/validation/test splits.
- Data Visualization**: Annotated bounding boxes and names rendered using `matplotlib`.
- PyTorch Training Loop**: Includes dataloader, collate functions, optimizer setup, and a first-batch training example.

---

## Project Structure

