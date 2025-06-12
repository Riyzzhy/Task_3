Name : PATEL RIYANSH BHAVESHBHAI Company : CODTECH IT SOLUTIONS Intern_ID : CT04DF487 Domain : MACHINE LEARNING Duration Time : 24 MAY TO JUNE 2025 Mentor Name : NEELA SANTHOSH KUMAR

✅ TASK 3: Image Classification Module – Overview

🎯 Objective:
Build a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset into 10 predefined categories such as car, cat, ship, etc.

🧱 Modules & Libraries Used:
Module	Purpose
torch	Core PyTorch library for deep learning operations
torchvision	For downloading and transforming the CIFAR-10 dataset
torch.nn	Provides neural network layers and activations
torch.optim	Offers optimization algorithms like Adam
matplotlib.pyplot	Used to visualize training and validation accuracy

📦 Dataset Used:
Dataset: CIFAR-10 (available through torchvision.datasets)

Total Images: 60,000 (50,000 for training, 10,000 for testing)

Image Size: 32x32 pixels with 3 color channels (RGB)

Classes: 10 — ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

🧠 Model Architecture (CNN):
Layer	Configuration
Conv1	Input: 3×32×32 → Output: 32 feature maps, kernel=3
ReLU + MaxPool	2x2 max pooling
Conv2	64 feature maps, kernel=3
ReLU + MaxPool	Again 2x2
FC1	Fully connected layer (64×8×8 → 128 neurons)
FC2	Final classification layer (128 → 10 classes)

🏃‍♂️ Training Details:
Loss Function: CrossEntropyLoss (suitable for multi-class classification)

Optimizer: Adam (learning rate = 0.001)

Epochs: 10

Batch Size: 64

Metrics: Accuracy on both training and test datasets
