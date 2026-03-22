# Scene Style Classification

## Overview

This model uses a convolutional neural network to classify the data into one of 17 classes.

---

## Preprocessing

For each image loaded, a series of transformations are applied to standardize the images and to improve accuracy and training speed.

The transformations are:

- **`PadToSquare()`**: A custom transformation that ensures the image is square; if not, it pads the image with black pixels.
- **`transforms.Resize(128, 128)`**: Resizes all pictures to 128×128 pixels to reduce computations required during training.
- **`EnsureRGB()`**: Custom transformation to ensure that all input images are 3-channel RGB images.
- **`transforms.ToTensor()`**: Transforms the image into a 3×128×128 tensor so that it can be batched and loaded to the GPU.
- **`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`**: Normalizes the tensor using the standard deviation and mean of the image tensors of the ImageNet dataset.

---

## Testing

Because the test data is not labeled, the `train_copy.sh` bash script was used to move 50 images from each class in the training directory to a directory with the same name under the test directory. These 50×17 images were then used as the benchmark to test the model.

---

## Network Architectures

During the development of the model, multiple architectures were used and tested.

### Shallow Two-Layer Network

The first network used was of the following architecture:

```python
class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.full = nn.Linear(16 * 72 * 72, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.full(x)
        return x
```

It provided fast training speeds but generalized poorly, giving results close to **18% accuracy**.

---

### Deeper Network Without ReLU Between Convolutions

The second architecture used is shown below:

```python
class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.full = nn.Linear(64 * 72 * 72, 128)
        self.full1 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.full(x)
        x = self.relu(x)
        x = self.full1(x)
        return x
```

The second network is deeper; however, the aim was to see whether chaining convolutional layers without ReLU activation between them would result in an improvement. This architecture provided slightly better results than the first architecture, but not enough to say it is an overall better architecture.

---

### Final Architecture

The final architecture used is shown below:

```python
class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()

        # Conv Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv Block 1: 128x128 -> 64x64
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Conv Block 2: 64x64 -> 32x32
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Conv Block 3: 32x32 -> 16x16
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # Conv Block 4: 16x16 -> 8x8
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)

        # Global Average Pooling: 8x8 -> 1x1
        x = self.global_avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

This architecture uses a convolution block with the pattern: **convolve → ReLU → convolve → ReLU → Pool**.

Key features include:

- Feature maps double with each convolutional block in order to capture a larger number of more abstract features.
- A global average pooling layer is used before the fully connected layers.
- Two fully connected linear layers are used, with dropout applied between them to mitigate overfitting.

This architecture proved to be the best, achieving an accuracy of **27%**.
