# CIFAR-10 Classifier

This repository contains a Jupyter Notebook designed for building and training a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset using PyTorch. The code includes creating a custom model, training, and evaluating its performance.

## Features

- **Neural Network Architecture**:  
  A custom convolutional neural network (CNN) built with PyTorch, featuring:
  - Two convolutional layers.
  - Three fully connected layers.
  - Activation and pooling functions for efficient feature extraction.
  
- **Training and Evaluation**:  
  The notebook includes methods for training and evaluating the model on the CIFAR-10 dataset.

- **Visualization**:  
  Uses `matplotlib` for plotting results and analyzing model behavior.

## Code Snippet Example

```python
class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        a1 = F.relu(self.conv1(input))
        a1 = F.max_pool2d(a1, (2,2))
        a2 = F.relu(self.conv2(a1))
        a2 = F.max_pool2d(a2, (2,2))
        a2 = torch.flatten(a2, 1)
        a3 = F.relu(self.fc1(a2))
        a4 = F.relu(self.fc2(a3))
        output = self.fc3(a4)
        return output
```

## Requirements

- Python 3.7 or above
- PyTorch
- Matplotlib
- NumPy

## Contributions

Feel free to fork the repo, submit pull requests, or raise issues for improvements or questions.
```

You can copy and paste this code into a `README.md` file in your repository. Let me know if you'd like further edits! 😊
