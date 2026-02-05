# PyTorch Workflow Guide

## Overview of the ML Pipeline with PyTorch

The machine learning pipeline with PyTorch typically involves the following key stages:

1. **Data ingestion**: Load and preprocess data using libraries like torchvision, torchaudio, or custom datasets.
2. **Data preparation**: Split data into training, validation, and test sets. Apply transformations and augmentations as needed.
3. **Modeling**: Define the neural network architecture using `torch.nn.Module`. Choose appropriate layers, activation functions, and loss functions.
4. **Training**: Set up the training loop, including forward pass, loss computation, backpropagation, and optimization using `torch.optim`.
5. **Evaluation**: Assess model performance on validation and test sets using relevant metrics. Adjust hyperparameters as necessary.
6. **Deployment**: Save the trained model using `torch.save` and load it for inference in production environments.
7. **Monitoring**: Track model performance over time and retrain as needed to maintain accuracy and relevance.

---

## Model Building Phase

We're going to focus on the model building phase here.

### Using Sequential API

```python
model = nn.Sequential(
    nn.Linear(1, 20), 
    nn.ReLU(),
    nn.Linear(20, 1)
)
```

### Using Custom Class

This can be replaced with a custom class as follows:

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x 
```

**Key Concepts:**
- `__init__()` defines the layers pretty much like gathering your ingredients for a recipe
- `forward()` describes how the input data flows through the network layers like following the steps of a recipe to create a dish.

---

## Loss Functions and Optimizers

### Loss Computation and Optimization Flow

```python
loss = loss_function(outputs, targets)  # measure
loss.backward()                          # diagnose (compute gradients)
optimizer.step()                         # update (update weights)
```

### Common Loss Functions

- **`nn.MSELoss()`**: Mean Squared Error Loss for regression tasks.
- **`nn.CrossEntropyLoss()`**: Cross-Entropy Loss for multi-class classification tasks, outputs confidence scores for each class.

Every loss function compares the model's predictions to the actual target values and computes a scalar value representing the error. Higher loss values indicate worse performance, while lower values indicate better performance.

⚠️ **Important**: 
- If you use `MSELoss` for classification tasks, the model may struggle to learn effectively because `MSELoss` does not account for the probabilistic nature of classification problems.
- If you use `CrossEntropyLoss` for regression tasks, the model may not converge properly because `CrossEntropyLoss` is designed for discrete class labels rather than continuous values.

### Other Loss Functions

1. **`nn.L1Loss()`**: Measures average absolute differences between predicted and target values. Useful for regression tasks where robustness to outliers is desired.

2. **`nn.BCEWithLogitsLoss()`**: Combines a sigmoid layer and binary cross-entropy loss in one class. Suitable for binary classification tasks.

3. **`nn.NLLLoss()`**: Negative Log Likelihood Loss, often used in conjunction with log-softmax outputs for multi-class classification tasks.

4. **`nn.SmoothL1Loss()`**: A combination of L1 and L2 loss, useful for regression tasks where you want to be less sensitive to outliers than `MSELoss` but more sensitive than `L1Loss`.

5. **`nn.KLDivLoss()`**: Kullback-Leibler Divergence Loss, used for measuring the difference between two probability distributions, often in tasks like variational autoencoders.

### Optimizers

---

## Understanding Backpropagation and Gradient Descent

Consider a neural network with 784 input features (28×28 image flattened), 128 hidden units, and 10 output classes (digits 0-9).

**First layer**: 784 × 128 weights = 100,352 parameters + 128 biases = **100,480 parameters**

**Second layer**: 128 × 10 weights = 1,280 parameters + 10 biases = **1,290 parameters**

**Total**: 100,480 + 1,290 = **101,770 parameters**

### How Gradients Work

`loss.backward()` looks at each of those parameters and asks them how much they contributed to the overall loss. These diagnostic scores are called **gradients**.

**Key Questions Answered by Gradients:**
- Who contributed?
- By how much?
- In which direction?

**Gradient Direction & Magnitude:**
- **Positive gradient**: Increasing that parameter will increase the loss, so we need to decrease that parameter.
- **Negative gradient**: Increasing that parameter will decrease the loss, so we need to increase that parameter.
- **Large values**: Parameter had a big impact on the loss.
- **Small values**: Parameter had little impact.

> ℹ️ `backward()` only computes these gradients, and does not update the parameters themselves.

### How Optimizers Work

`optimizer.step()` looks at those gradients and decides how to update each parameter to reduce the loss.

**The Goal**: Minimize the loss by adjusting the model's parameters based on the computed gradients. 

The gradient gives us the slope of the loss function with respect to each parameter. The slope tells us the direction and rate of change of the loss. We move in the opposite direction of the gradient to reduce the loss. That's why we go "downhill" on the loss landscape, hence the term **"gradient descent"**.

### SGD Optimizer Example

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**SGD** (Stochastic Gradient Descent) updates parameters using a small random subset (mini-batch) of the data at each step, rather than the entire dataset. This makes training faster and allows the model to start learning before seeing all the data.

**Learning Rate (lr)**: Controls how big of a step we take in the direction of the negative gradient.
- **Small learning rate**: Tiny steps, slow convergence but more precise adjustments.
- **Large learning rate**: Big steps, faster convergence but risks overshooting the optimal parameters.

### Advanced Optimizers

**Momentum Optimizer**: Momentum is an extension of SGD that helps accelerate gradients vectors in the right directions, thus leading to faster convergence.

**RMSProp Optimizer**: Root Mean Square Propagation (RMSProp) is an adaptive learning rate optimization algorithm designed to maintain a per-parameter learning rate that improves performance on non-stationary problems.

**Adam Optimizer**: Adaptive Moment Estimation (Adam) is an advanced optimization algorithm that combines the benefits of two other extensions of stochastic gradient descent, i.e., Momentum and RMSProp.

---

## Putting It All Together

### Basic Training Loop

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()              # Clear previous gradients
    outputs = model(inputs)            # Forward pass
    loss = loss_function(outputs, targets)  # Compute loss
    loss.backward()                    # Backward pass (compute gradients)
    optimizer.step()                   # Update weights
```

---

## Device Management

PyTorch doesn't automatically move tensors or models to GPU. You have to explicitly specify the device.

### Checking GPU Availability

```python
torch.cuda.is_available()
```

### Setting Up Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
```

### Moving Data to Device

```python
for inputs, targets in dataloader:
    inputs = inputs.to(device)         # Move inputs to the specified device
    targets = targets.to(device)       # Move targets to the specified device
```

### Checking Device Location

**For tensors:**
```python
print(inputs.device)  # Outputs: cuda:0 (if on GPU) or cpu (if on CPU)
```

**For models:**
```python
print(next(model.parameters()).device)  # Outputs: cuda:0 (if on GPU) or cpu (if on CPU)
```

---

## Complete Training Loop with Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Batch Size Recommendations

Batch size of 32-64 is common for training on GPUs, while larger batch sizes may be used on TPUs or specialized hardware.

