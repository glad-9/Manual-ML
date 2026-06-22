# Manual ML Project Architecture

## Overview

The Manual ML project is a hands-on implementation of machine learning and deep learning concepts from first principles. It provides educational implementations of core ML components using NumPy, CuPy, Pandas, and Matplotlib.

## Core Components

### 1. Tensor System (`core/tensor.py`)

The foundation of the project is a custom tensor class that implements:
- Basic tensor operations (addition, multiplication, matrix multiplication)
- Automatic differentiation with backward pass computation
- Various activation functions (ReLU, Sigmoid, Tanh, etc.)
- Convolutional operations (im2col, pool2d)
- Data manipulation methods (reshape, transpose, pad2d)

Key features:
- GPU acceleration using CuPy
- Automatic gradient computation through reverse-mode automatic differentiation
- Support for complex computational graphs

### 2. Neural Network Components (`nn/` directory)

#### Layers
- Linear/Dense layers with configurable activation functions
- BatchNorm and Dropout layers
- Convolutional layers (with im2col implementation)
- Pooling layers (MaxPool2D)

#### Network Class (`nn/network.py`)
- Complete network training framework
- Forward and backward propagation
- Loss computation and optimization
- Training loop with validation and early stopping

### 3. Data Processing (`data_processing/`)

- Data loading utilities
- Data normalization and preprocessing
- Dataset splitting functionality
- Modality Support : Tabular, Image

### 4. Visualization Tools (`viz/`)

- Evaluation metrics visualization
- Loss curves and confusion matrices
- Model performance analysis

## Implementation Approach

This project emphasizes understanding concepts over optimization:
- Pure Python implementations using only NumPy & CuPy without external ML frameworks
- Explicit computation of gradients and backpropagation steps
- Clear separation between forward and backward passes
- Educational focus on mathematical foundations

## Key Features

### Automatic Differentiation
The tensor system implements reverse-mode automatic differentiation, allowing users to define complex computational graphs and automatically compute gradients.

### GPU Acceleration
Leverages CuPy for efficient GPU computations, enabling faster training of neural networks.

### Modular Design
Components are designed to be easily understood and modified, making this a great learning tool for understanding ML internals.
