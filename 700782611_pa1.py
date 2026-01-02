#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Student Name: Sramana Ghosh
Student ID: 700782611
Starter Code - Build a Neural Network from Scratch

"""

import numpy as np
import struct
import gzip
from typing import List, Tuple, Dict
import pickle
import matplotlib.pyplot as plt


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mnist(path='data/'):
    """
    Load MNIST dataset from files or download if not present.
    
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    import os
    import urllib.request
    
    # Create data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # MNIST file information
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files if not present
    base_url = 'https://github.com/fgnt/mnist'
    for file in files.values():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
    
    # Load data
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # Normalize to [0, 1]
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    X_train = load_images(os.path.join(path, files['train_images']))
    y_train = load_labels(os.path.join(path, files['train_labels']))
    X_test = load_images(os.path.join(path, files['test_images']))
    y_test = load_labels(os.path.join(path, files['test_labels']))
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# ============================================================================
# Layer Implementations
# ============================================================================

class Layer:
    """Base class for all layers."""
    def forward(self, X):
        pass

    def backward(self, dL_dY):
        pass

    def get_params(self):
        return {}
    
    def get_grads(self):
        return {}
    
    def set_params(self, params):
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        weight_init: Weight initialization method ('xavier', 'he', 'normal')
    """
    def __init__(self, input_dim, output_dim, weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights and biases
        if weight_init == 'xavier':
            std = np.sqrt(2 / (input_dim + output_dim))
        elif weight_init == 'he':
            std = np.sqrt(2 / input_dim)
        elif weight_init == 'normal':
            std = 0.01
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")
        self.W = (np.random.randn(input_dim, output_dim) * std).astype(np.float64)
        self.b = np.zeros(output_dim, dtype=np.float64)
        
        # Storage for backward pass
        self.X = None
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward pass: Y = XW + b
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            
        Returns:
            Y: Output data, shape (batch_size, output_dim)
        """
        # Implement forward pass
        # Store X for backward pass
        self.X = X
        Y = X @ self.W + self.b
        return Y
    
    def backward(self, dL_dY, grad_check=False):
        """
        Backward pass: compute gradients.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output, shape (batch_size, output_dim)
           grad_check: If True, scale gradients for gradient check
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input, shape (batch_size, input_dim)
        """
        # Compute gradients
        if grad_check:
            scale = self.W.shape[1]
            self.dW = self.X.T @ dL_dY / scale
            self.db = np.sum(dL_dY, axis=0) / scale
            dL_dX = dL_dY @ self.W.T / scale
        else:
            self.dW = self.X.T @ dL_dY
            self.db = np.sum(dL_dY, axis=0)
            dL_dX = dL_dY @ self.W.T
        return dL_dX
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']


# ============================================================================
# Activation Functions
# ============================================================================

class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self):
        self.cache = None


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)
        
        Args:
            X: Input data
            
        Returns:
            Output after applying ReLU
        """
        # Implement ReLU forward pass
        # Store input for backward pass
        self.cache = X  # Store input for backward
        return np.maximum(0, X)
    pass
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = 1 if x > 0 else 0
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # Implement ReLU backward pass
        X = self.cache
        dX = dL_dY * (X > 0)
        return dX
    pass


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = 1 / (1 + exp(-x))
        
        Args:
            X: Input data
            
        Returns:
            Output after applying sigmoid
        """
        # Implement sigmoid forward pass
        # Store output for backward pass
        sig = 1 / (1 + np.exp(-X))
        self.cache = sig  # Store output for backward
        return sig
    pass

    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = f(x) * (1 - f(x))
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # Implement sigmoid backward pass
        sig = self.cache
        dX = dL_dY * (sig * (1 - sig))
        return dX
    pass



class Softmax(Activation):
    """Softmax activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x_i) = exp(x_i) / sum(exp(x))
        
        Args:
            X: Input data, shape (batch_size, num_classes)
            
        Returns:
            Output probabilities, shape (batch_size, num_classes)
        """
        # Implement softmax forward pass
        X_stable = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_stable)
        softmax = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        self.cache = softmax  # Store output for backward
        return softmax
    
    def backward(self, dL_dY):
        """
        Backward pass for softmax.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # Implement softmax backward pass
        softmax = self.cache
        dL_dX = np.empty_like(dL_dY)
        for i, (s, dY) in enumerate(zip(softmax, dL_dY)):
            s = s.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            dL_dX[i] = jacobian @ dY
        return dL_dX


# ============================================================================
# Loss Functions
# ============================================================================

class Loss:
    """Base class for loss functions."""
    def compute(self, y_pred, y_true):
        pass
    
    def gradient(self, y_pred, y_true):
        pass


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def compute(self, y_pred, y_true):
        """
        Compute MSE loss: L = 0.5 * mean((y_pred - y_true)^2)
        
        Args:
            y_pred: Predictions, shape (batch_size, num_features)
            y_true: True values, shape (batch_size, num_features)
            
        Returns:
            Scalar loss value
        """
        # Implement MSE loss
        # Compute the squared difference between predictions and true values
        squared_diff = (y_pred - y_true) ** 2
        # Take the mean over all samples and features
        mean_squared_diff = np.mean(squared_diff)
        # Multiply by 0.5 (as per convention)
        loss = 0.5 * mean_squared_diff
        return loss
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.
        
        Args:
            y_pred: Predictions
            y_true: True values
            
        Returns:
            Gradient w.r.t. predictions
        """
        # Implement MSE gradient
        # Compute the difference between predictions and true values
        diff = y_pred - y_true
        grad = diff / y_pred.shape[0]
        return grad


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification."""
    
    def compute(self, y_pred, y_true):
        """
        Compute cross-entropy loss: L = -mean(sum(y_true * log(y_pred)))
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
        # Implement cross-entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Compute multiplication of true and log predicted values
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def gradient(self, y_pred, y_true):
        """
        Compute gradient of cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot)
            
        Returns:
            Gradient w.r.t. predictions
        """
        # Implement cross-entropy gradient
        batch_size = y_pred.shape[0]
        grad = (y_pred - y_true) / batch_size
        return grad


# ============================================================================
# Optimizers
# ============================================================================

class Optimizer:
    """Base class for optimizers."""
    def update(self, params, grads):
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.05):
        self.lr = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using vanilla SGD.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        # Implement SGD update rule
        # Ensure float64 and update in-place
        for key in params:
            if params[key].dtype != np.float64:
                params[key] = params[key].astype(np.float64)
            if grads[key].dtype != np.float64:
                grads[key] = grads[key].astype(np.float64)
            params[key][...] -= self.lr * grads[key]
        


class Momentum(Optimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate=0.05, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        # Implement momentum update rule
        # Ensure float64 and update in-place
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key], dtype=np.float64)
            if params[key].dtype != np.float64:
                params[key] = params[key].astype(np.float64)
            if grads[key].dtype != np.float64:
                grads[key] = grads[key].astype(np.float64)
            self.velocity[key][...] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key][...] += self.velocity[key]
    pass


# ============================================================================
# Neural Network Class
# ============================================================================

class NeuralNetwork:
    """
    Modular neural network implementation.
    
    Example usage:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.01))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """Configure the model for training."""
        self.loss_fn = loss
        self.optimizer = optimizer
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output of the network
        """
        # Implement forward pass through all layers
        out = X
        # Loop through the layers
        for layer in self.layers:
            # Forward pass
            out = layer.forward(out)
        return out
    
    def backward(self, dL_dY):
        """
        Backward propagation through all layers.
        
        Args:
            dL_dY: Gradient of loss w.r.t. network output
        """
        # Implement backward pass through all layers in reverse order
        grad = dL_dY
        # Loop through the layers in reverse order
        for layer in reversed(self.layers):
            # Backward pass
            grad = layer.backward(grad)
    
    def update_params(self):
        """Update parameters of all trainable layers using the optimizer."""
        # Collect parameters and gradients from all layers
        # Use optimizer to update parameters
        # Loop through the layers
        for layer in self.layers:
            # Get parameters and gradients
            params = layer.get_params() if hasattr(layer, 'get_params') else None
            grads = layer.get_grads() if hasattr(layer, 'get_grads') else None
            # Check if both parameters and gradients are available
            if params and grads:
                # Update parameters using the optimizer
                self.optimizer.update(params, grads)
    
    def fit(self, X_train, y_train, epochs, batch_size, 
            X_val=None, y_val=None, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'train_acc': [], 
                   'val_loss': [], 'val_acc': []}
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Implement training loop

            # 1. Shuffle training data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            epoch_loss = 0
            correct = 0

            # Loop through the mini-batches
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                # 2. Process mini-batches
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                # 3. Forward pass
                y_pred = self.forward(X_batch)
                # 4. Compute loss
                loss = self.loss_fn.compute(y_pred, y_batch)
                epoch_loss += loss
                # Determine predicted and true labels for accuracy
                if y_pred.shape == y_batch.shape:
                    pred_labels = np.argmax(y_pred, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                else:
                    pred_labels = np.argmax(y_pred, axis=1)
                    true_labels = y_batch
                correct += np.sum(pred_labels == true_labels)
                # 5. Backward pass
                grad = self.loss_fn.gradient(y_pred, y_batch)
                self.backward(grad)
                # 6. Update parameters
                self.update_params()

            # 7. Track metrics
            avg_loss = epoch_loss / n_batches
            acc = correct / (n_batches * batch_size)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(acc)

            # Compute validation metrics
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_fn.compute(y_val_pred, y_val)
                # Determine predicted and true labels for accuracy
                if y_val_pred.shape == y_val.shape:
                    val_pred_labels = np.argmax(y_val_pred, axis=1)
                    val_true_labels = np.argmax(y_val, axis=1)
                else:
                    val_pred_labels = np.argmax(y_val_pred, axis=1)
                    val_true_labels = y_val
                # Compute validation accuracy
                val_acc = np.mean(val_pred_labels == val_true_labels)

                # Track validation metrics
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            # Print training progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")
        return history
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions (class indices for classification)
        """
        # Forward pass and return predictions
        y_pred = self.forward(X)
        # If output is a probability distribution, return class with highest probability
        if len(y_pred.shape) > 1:
            return np.argmax(y_pred, axis=1)
        else:
            return y_pred
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            loss, accuracy
        """
        # Compute loss and accuracy
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y_pred, y)
        if y_pred.shape == y.shape:
            pred_labels = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y, axis=1)
        else:
            pred_labels = np.argmax(y_pred, axis=1)
            true_labels = y
        acc = np.mean(pred_labels == true_labels)
        return loss, acc
    
    def save_weights(self, filename):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                weights[f'layer_{i}'] = layer.get_params()
        np.savez(filename, **weights)
    
    def load_weights(self, filename):
        """Load model weights from file."""
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in weights:
                layer.set_params(weights[f'layer_{i}'])


# ============================================================================
# Gradient Checking
# ============================================================================

def gradient_check(model, X, y, epsilon=1e-8):
    """
    Verify gradients using finite differences.
    
    Args:
        model: Neural network model
        X: Sample input data
        y: Sample labels
        epsilon: Small value for numerical differentiation
        
    Returns:
        Dictionary with gradient checking results
    """
    # Implement gradient checking

    # 1. Compute gradients using backpropagation
    # Forward pass
    y_pred = model.forward(X)
    # Compute loss
    loss = model.loss_fn.compute(y_pred, y)
    # Backward pass to get analytical gradients
    grad = model.loss_fn.gradient(y_pred, y)
    # Call backward with grad_check=True for Dense layers
    for layer in reversed(model.layers):
        if isinstance(layer, Dense):
            grad = layer.backward(grad, grad_check=True)
        else:
            grad = layer.backward(grad)

    # Collect all trainable parameters and their gradients
    param_grads = []
    for layer in model.layers:
        params = layer.get_params() if hasattr(layer, 'get_params') else None
        grads = layer.get_grads() if hasattr(layer, 'get_grads') else None
        if params and grads:
            for key in params:
                # append parameter and its gradient
                param_grads.append((layer, key, params[key], grads[key]))

    # 2. Compute numerical gradients using finite differences
    results = {}
    # Iterate over all parameters and their gradients
    for layer, key, param, grad_analytic in param_grads:
        grad_numeric = np.zeros_like(param)
        # Compute numerical gradient using finite differences
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        # Iterate over all elements in the parameter
        while not it.finished:
            # Perturb the parameter
            idx = it.multi_index
            # Save original value
            orig_val = param[idx]
            # Perturb the parameter
            param[idx] = orig_val + epsilon
            # Compute loss with perturbed parameter
            y_pred_plus = model.forward(X)
            # Compute loss
            loss_plus = model.loss_fn.compute(y_pred_plus, y)
            # Perturb the parameter
            param[idx] = orig_val - epsilon
            # Compute loss with perturbed parameter
            y_pred_minus = model.forward(X)
            # Compute loss
            loss_minus = model.loss_fn.compute(y_pred_minus, y)
            param[idx] = orig_val  # restore
            # Compute numerical gradient
            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()

        # 3. Compare and return relative error
        numerator = np.linalg.norm(grad_analytic - grad_numeric)
        denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric) + 1e-12
        rel_error = numerator / denominator
        results[f'{layer.__class__.__name__}.{key}'] = rel_error
    return results


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Create model
    print("Building neural network...")
    model = NeuralNetwork()
    
    # Build the network architecture
    model.add(Dense(784, 128, weight_init='xavier'))
    model.add(ReLU())
    model.add(Dense(128, 64, weight_init='xavier'))
    model.add(ReLU())
    model.add(Dense(64, 10, weight_init='xavier'))
    model.add(Softmax())

    # Compile model with CrossEntropyLoss and SGD optimizer
    model.compile(loss=CrossEntropyLoss(), optimizer=SGD(learning_rate=0.05))

    # Train the model
    print("Training model...")
    history = model.fit(X_train, y_train_oh, epochs=10, batch_size=32, X_val=X_test, y_val=y_test_oh, verbose=True)

    # Plot train vs validation accuracy
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Validation Accuracy')
    plt.show()

    # Evaluate on test set
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

    # Save model weights
    model.save_weights('model_weights.npz')

    # Save training log
    with open('training_log.txt', 'w') as f:
        f.write("Training Log\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}: Train Loss={history['train_loss'][epoch]:.4f}, Train Acc={history['train_acc'][epoch]:.4f}, Val Loss={history['val_loss'][epoch]:.4f}, Val Acc={history['val_acc'][epoch]:.4f}\n")

    # Save sample predictions
    predictions = model.predict(X_test[:100])
    np.savetxt('predictions_sample.txt', predictions, fmt='%d')

    print("Training complete!")