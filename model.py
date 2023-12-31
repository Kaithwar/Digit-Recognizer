import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def simple(epochs,learning_rate,X,y,X_train, X_test, y_train, y_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mnist_dim = X_train.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(y_train))
    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
        ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X

    torch.manual_seed(0)
    model = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=epochs,
        lr=learning_rate,
        device=device,
    )
    return model

def digitrecognizer(model_choice,epochs,learning_rate,X,y,X_train, X_test, y_train, y_test):
    
    # Build Neural Network with PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_choice == "Simple Neural Network":
        mnist_dim = X_train.shape[1]
        hidden_dim = int(mnist_dim/8)
        output_dim = len(np.unique(y_train))
        class ClassifierModule(nn.Module):
            def __init__(
                    self,
                    input_dim=mnist_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    dropout=0.5,
            ):
                super(ClassifierModule, self).__init__()
                self.dropout = nn.Dropout(dropout)
                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, X, **kwargs):
                X = F.relu(self.hidden(X))
                X = self.dropout(X)
                X = F.softmax(self.output(X), dim=-1)
                return X

        torch.manual_seed(0)
        model = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=epochs,
            lr=learning_rate,
            device=device,
        )
        # model.fit(X_train, y_train)

    else:  # Convolutional Neural Network
        XCnn = X.reshape(-1, 1, 28, 28)
        XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)
        class Cnn(nn.Module):
            def __init__(self, dropout=0.5):
                super(Cnn, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
                self.conv2_drop = nn.Dropout2d(p=dropout)
                self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
                self.fc2 = nn.Linear(100, 10)
                self.fc1_drop = nn.Dropout(p=dropout)

            def forward(self, x):
                x = torch.relu(F.max_pool2d(self.conv1(x), 2))
                x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

                # flatten over channel, height and width = 1600
                x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

                x = torch.relu(self.fc1_drop(self.fc1(x)))
                x = torch.softmax(self.fc2(x), dim=-1)
                return x
            

        torch.manual_seed(0)
        model = NeuralNetClassifier(
            Cnn,
            max_epochs=epochs,
            lr=learning_rate,
            optimizer=torch.optim.Adam,
            device=device,
        )
        X_train = XCnn_train;
        X_test = XCnn_test;

    return model