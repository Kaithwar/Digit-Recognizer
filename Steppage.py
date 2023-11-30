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
import cv2
from model import digitrecognizer,simple 

def model_training_page():
    with st.container():
        st.markdown("<h1 style='text-align: center; text-decoration: underline; margin-bottom:10px '>Digit Recognizer Model Training</h1>", unsafe_allow_html=True)
        st.header("Step-by-Step Guide: Training a Digit Recognizer Model and Analyzing Accuracy")
        # Add your steps here
        st.markdown("<p style='font-weight: 600; font-size:20px; '>Step 1: Introduction to Machine Learning Model Training</p>", unsafe_allow_html=True)
        st.write("Welcome to the world of machine learning! Today, we'll learn how to train a model that can recognize handwritten digits from 0 to 9. Digit recognition is about teaching a computer to identify handwritten numbers. For example, recognizing '5' written on paper.")
        
        st.markdown("<p style='font-weight: 600; font-size:20px; '>Step 2: Understanding the Data</p>", unsafe_allow_html=True)
        st.write("We'll use a famous dataset called MNIST, which contains thousands of images of handwritten digits. Each image is a 28x28 pixel grayscale picture.We have a bunch of handwritten numbers from 0 to 9. These are like the numbers you write in your notebook.")
        
        st.markdown("<p style='font-weight: 600; font-size:20px; '>Step 3: Splitting Data into 'Learn' and 'Test'</p>", unsafe_allow_html=True)
        st.write("To train our model, we'll divide the dataset into two parts: the training set and the test set. The training set will help the model learn, while the test set will evaluate its accuracy.")
        left, right = st.columns(2)
        with left: 
            split_ratio = st.slider("Split Ratio", min_value=0.1, max_value=0.9, value=0.75, step=0.1)
        with right:
            st.empty()
        train_path = 'train.csv'  # Update with the actual path
        test_path = 'test.csv'    # Update with the actual path

        # Load the training dataset
        train_df = pd.read_csv(train_path)
        X = train_df.drop(columns=['label']).values.astype('float32') / 255.0
        y = train_df['label'].values.astype('int64')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)
        
        st.write("---") 
        st.write("Now Discuss some Parameters on which the model Accuracy depends:")
        st.write("1)Learning Rate: Think of the learning rate as a step size or speed at which a model learns during training. It determines how much the model's parameters are updated during each iteration.")
        st.write("High Learning Rate: Rapid learning but might overshoot the optimal point. and May fail to converge or oscillate around the minimum.")
        st.write("Low Learning Rate: Slower learning, smaller parameter updates and Reduces the risk of overshooting but may take longer to converge.")

        st.write("2)Epochs: An epoch is one complete pass through the entire dataset during training.")
        st.write("Few Epochs: Model might not learn complex patterns, leading to underfitting and Incomplete learning, lower accuracy.")
        st.write("Too Many Epochs: Risk of overfitting, memorizing training data instead of learning and High accuracy on training data but lower accuracy on unseen data (test/validation set).")

        st.markdown("<p style='font-weight: 600; font-size:20px; '>Step 4: Making a Simple Brain (Neural Network)</p>", unsafe_allow_html=True)
        st.write("Now, we'll build a simple neural network - a type of model inspired by the human brain. It consists of interconnected nodes and layers.")
        st.write("Now, Firstly we will take low learning rate and high epoch and check its accuracy.")
        st.write("Let's take Learning Rate = 0.1 and Number of Epochs = 20")
        learning_rate = 0.1;
        epochs = 20;
        train_button1 = st.button("Train Model1")
        if train_button1:
            st.text("Training in progress...")            
            st.session_state.model_new = simple(epochs,learning_rate,X,y,X_train, X_test, y_train, y_test)
            history = st.session_state.model_new.fit(X_train, y_train)
            # Store training results in session state
            st.session_state.training_results1 = {
                'history': history.history,
                'X_test': X_test,
                'y_test': y_test
            }
        if 'training_results1' in st.session_state:
            st.write("---")
            st.title("Training Results of first model")
            # Retrieve training results from session state
            result_df = pd.DataFrame(st.session_state.training_results1['history'])
            selected_columns = ["epoch", "valid_acc", "train_loss", "valid_loss", "dur"]
            result_df = result_df[selected_columns]
            st.table(result_df)
            st.success(f"Training completed. Final Accuracy: {result_df['valid_acc'].iloc[-1]:.2%}")
            st.success(f"Training completed. Training Loss: {result_df['train_loss'].iloc[-1]:.2%}")
            y_pred = st.session_state.model_new.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Training completed. Accuracy: {accuracy:.2%}")
        
        st.write("---")    
        st.write("Seondly we will take high learning rate and low epoch and check its accuracy.")
        st.write("Take Learning Rate = 0.30 and Number of Epochs = 2")
        learning_rate = 0.3;
        epochs = 2;
        train_button2 = st.button("Train Model2")
        if train_button2:
            st.text("Training in progress...")            
            st.session_state.model_new = simple(epochs,learning_rate,X,y,X_train, X_test, y_train, y_test)
            history = st.session_state.model_new.fit(X_train, y_train)
            # Store training results in session state
            st.session_state.training_results2 = {
                'history': history.history,
                'X_test': X_test,
                'y_test': y_test
            }
        if 'training_results2' in st.session_state:
            st.write("---")
            st.title("Training Results of second model")
            # Retrieve training results from session state
            result_df = pd.DataFrame(st.session_state.training_results2['history'])
            selected_columns = ["epoch", "valid_acc", "train_loss", "valid_loss", "dur"]
            result_df = result_df[selected_columns]
            st.table(result_df)
            st.success(f"Training completed. Final Accuracy: {result_df['valid_acc'].iloc[-1]:.2%}")
            st.success(f"Training completed. Training Loss: {result_df['train_loss'].iloc[-1]:.2%}")
            y_pred = st.session_state.model_new.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Training completed. Accuracy: {accuracy:.2%}")
            
        st.write("---")
        st.write("In comparing two models, the first model utilized a low learning rate with a high number of epochs, while the second model employed a high learning rate and a low number of epochs. Surprisingly, the first model, characterized by its deliberate, methodical learning (low learning rate) and extensive practice (high epochs), showcased a noteworthy improvement in accuracy, surpassing the second model by more than 2%")

        st.write("This outcome demonstrates that allowing the model to learn slowly and steadily with ample practice epochs resulted in a significantly enhanced accuracy compared to the model trained with a swift but less thorough learning approach. It underscores the importance of patient, persistent learning over rapid, shorter learning sessions in achieving higher accuracy and better model performance.")
        