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
from model import digitrecognizer 
from Steppage import model_training_page

st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

def Home():
    with st.container():
        st.markdown("<h1 style='text-align: center; text-decoration: underline; margin-bottom:10px '>Introduction to Image Classification</h1>", unsafe_allow_html=True)
            
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("How Does it Work?")
            st.write(
                "Image classification is a computer vision task where a model learns to classify images into different categories or labels. "
                "It's like teaching a computer to recognize and understand the content of images, just like humans do."
            )
            st.write("Image classification is important because it allows computers to see and understand the world, just like humans. It's used in many cool things, like identifying faces in photos, helping doctors analyze medical images, or even in self-driving cars to recognize objects on the road.")
            
        with right_column:
            st.image("gal1.webp", caption="Image Classification", use_column_width=True)



    with st.container():
        st.header("How Computers See Images?")
        st.write("Computers see images as a bunch of tiny dots called pixels. Each pixel has a color, and together they create the whole picture. Image classification helps computers understand what's in the picture by looking at these colors and patterns.")
        st.markdown("<p style='font-weight: 600; font-size:18px; '>Examples of how computer perceive images:</p>", unsafe_allow_html=True)

        left, right = st.columns((1, 2))
        with left:
            st.image("digit.png",caption="Recognization of digit", use_column_width=True)
        with right:
            st.image("cat.png", caption="Different ways how computer see", width=600)
            
    # Load the MNIST digit classification dataset
    digits = load_digits()
            
    with st.container():
        st.write("---")
        st.title("MNIST Digit Classification")
        st.header("What is MNIST datset?")
        st.write("The MNIST dataset is a famous collection of handwritten digits that is commonly used for training and testing machine learning models in the field of image recognition. MNIST stands for (Modified National Institute of Standards and Technology). It consists of 28x28 pixel grayscale images of handwritten digits (0 through 9) along with their corresponding labels.")
        
        left, right = st.columns(2)
        with left:
            st.write("Enter the number of images to display:")
            # Take input from the user for the number of images
            num_images = st.slider("Choose the Number of Images to Display", 1, 10,9,1)
            st.write(f"Displaying {num_images} images.")

            # Display the selected number of images
            fig, axes = plt.subplots(3, 3, figsize=(6, 6), subplot_kw={'aspect': 'equal'})
            for i, ax in enumerate(axes.flat):
                if i < num_images:
                    resized_image = cv2.resize(digits.images[i], (64, 64), interpolation=cv2.INTER_LINEAR)
                    ax.imshow(resized_image, cmap='gray', interpolation='nearest')
                    ax.axis('off')
                    ax.set_title(f"Label: {digits.target[i]}")

            st.pyplot(fig)
            
        with right:
            # Button to visualize the dataset
            st.markdown("<br><br><br><br>", unsafe_allow_html=True)
            if st.button("Visualize Dataset"):
                # Code to generate and display the graph goes here
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(digits.images[i], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Label: {digits.target[i]}")

                st.pyplot(fig)
                
    with st.container():
        st.write("---")
        st.title("Test-Train Split")
        st.write("Test-train split is a technique used in machine learning to help our computers learn and understand things better, similar to how we learn from our homework and tests at school.  Let's imagine we are learning to recognize digits like 0 to 9. We have a bunch of examples to practice with, but we don't want to use all of them at once. That's where test-train split helps. We take most of our examples for practice (training set) and some for a test (testing set).")
        
        st.write("For example, if we have 100 digits to learn, we might use 80 for practice and keep 20 hidden for the test. This way, we can see how well we've really learned without peeking at the answers we already practiced. It's like a surprise quiz after doing homework. Test-train split helps us understand if we are really good at recognizing digits or if we need more practice. It's a smart way to make sure we really know our stuff!")
        
        left, right = st.columns(2)
        with left:  
            st.write("Enter the number of data samples and the split ratio:")

            # Take input from the user for number of data samples and split ratio
            num_samples = st.number_input("Number of Data Samples", min_value=1, value=100, step=1)
            split_ratio = st.slider("Split Ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
            
        with right:
            st.markdown("<br><br>", unsafe_allow_html=True)
            # Button to generate test and train data
            if st.button("Generate"):
                # Generate random data for demonstration purposes
                X = range(num_samples)
                y = [i % 2 for i in range(num_samples)]

                # Perform test-train split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)

                # Print test and train data
                st.markdown(f"<strong style='text-align: center; margin-bottom:10px '>Number of Training Samples: {len(X_train)}</strong>", unsafe_allow_html=True)
                st.markdown(f"<strong style='text-align: center; margin-bottom:10px '>Number of Test Samples: {len(X_test)}</strong>", unsafe_allow_html=True)


    with st.container():
        st.write("---")
        st.title("Build Your Digit Recognizing Model")
        st.write("In creating a digit recognition model, we employed two different methods – a simple neural network and a convolutional neural network (CNN). The simple neural network is like a smart student trying to recognize digits by learning patterns in pixel values. It has layers that help it understand and remember different aspects of the digit images. ")
        st.write("On the other hand, the CNN is like a special detective who looks at smaller parts of the images and gradually puts together the whole story. It uses layers like 'convolutional' and 'max-pooling' to find unique features in the digits.")
        
        left, right = st.columns(2)
        with left: 
            st.write("In our digit recognition adventure, we encountered two special friends called epochs and learning rate. Imagine teaching a friend to ride a bicycle. Each time they practice is like an epoch. The more they practice (epochs), the better they get at riding. Similarly, in our model world, epochs are the rounds of practice to improve accuracy. If we increase the learning rate or practice more epochs, our models might get smarter faster, but be careful not to make them too hasty – finding the right balance is the key to their success! ")
            st.write("Think of the learning rate as a guide for our models – it decides how big of a step they take while learning from examples. If the rate is too small, they might walk too slowly and miss important details; if it's too big, they might overshoot and make mistakes.")
            
        with right:
            st.image("gal4.jpg",caption="Structure of Model", use_column_width=True) 
        
        
    # Using SciKit-Learns fetch_openml to load MNIST data.
    # Load the training and testing datasets from local CSV files
    train_path = 'train.csv'  # Update with the actual path
    test_path = 'test.csv'    # Update with the actual path

    # Load the training dataset
    train_df = pd.read_csv(train_path)
    X = train_df.drop(columns=['label']).values.astype('float32') / 255.0
    y = train_df['label'].values.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with st.container():
        st.write("---")
        st.title("Digit Recognition Model Training")
        
        left, right = st.columns(2)
        with left: 
            # Model selection
            model_choice = st.radio("Select a Model", ["Simple Neural Network", "Convolutional Neural Network"])

            # Hyperparameter input
            epochs = st.slider("Number of Epochs", min_value=1, max_value=20, value=5)
            learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.5, value=0.1)
            
        with right:
            st.empty()            
        
        
    # Initialize session state
    if 'model_new' not in st.session_state:
        st.session_state.model_new = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'   

    train_button = st.button("Train Model")

    if train_button:
        st.text("Training in progress...")
        
        st.session_state.model_new = digitrecognizer(model_choice,epochs,learning_rate,X,y,X_train, X_test, y_train, y_test)
        
        # Training loop
        history = st.session_state.model_new.fit(X_train, y_train)
        
        # Store training results in session state
        st.session_state.training_results = {
            'history': history.history,
            'X_test': X_test,
            'y_test': y_test
        }
        # Display training results if available
    if 'training_results' in st.session_state:
        st.write("---")
        st.title("Training Results")

        # Retrieve training results from session state
        result_df = pd.DataFrame(st.session_state.training_results['history'])
        selected_columns = ["epoch", "valid_acc", "train_loss", "valid_loss", "dur"]
        result_df = result_df[selected_columns]

        st.table(result_df)

        st.success(f"Training completed. Final Accuracy: {result_df['valid_acc'].iloc[-1]:.2%}")
        st.success(f"Training completed. Training Loss: {result_df['train_loss'].iloc[-1]:.2%}")

        # Prediction
        y_pred = st.session_state.model_new.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Training completed. Accuracy: {accuracy:.2%}")
        
        # Plot Training Accuracy and Training Loss
        plt.figure(figsize=(12, 6))

        # Plot Training Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(result_df['epoch'], result_df['valid_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()

        # Plot Training Loss
        plt.subplot(1, 2, 2)
        plt.plot(result_df['epoch'], result_df['train_loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Display the plots
        st.pyplot(plt)
    


    with st.container():
        st.write("---")
        st.title("Make Predictions On Your Data")

        st.header("Test Your Trained Model:")
        st.write("The model has completed its training, and now you can test it with your own data to see how well it performs.")
        SIZE = 192

        canvas_result = st_canvas(
            fill_color="#ffffff",
            stroke_width=10,
            stroke_color='#ffffff',
            background_color="#000000",
            height=150,width=150,
            drawing_mode='freedraw',
            key="canvas",
        )


        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write('Input Image')
            st.image(img_rescaling) 

        # Predict button
        if st.button('Predict'):
            if st.session_state.model_new is not None:
                # Convert the image to grayscale
                test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                input_image = test_x.flatten().astype('float32') / 255.0
                
                # Convert to a PyTorch tensor
                input_tensor = torch.tensor(input_image).to(device)

                # Reshape the tensor to match the input dimensions expected by the model
                if model_choice == "Simple Neural Network":
                    input_tensor = input_tensor.view(1, -1)
                else:
                    input_tensor = input_tensor.view(1, -1,28,28)

                # Make the prediction using the pre-loaded model
                with torch.no_grad():
                    model_output = st.session_state.model_new.predict(input_tensor)

                # Print the predicted label
                predicted_label = model_output[0]
                
                st.success(f"The model predicts the digit: {predicted_label}")
                st.bar_chart(model_output)
                feedback = st.radio("Is the prediction correct?", ["Yes", "No"])
                if feedback == "No":
                    st.warning("Please try drawing a different image.")
                elif feedback == "Yes":
                    st.info("Great! If you'd like to predict another digit delete your present digit and make new one.")
            
            else:
                st.warning("Please train the model before making predictions.")
                
def main():
    st.sidebar.title("Navigation")
    pages = ["Digit Recognizer", "Model Training"]
    page = st.sidebar.selectbox("Choose a page", pages)

    if page == "Digit Recognizer":
        Home()
    elif page == "Model Training":
        model_training_page()
        
if __name__ == "__main__":
    main()


