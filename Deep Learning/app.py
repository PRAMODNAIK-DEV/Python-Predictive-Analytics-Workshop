import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Cache the model so Streamlit doesn’t retrain every time you reload the app
@st.cache_resource
def load_model():
    # ---------------------------
    # 1. Load the MNIST dataset
    # ---------------------------
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train: training images (60,000 samples, shape 28×28)
    # y_train: training labels (digits 0–9)
    # x_test, y_test: test set (10,000 samples)

    # ---------------------------
    # 2. Normalize pixel values
    # ---------------------------
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Converts values from [0, 255] → [0, 1] (easier for neural networks to process)
    # In the MNIST dataset (and most image datasets), each pixel value is an integer between 0 and 255:
        # 0 → black
        # 255 → white
        # values in between → shades of gray

    # ---------------------------
    # 3. Add channel dimension
    # ---------------------------
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # MNIST images are grayscale (28×28). CNN expects 3D input: (height, width, channels).
    # Each pixel can have one or more color values, and each such "layer" is called a channel.
    # Channel = how many numbers describe the color of a single pixel. 1 means any number from 0-255. 
    # In RGB the channel will be 3 one for each Red Green and Blue. ([0-255] [0-255] [0-255])
    # expand_dims adds a new axis at the end → shape becomes (28, 28, 1).


    # ---------------------------
    # 4. Build CNN Model
    # ---------------------------
    model = keras.Sequential([      # Sequential = stack layers one after another
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        # Conv2D: applies 32 filters of size 3×3 to detect patterns (edges, curves, etc.)
        # activation="relu": introduces non-linearity
        # input_shape=(28,28,1): image shape (H, W, channels)

        layers.MaxPooling2D((2, 2)),    # Pooling downsamples the image (reduces size).
        
        layers.Flatten(),           # Flatten: converts 2D feature maps → 1D vector (for Dense layers)
        
        layers.Dense(128, activation="relu"),
        # Dense: fully connected layer with 128 neurons
        # activation="relu": helps capture complex patterns

        layers.Dense(10, activation="softmax"),
        # Dense: output layer with 10 neurons (one for each digit 0–9)
    ])

    # ---------------------------
    # 5. Compile Model
    # ---------------------------
    model.compile(
        optimizer="adam",                       # Adam = adaptive gradient optimizer                   
        loss="sparse_categorical_crossentropy", # loss for multi-class classification
        metrics=["accuracy"]                    # track accuracy during training
    )

    # ---------------------------
    # 6. Train the Model
    # ---------------------------
    model.fit(
        x_train, y_train,                       # training data
        validation_data=(x_test, y_test),       # This is data the model never sees during training, used only for evaluation after each epoch.
        epochs=10,                               # train for 1 full pass through dataset
        # Epoch = one full pass through the training dataset.
        # If epochs=1, the model sees all 60,000 MNIST images once.
        # # If epochs=10, the model will go through the dataset 10 times.
        batch_size=128,                         # number of samples per gradient update
        # Instead of feeding all 60,000 images at once, we split into mini-batches.
        # Each batch → 128 samples.
        # The model updates weights once per batch (called one step).
        verbose=1                               # Controls training output display. 1 = show progress bar. For example: 469/469 [==============================] - 10s 20ms/step - loss: 0.2554 - accuracy: 0.9271 - val_loss: 0.0897 - val_accuracy: 0.9730
    )
    
    # ---------------------------
    # 7. Return trained model
    # ---------------------------
    return model

model = load_model()

st.title("MNIST Digit Classifier")      # Streamlit App Title. Displays a big title at the top of the app.
st.write("Draw a digit (0–9) in the canvas below and let the model predict it!")    # Adds a description text below the title.

# ---------------------------
# Drawing Canvas
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# ---------------------------
# Button to Trigger Prediction
# ---------------------------
if st.button("Predict"):
    # This Creates a button labeled "Predict"
    # When clicked, the code inside this block runs.

    if canvas_result.image_data is not None:
        # This condition is to make sure the user has drawn something on the canvas
        # canvas_result.image_data = raw RGBA pixel data (280x280x4 array)

        # ---------------------------
        # Convert Canvas Drawing → PIL Image
        # ---------------------------
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        # Take only the first channel [:,:,0] → Red channel (since drawing is white on black, it's enough)
        # Convert the numpy array into a PIL image

        # Resize from (280x280) → (28x28), like MNIST
        # "L" = convert to grayscale (single channel)
        img = img.resize((28, 28)).convert("L")

        # ---------------------------
        # Preprocess for Model Input
        # ---------------------------
        img_array = np.array(img).astype("float32") / 255.0     # Convert image to numpy array, normalize pixel values (0–255 → 0–1)
        
        # Expand dimensions:
        #   axis=0 → adds batch dimension → shape becomes (1, 28, 28, 1)
        #   axis=-1 → ensures channel dimension (1 for grayscale)
        img_array = np.expand_dims(img_array, axis=(0, -1))

        # ---------------------------
        # Make Prediction
        # ---------------------------
        prediction = model.predict(img_array)       # Model outputs probabilities for each digit (shape: (1, 10))
        
        predicted_class = np.argmax(prediction)     # np.argmax → index of the highest probability → predicted digit (0–9)

        # ---------------------------
        # Show Results in Streamlit
        # ---------------------------
        st.subheader(f"Predicted Digit: **{predicted_class}**")     # Displays the predicted digit
        
        # Shows a bar chart of probabilities for all 10 digits
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before predicting.")
