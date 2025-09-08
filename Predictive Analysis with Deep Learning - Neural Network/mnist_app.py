import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# -------------------------------
# Load or train MNIST model
# -------------------------------
@st.cache_resource
def load_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Build a simple model
    model = keras.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train for 1 epoch (quick demo)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=128, verbose=1)

    return model

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("MNIST Digit Classifier")
st.write("Draw a digit (0–9) in the canvas below and let the model predict it!")

# Create a canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Black ink
    stroke_width=10,
    stroke_color="#FFFFFF",  # White digit
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to PIL
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

        # Resize to 28x28
        img = img.resize((28, 28)).convert("L")

        # Normalize
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.subheader(f"Predicted Digit: **{predicted_class}**")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before predicting.")
