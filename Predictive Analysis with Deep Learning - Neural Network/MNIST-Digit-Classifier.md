# MNIST Digit Classifier with Streamlit

This is a beginner-friendly **Deep Learning + Streamlit mini project** that allows you to draw digits (0–9) on a canvas and lets a **Convolutional Neural Network (CNN)** predict them in real time.


---

## Concepts Covered
1. **Data Science workflow** → dataset, preprocessing, training, evaluation  
2. **Convolutional Neural Networks (CNNs)** → Conv2D, MaxPooling, Flatten, Dense  
3. **Training MNIST dataset** → Handwritten digits recognition  
4. **Streamlit basics** → interactive web app for ML models  
5. **Drawable Canvas** → drawing digits and feeding into the model  

---

## Installation & Setup

### Step 1: Create a virtual environment (optional but recommended)
Let's create a folder named `Deep Learning` and Create a virtual environment as follows in your `CMD` or `Terminal`:
```bash
cd Deep Learning
python -m venv venv
.\venv\Scripts\activate
```
Note: If you get any Error or Warning then Run the following script in your Terminal
```bash 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install the dependencies

Create a `requirements.txt` inside the `Deep Learning` project root folder with the following:

```python
streamlit
tensorflow
numpy
pillow
streamlit-drawable-canvas
```
### Now install:

```bash
pip install -r requirements.txt
```

### Step 3: Understanding MNIST
- MNIST: 70k grayscale images of handwritten digits (28×28). Values are in 0–255; we normalize to 0–1.

- CNN (Convolutional Neural Network):

    - Conv2D: Learns filters to detect strokes/edges.
    - MaxPooling2D: Downsamples (keeps salient features).
    - Flatten: Converts 2D feature maps into a vector. A vector is simply a 1-dimensional array of numbers (a list).
    - Dense: Fully connected layers for classification.
    - softmax: Outputs probabilities across the 10 classes.

Mini example (concept):
- If your input pixel is 255 (white), dividing by 255 → 1.0. Black background is 0.0.
- The first conv layer might learn a vertical-edge detector that fires strongly on vertical strokes of “1”, “4”, etc.

## Project Implementation
Create a main project file name app.py and place the below codes one by one:
###  Step 4: Import Libraries

```python
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image
```
Here,
- `streamlit` → Creates interactive web apps in Python (used for data apps & ML demos).
- `numpy` → Handles numerical computations and arrays efficiently.
- `tensorflow` → Deep learning framework for building and training neural networks.
- `keras` → High-level API inside TensorFlow for easy model building.
- `layers` → Module in Keras for defining neural network layers (Dense, Conv2D, etc.).
- `streamlit_drawable_canvas` → Lets users draw on a canvas inside a Streamlit app.
- `PIL (Pillow)` → Library for opening, manipulating, and saving images.

### Step 5: Load/Train Model

```python
# Cache the model so Streamlit doesn’t retrain every time you reload the app
# This is because in Streamlit, every time you reload or interact with the app (like pressing a button, changing a slider, etc.), the script runs from top to bottom again.
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
        # Conv2D: Convolutional layer for 2D images. It applies 32 filters of size 3×3 to detect patterns (edges, curves, etc.)
        # Filters (or kernels) = small sliding windows that look for patterns in the image.
        # 32 filters = the layer will learn 32 different kinds of patterns.
        # Size 3×3 = each filter looks at a small patch of 3×3 pixels at a time.
        # activation="relu": introduces non-linearity. Without activation, a neural network would just be linear (like a straight line equation).
        # input_shape=(28,28,1): image shape (H, W, channels)

        layers.MaxPooling2D((2, 2)),    # Pooling downsamples the image (reduces size) to improve the performance by Keeping important info
        
        layers.Flatten(),           # Flatten: converts 2D feature maps → 1D vector (for Dense layers)
        # Convolutions & pooling → extract features (edges, shapes, textures).
        # Flatten → turns those features into a vector.
        # Dense layers → combine those features to make the final decision (e.g., "This is a cat").
        
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
        epochs=1,                               # train for 1 full pass through dataset
        # Epoch = one full pass through the training dataset.
        # If epochs=1, the model sees all 60,000 MNIST images once.
        # If epochs=10, the model will go through the dataset 10 times.
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
```

Where,
1. Conv2D = Convolution layer (extracts features like edges, curves, shapes).
    - `32` = number of filters (also called kernels). Each filter learns to detect a different feature.
    - `kernel_size=(3, 3)` = each filter is a 3×3 grid that slides over the image.
    - `activation="relu"` = introduces non-linearity (turns negatives → 0).
    - `input_shape=(28, 28, 1)` = input images are 28×28 grayscale (1 channel).

2. MaxPooling2D(pool_size=(2, 2))
    - Pooling downsamples the image (reduces size).
    - `pool_size=(2, 2)` = take the max value from each 2×2 block.
    - This reduces computation and makes features more robust to shifts.
    - `Example`: (28×28 → 14×14).

3. layers.Flatten()
    - Converts the 2D feature maps into a 1D vector.
    - Needed before passing data to Dense (fully connected) layers.

4. Dense(128, activation="relu")
    - A fully connected layer with 128 neurons.
    - Each neuron learns different combinations of features from the convolution layers.
    - relu makes it non-linear.

5. layers.Dense(10, activation="softmax")
    - Final output layer.
    - 10 = number of classes in MNIST (digits 0–9).
    - softmax = converts outputs into probabilities that sum to 1.
    - Example output for an image of "7":
      - [0.01, 0.02, 0.00, 0.00, 0.05, 0.00, 0.01, 0.89, 0.01, 0.01]
      - Here the model predicts `7` with `89%` confidence.

6. Adam = adaptive gradient optimizer:
In training a neural network:
     - The model makes predictions.
     - The loss function measures how wrong those predictions are.
     - The optimizer updates the model’s weights to reduce the loss.
     - So an optimizer = the “engine” that learns by adjusting weights.


### Step 6: Create Streamlit UI

```python
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
```
Here,
 - `st_canvas()` creates a drawable area where you can sketch digits.
 - Returns an object (canvas_result) containing the drawn image in `RGBA` format.


### Step 7: Make Prediction

```python
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

```

## Run the app
```bash
streamlit run app.py
```

Please Monitor the `Terminal`/`CMD`:
```bash
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9384 - loss: 0.2175 - val_accuracy: 0.9754 - val_loss: 0.0774
```
This indicates training log output from Keras/TensorFlow:
1. 469/469
    - Number of batches processed in this epoch.
    - Your dataset was divided into 469 batches, and the model processed all of them.


2. 14s 28ms/step
    - Training this epoch took 14 seconds total.
    - Each batch (step) took 28 milliseconds.

3. accuracy: 0.9384
    - Training accuracy after this epoch = 93.84%.
    - On the training data, the model predicted correctly ~94% of the time.

4. loss: 0.2175
    - Training loss (how wrong the model is) = 0.2175.
    - Lower loss = better fit.
    - Loss is the value the optimizer tries to minimize.

5. val_accuracy: 0.9754
    - Validation accuracy (on unseen validation data) = 97.54%.
    - Means the model generalized well — it’s even doing better on validation than training.

6. val_loss: 0.0774
    - Validation loss = 0.0774 (much lower than training loss).
    - Suggests the model is not overfitting yet and is learning meaningful patterns.


## THe Prediction is Not Proper?
The reason is drawn digit ≠ MNIST digit.
- MNIST digits are centered, cropped, padded to square, thin/blurred strokes, 28×28 grayscale, white-on-black.
- Our canvas image is big (280×280), off-center, thick strokes, and just resized — so the network sees a different distribution and misclassifies.

#### Solution
Preprocess the canvas like MNIST:
Add the below function that crops to the digit, pads to a square, gently blurs, resizes to 28×28, normalizes, and keeps white digit on black.

```python
from PIL import Image, ImageOps, ImageFilter

def preprocess_canvas_image(image_data):
    # image_data is RGBA from st_canvas, shape (H, W, 4)
    img = Image.fromarray(image_data.astype(np.uint8)).convert("L")

    # Stretch contrast a bit (removes faint noise, increases separation)
    img = ImageOps.autocontrast(img)

    arr = np.array(img)
    # Remove tiny noise
    arr[arr < 10] = 0

    # Find tight bounding box around the digit
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None  # nothing drawn

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = arr[y0:y1+1, x0:x1+1]

    # Pad to square with a small margin
    h, w = crop.shape
    s = max(h, w) + 8  # margin
    square = np.zeros((s, s), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = crop

    # Slight blur to mimic MNIST stroke style
    square = Image.fromarray(square).filter(ImageFilter.GaussianBlur(radius=1))

    # Resize to 28x28 with high-quality downsampling
    square = square.resize((28, 28), Image.LANCZOS)

    arr28 = np.array(square).astype("float32") / 255.0  # 0..1

    # Ensure it's white-on-black; if user flips colors, auto-correct:
    if arr28.mean() > 0.5:  # likely white background
        arr28 = 1.0 - arr28

    return np.expand_dims(arr28, (0, -1))  # (1,28,28,1)

```

Replace the Button to Trigger Prediction Block with the below code:

```python
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img_array = preprocess_canvas_image(canvas_result.image_data)
        if img_array is None:
            st.warning("Please draw a digit before predicting.")
        else:
            st.image((img_array[0, :, :, 0] * 255).astype("uint8"), caption="Preprocessed 28×28", width=96, clamp=True)
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            st.subheader(f"Predicted Digit: **{predicted_class}**")
            st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before predicting.")

```