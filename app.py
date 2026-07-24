import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tempfile
import zipfile

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Facial Emotion Detector",
    page_icon="😊",
    layout="wide"
)

st.title("😊 AI Facial Emotion Detector")
st.write("Train your own emotion model and use it for image or webcam prediction.")

# -----------------------------
# Emotion labels
# -----------------------------
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = 48
MODEL_PATH = "emotion_model.keras"

# -----------------------------
# Create CNN model
# -----------------------------
def create_model(num_classes):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(48, 48, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Upload dataset
# -----------------------------
st.sidebar.header("📁 Dataset")

dataset_zip = st.sidebar.file_uploader(
    "Upload your dataset ZIP file",
    type=["zip"]
)

if dataset_zip is not None:

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = os.path.join(temp_dir, "dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(dataset_zip.getbuffer())

        extract_path = os.path.join(temp_dir, "dataset")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        st.success("Dataset uploaded and extracted successfully!")

        st.info(
            "Your dataset should contain train and test folders "
            "with emotion folders inside them."
        )

# -----------------------------
# Load existing model
# -----------------------------
model = None

if os.path.exists(MODEL_PATH):

    model = tf.keras.models.load_model(MODEL_PATH)

    st.sidebar.success("✅ Trained model loaded")


# -----------------------------
# Train model
# -----------------------------
st.sidebar.header("🧠 Train Model")

train_folder = ""
test_folder = ""

if dataset_zip is not None:
    train_folder = os.path.join(extract_path, "archive", "train")
    test_folder = os.path.join(extract_path, "archive", "test")


epochs = st.sidebar.slider(
    "Training epochs",
    min_value=1,
    max_value=30,
    value=5
)

if st.sidebar.button("🚀 Train Model"):

    if not os.path.exists(train_folder):

        st.error(
            "Training folder not found. "
            "Place the train folder in the same directory as app.py."
        )

    else:

        st.info("Loading training dataset...")

        train_data = tf.keras.utils.image_dataset_from_directory(
            train_folder,
            image_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
            batch_size=64,
            shuffle=True
        )

        test_data = None

        if os.path.exists(test_folder):

            test_data = tf.keras.utils.image_dataset_from_directory(
                test_folder,
                image_size=(IMG_SIZE, IMG_SIZE),
                color_mode="grayscale",
                batch_size=64,
                shuffle=False
            )

        model = create_model(len(train_data.class_names))

        st.info("Training started...")

        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs
        )

        model.save(MODEL_PATH)

        st.success("🎉 Model trained and saved successfully!")


# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = image.convert("L")

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    image_array = np.expand_dims(image_array, axis=-1)

    prediction = model.predict(image_array)

    emotion_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return emotion_index, confidence


# -----------------------------
# Input selection
# -----------------------------
st.header("📸 Emotion Detection")

source = st.radio(
    "Choose input method",
    [
        "📷 Live Webcam",
        "🖼 Upload Image"
    ]
)


# -----------------------------
# Upload image prediction
# -----------------------------
if source == "🖼 Upload Image":

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        if model is None:

            st.warning(
                "Please train or load a model first."
            )

        else:

            emotion_index, confidence = predict_emotion(image)

            emotion = EMOTIONS[emotion_index]

            st.success(
                f"Emotion: {emotion.upper()} | "
                f"Confidence: {confidence:.2f}%"
            )


# -----------------------------
# Live webcam prediction
# -----------------------------
if source == "📷 Live Webcam":

    camera_image = st.camera_input(
        "Take a picture with your webcam"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        st.image(
            image,
            caption="Webcam Image",
            width=400
        )

        if model is None:

            st.warning(
                "Please train or load a model first."
            )

        else:

            emotion_index, confidence = predict_emotion(image)

            emotion = EMOTIONS[emotion_index]

            st.success(
                f"Detected Emotion: {emotion.upper()} | "
                f"Confidence: {confidence:.2f}%"
            )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "AI Facial Emotion Detection System | "
    "Custom CNN trained on uploaded emotion dataset"
)
