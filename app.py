import subprocess
import sys
import os
dir_path = os.getcwd()

# List all files in the directory
files = os.listdir(dir_path)

# Print the list of files
print(files)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

with open("food-vision//req.txt") as file:

    # Create an empty list to store the lines
    lines = []

    # Loop through each line in the file
    for line in file:

        # Append the line to the list
        lines.append(line.strip())


for line in lines:
    install(line) 
    

import os
import json
import requests
import SessionState
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json

import os
import json
import requests
import SessionState
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json

# Setup environment credentials (you'll need to change these)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json" # change for your GCP key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "fire-detection-v0001-b5b267b2b62e.json"
PROJECT = "fire-detection-v0001" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Fire Detection 🔥📸")
st.header("Identify if there is a fire in your image!")


@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    
    index = tf.argmax(preds[0])
    pred_conf = tf.reduce_max(preds[0])
    pred_class = class_names[index]
    return image, pred_class, pred_conf

# Pick the model versionn 
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (Fire or No fire)", # original 10 classes ---- THIS 
     # original 10 classes + donuts
     ) # 11 classes (same as above) + not_food class
)

# Model choice logic
if choose_model == "Model 1 (Fire or No fire)": # ------------ (from up) THIS should be same 
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]
# elif choose_model == "Model 2 (11 food classes)":
#     CLASSES = classes_and_models["model_2"]["classes"]
#     MODEL = classes_and_models["model_2"]["model_name"]
# else:
#     CLASSES = classes_and_models["model_3"]["classes"]
#     MODEL = classes_and_models["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    if (session_state.pred_conf > 0.8):
        st.write(f"Prediction: {session_state.pred_class}, \
                Confidence: {session_state.pred_conf:.3f}")
    else:
        st.write("Can't predict accurately. However, an estimated prediction is: ")
        
        st.write(f"Prediction: {session_state.pred_class}, \
                Confidence: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass             
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()
