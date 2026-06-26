import streamlit as st
import cv2
import requests
import mediapipe as mp
import requests
from PIL import Image
from io import BytesIO
import os
import random
import matplotlib.pyplot as plt
from help_func import extract_landmark_features


# Dataset directory
DATASET_DIR = "D:/resources/Datasets/sign alphabets/asl_alphabet_train"

#configeration of page
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)



# Navigation state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'


#  MediaPipe Hands 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

#Side_Bar
st.sidebar.markdown("""
    <h1 style='font-size:30px;'> Navigation Menu 🧐</h1>
""", unsafe_allow_html=True)

if st.sidebar.button("🤚Sign → Text"):
    st.session_state['page'] = 'sign_to_text'

if st.sidebar.button("📝Text → Sign"):
    st.session_state['page'] = 'text_to_sign'

if st.sidebar.button("📊About Dataset"):
    st.session_state['page'] = 'About_Dataset'

# Style buttons of sidebar by CSS
st.sidebar.markdown("""
    <style>
    div.stButton > button {
        font-size: 25px;
        height: 50px;
        width: 200px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)


#  HOME Page  
if st.session_state['page'] == 'home':
     st.markdown("<h1 style='text-align: center; font-size: 50px;'>Sign Language Translator 🤟</h1>",
                  unsafe_allow_html=True) 
     st.markdown(""" <p style='font-size:20px;'> <b>About the Project:</b><br> 
                                                      This project bridges the gap between English Sign Language and written text in an easy and interactive way. 
                 Users can perform hand signs to convert them into text or type a word and see its visual representation in sign language.
                 <br><br> <p style='font-size:20px;'> <b>Benefits:</b><br> - Facilitate communication with deaf and hard-of-hearing individuals<br> - Interactive educational tool for learning sign language<br>
                  - Convert text to sign language visuals for students and learners<br><br> <p style='font-size:20px;'> 
                 <b>Features:</b><br> - <b>Sign → Text:</b> Recognize hand gestures live <br> - <b>Text → Sign:</b> 
                 Convert written words into a sequence of sign language images </p> """,
                                                       unsafe_allow_html=True)
     st.markdown("---")

#  Sign → Text Page 
elif st.session_state['page'] == 'sign_to_text':
    st.markdown("<h2 style='font-size:40px;'>🤚Sign → Text</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:30px;'>📷Camera input with prediction:</h2>", unsafe_allow_html=True)

#checkbox for start camera
    run = st.checkbox("Start Camera")
    stframe = st.empty()    # Space for camera
    pred_box = st.empty()   # Prediction Box
    if run:
        cap = cv2.VideoCapture(0) #open camera if run =true

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) #flip image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert frame from BGR to RGB
            results = hands.process(rgb_frame)

            prediction_label = "No hand detected"

            # Extract Features from detected hand
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    features = extract_landmark_features(hand_landmarks)

                    response = requests.post(
                        "http://127.0.0.1:8000/sign_to_text",
                        json={"features": features.tolist()}
                    )
                    prediction_label = response.json().get("prediction")

                    # Show landmarks on hand
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            #  Display camera
            stframe.image(frame, channels="BGR")

            # Prediction Box
            pred_box.markdown(
                f"<h3 style='color: pink; text-align: center;'>Prediction: {prediction_label}</h3>", 
                unsafe_allow_html=True
            )
        #stop camera
        cap.release()

     # Back_home Button
    if st.button("Back to Home"):
       st.session_state['page'] = 'home'


# ---- Text → Sign Page ----
elif st.session_state['page'] == 'text_to_sign':
    st.title("👐 Sign Language Translator")

    user_input = st.text_input("Enter your sentence:")

  # Checkbox to enable/disable Spell & Grammar Correction
    enable_correction = st.checkbox("Enable Spell & Grammar Correction", value=True)

    if st.button("Translate to Sign"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            try:
                api_url = "http://127.0.0.1:8000/text_to_sign_image/preview"
                
                # Send the text and correction option to the API
                response = requests.post(
                    api_url,
                    params={
                        "text": user_input,
                        "enable_correction": enable_correction
                    }
                )

                if response.status_code == 200:
                    if response.headers.get("content-type") == "image/png":
                        img_bytes = BytesIO(response.content)
                        img = Image.open(img_bytes)
                        st.image(img, caption="Sign Language Translation", use_container_width=True)

                        # Get file name from API response or default
                        content_disposition = response.headers.get("content-disposition")
                        if content_disposition:
                            file_name = content_disposition.split("filename=")[-1].strip('"')
                        else:
                            file_name = "sign_translation.png"

                        img_bytes.seek(0)
                        st.download_button(
                            label="⬇️ Download Sign Translation",
                            data=img_bytes,
                            file_name=file_name,
                            mime="image/png"
                        )
                    else:
                        st.error(f"API Error: {response.json()}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")


    if st.button("Back to Home"):
       st.session_state['page'] = 'home'


# ---- About Dataset Page ----
elif st.session_state['page'] == 'About_Dataset':
    st.title(" 🔠 ASL Dataset")
    
    st.subheader("About Dataset")
    st.markdown("""
    The data set is a collection of images of alphabets from the <b style='color:lightblue;'>American Sign Language</b>, separated in 29 folders which represent the various classes.
    """, unsafe_allow_html=True)
    
    st.subheader("Content")
    st.markdown("""
    The training data set contains **87,000 images** which are 200x200 pixels.  
    There are **29 classes**, of which **26 are for the letters A-Z** and **3 classes for SPACE, DELETE and NOTHING**.
    These **3 classes** are very helpful in real-time applications and classification.  
    Each letter from **A-Z** has a folder containing approximately **3000 images** with different lighting conditions and various skin tones.  
    The test data set contains a mere **29 images**, to encourage the use of real-world test images.
    """)
    
    st.title("Data Visualization")
    letters = sorted(os.listdir(DATASET_DIR))  

    st.subheader("One Sample Image per Letter")
    cols_per_row = 5  # Number of columns

    for i in range(0, len(letters), cols_per_row):
        row_letters = letters[i:i+cols_per_row]  
        cols = st.columns(len(row_letters))
    
        for col, letter in zip(cols, row_letters):
            folder = os.path.join(DATASET_DIR, letter)
            img_file = random.choice(os.listdir(folder))  
            img = Image.open(os.path.join(folder, img_file))
            with col:
                st.image(img, caption=letter, use_container_width=True)
    
    # Count number of images per class
    classes = sorted(os.listdir(DATASET_DIR))  
    counts = [len(os.listdir(os.path.join(DATASET_DIR, c))) for c in classes]

    st.title("ASL Dataset Balance Analysis")

    # Bar Chart
    st.subheader("Bar Chart of Images per Class")
    plt.figure(figsize=(12,6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Letters / Classes")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Class")
    st.pyplot(plt)

    # Pie Chart
    st.subheader("Pie Chart of Images per Class")
    plt.figure(figsize=(8,8))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    plt.title("Proportion of Images per Class")
    st.pyplot(plt)

    # Evaluate dataset balance
    min_count = min(counts)
    max_count = max(counts)
    if max_count - min_count <= 100:  
        st.success("The dataset is approximately balanced.")
    else:
        st.warning("The dataset is imbalanced. Some classes have more images than others.")

    if st.button("Back to Home"):
       st.session_state['page'] = 'home'
