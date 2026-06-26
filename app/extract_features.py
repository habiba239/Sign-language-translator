import mediapipe as mp
import os
import pandas as pd
import numpy as np
import joblib
import cv2

from help_func import extract_landmark_features


#for counting num of undetected images
counter = 0

#prepare module for mediapipe of hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# path of dataset
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "asl_alphabet_train")
data = []

#loop on directories in dataset
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)#create label path
    if not os.path.isdir(label_path):#check label path
        continue

    #loop on image to detect is available or not
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file) 
        image = cv2.imread(img_path)
        if image is None:
            print(f"image is not available: {img_path}")
            continue

        #convert images from BGR to RGB as mediapipe deal with RGB imaes
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #results keep landmarks(x, y, z) which detected from each images
        results = hands.process(image_rgb)

        
        if results.multi_hand_landmarks: #normalized landmarks 
            for hand_landmarks in results.multi_hand_landmarks:
                coords = extract_landmark_features(hand_landmarks).tolist() #extract all features as list
                coords.append(label) # add the label (alphapet) at the end of the list
                data.append(coords) # add coords to be as list in a data list which be converted to .csv
        else:
            counter += 1  # to check num of undetected hand 

# create columns of csv 
columns = []
for i in range(21): # cols of 21 landmarks
    columns.append(f"x{i+1}")
    columns.append(f"y{i+1}")
    columns.append(f"z{i+1}")

for j in range(3):   # cols of angels
    columns.append(f"angle{j+1}")

for k in range(2):   # cols of distances
    columns.append(f"dist{k+1}")

columns.append("label")

# save data to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv("asl_landmarks.csv", index=False)

print("Done.csv")
#print num of undetected hands
print(f"no of undetected hand {counter}") 
