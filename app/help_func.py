import os
import pandas as pd
import numpy as np

# help_functions
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def extract_landmark_features(hand_landmarks):
    #extract coordinates (x, y, z)
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
        coords.append(lm.z)

    # tuple (x, y) of each landmark of image
    landmarks_xy = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    # calculate angels 
    angles = []
    angles.append(calculate_angle(landmarks_xy[0], landmarks_xy[1], landmarks_xy[2]))
    angles.append(calculate_angle(landmarks_xy[5], landmarks_xy[6], landmarks_xy[7]))
    angles.append(calculate_angle(landmarks_xy[9], landmarks_xy[10], landmarks_xy[11]))

    # calculate distances
    distances = []
    distances.append(calculate_distance(landmarks_xy[4], landmarks_xy[8]))
    distances.append(calculate_distance(landmarks_xy[8], landmarks_xy[12]))

    coords.extend(angles)
    coords.extend(distances)

    return np.array(coords)