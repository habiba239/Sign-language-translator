from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


import pandas as pd
import numpy as np
import joblib



# read new dataset from csv
df = pd.read_csv("asl_landmarks.csv")   

X = df.drop(columns=["label"]).values #features
y_text = df["label"].values  #target

# Encode on target
le = LabelEncoder()
y = le.fit_transform(y_text)

# Split data with startify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)# do fit and transform
X_test_sc  = scaler.transform(X_test) # do transform only on test to prevent data leakage

#  MLP Model
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),    
    activation='relu',              #activation function
    solver='adam',
    alpha=0.01,                    # regularization (L2) parameter
    max_iter=200,                   # iterations 
    early_stopping=True,           
    n_iter_no_change=50,           
    random_state=42
)

# fit model
mlp.fit(X_train_sc, y_train)

# Evaluation_model
y_train_pred = mlp.predict(X_train_sc)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))

y_test_pred = mlp.predict(X_test_sc)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

#Cross_validation:
# Cross-validation on full dataset
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(mlp, scaler.fit_transform(X), y, cv=cv, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean CV accuracy:", np.mean(scores))
print("Std deviation:", np.std(scores))


# save model , encoding , scaling
joblib.dump(mlp,   "asl_mlp.joblib")
joblib.dump(scaler,"asl_scaler.joblib")
joblib.dump(le,    "asl_label_encoder.joblib")
###############################
# results of training Model and Cross_validation
#Train Accuracy: 0.9985669414998037
#Test Accuracy: 0.9929334170854272
#Cross-validation scores: [0.99298031 0.99151943 0.99316843]
#Mean CV accuracy: 0.9925560584168486
#Std deviation: 0.0007370162967705016