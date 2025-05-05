import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Directories - update these paths to where your dataset is located
yawn_dir = 'Dataset/yawn'
no_yawn_dir = 'Dataset/no yawn'

# Parameters
img_size = (64, 64)  # Resize images to 64x64
X, y = [], []

# Load and preprocess images, extract HOG features
def load_images(directory, label):
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            # Extract HOG features
            features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
            X.append(features)
            y.append(label)

# Load images from both classes
load_images(yawn_dir, 1)      # Label 1 for yawning
load_images(no_yawn_dir, 0)   # Label 0 for not yawning

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model to a file for later use
joblib.dump(clf, 'yawning_svm_model.joblib')
print("Model saved as yawning_svm_model.joblib")
