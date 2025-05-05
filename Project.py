import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from threading import Thread
from pygame import mixer
from skimage.feature import hog
import joblib

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yawn_model = joblib.load("yawning_svm_model.joblib")
  # Your trained model

# Constants
EAR_THRESHOLD = 0.25
FRAME_CHECK = 50
MAR_THRESHOLD = 0.7
YAWN_CHECK = 30

# Globals
flag = 0
yawn_count = 0
alert_playing = False

# Initialize mixer
mixer.init()

# EAR calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])  # 14-20
    B = np.linalg.norm(mouth[14] - mouth[18])  # 15-19
    C = np.linalg.norm(mouth[15] - mouth[17])  # 16-18
    D = np.linalg.norm(mouth[12] - mouth[16])  # 13-17
    return (A + B + C) / (3.0 * D)

# Play alert
def sound_alert(msg):
    global alert_playing
    if not alert_playing:
        alert_playing = True
        mixer.music.load("music.wav")
        mixer.music.play()
        messagebox.showwarning("ALERT", msg)

# Detection function
def start_detection(status_var, status_color_label, progress_bar):
    global flag, yawn_count, alert_playing
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # Face box
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[36:42]
            rightEye = shape[42:48]
            mouth = shape[48:68]

            # Eye contours
            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

            # Mouth contours
            cv2.polylines(frame, [mouth], True, (255, 0, 0), 1)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            (mx, my, mw, mh) = cv2.boundingRect(np.array([mouth]))
            mouth_roi = gray[my:my + mh, mx:mx + mw]

            try:
                if mouth_roi.size == 0 or mouth_roi.shape[0] < 10 or mouth_roi.shape[1] < 10:
                    raise ValueError("Invalid mouth ROI")
                mouth_resized = cv2.resize(mouth_roi, (64, 64)).astype("float32") / 255.0
                features = hog(mouth_resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False, channel_axis=None).reshape(1, -1)
                yawn_pred = yawn_model.predict(features)[0]
            except:
                yawn_pred = 0

            if mar > MAR_THRESHOLD or yawn_pred == 1:
                yawn_count += 1
                if yawn_count >= YAWN_CHECK:
                    status_var.set("Yawning Detected!")
                    status_color_label.config(fg="orange")
                    progress_bar["value"] = 70
                    sound_alert("Yawning Detected")
                    continue
            else:
                yawn_count = 0

            if ear < EAR_THRESHOLD:
                flag += 1
                if flag >= FRAME_CHECK:
                    status_var.set("Drowsiness Detected!")
                    status_color_label.config(fg="red")
                    progress_bar["value"] = 100
                    sound_alert("Drowsiness Detected")
                    continue
            else:
                flag = 0
                alert_playing = False
                if not mixer.music.get_busy():
                    status_var.set("Monitoring...")
                    status_color_label.config(fg="green")
                    progress_bar["value"] = 30

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection in a new thread
def start_thread(status_var, status_color_label, progress_bar):
    Thread(target=start_detection, args=(status_var, status_color_label, progress_bar), daemon=True).start()

# GUI Setup
def setup_gui():
    root = tk.Tk()
    root.title("Driver Drowsiness & Yawning Detection")
    root.geometry("500x300")
    root.resizable(False, False)

    title = tk.Label(root, text="Driver Drowsiness & Yawning Detection", font=("Helvetica", 16, "bold"))
    title.pack(pady=20)

    status_var = tk.StringVar(value="Monitoring...")
    status_label = tk.Label(root, text="Status:", font=("Helvetica", 12))
    status_label.pack()
    status_color_label = tk.Label(root, textvariable=status_var, font=("Helvetica", 14, "bold"), fg="green")
    status_color_label.pack()

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", maximum=100)
    progress_bar.pack(pady=10)
    progress_bar["value"] = 30

    start_button = tk.Button(root, text="Start Detection", font=("Helvetica", 12), bg="blue", fg="white",
                             command=lambda: start_thread(status_var, status_color_label, progress_bar))
    start_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", font=("Helvetica", 12), bg="red", fg="white", command=root.quit)
    exit_button.pack()

    root.mainloop()

# Run the GUI
setup_gui()
