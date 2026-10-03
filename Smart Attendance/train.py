import cv2
import os
import numpy as np
from PIL import Image
import csv

# -----------------------------
# Paths
# -----------------------------
dataset_path = "dataset"
trainer_path = "trainer"

os.makedirs(trainer_path, exist_ok=True)

# -----------------------------
# Create LBPH Recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

# -----------------------------
# Face Detector
# -----------------------------
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if detector.empty():
    print("ERROR: Face detector XML could not be loaded.")
    exit()

# -----------------------------
# Student ID Mapping
# -----------------------------
students_file = "students.csv"

student_ids = []
student_names = []

if os.path.exists(students_file):

    with open(students_file, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            sid = row["ID"]
            name = row["Name"]

            if sid not in student_ids:
                student_ids.append(sid)
                student_names.append(name)

else:
    print("ERROR: students.csv not found.")
    exit()

# -----------------------------
# Training Data
# -----------------------------
face_samples = []
labels = []

print("Starting training...")

# -----------------------------
# Read Dataset
# -----------------------------
for filename in os.listdir(dataset_path):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        dataset_path,
        filename
    )

    # -----------------------------
    # Extract Student ID
    # -----------------------------
    parts = filename.split(".")

    if len(parts) < 4:
        print("Skipping:", filename)
        continue

    student_id = parts[1]

    # -----------------------------
    # Find Numeric Label
    # -----------------------------
    if student_id not in student_ids:

        print("Student ID not found:", student_id)
        print("Skipping:", filename)
        continue

    label = student_ids.index(student_id) + 1

    # -----------------------------
    # Read Image
    # -----------------------------
    try:

        pil_image = Image.open(
            image_path
        ).convert("L")

        image_numpy = np.array(
            pil_image,
            "uint8"
        )

    except Exception as e:

        print("Error reading:", filename)
        print(e)
        continue

    # -----------------------------
    # Detect Face
    # -----------------------------
    faces = detector.detectMultiScale(
        image_numpy,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:

        print("No face detected:", filename)
        continue

    # -----------------------------
    # Add Faces
    # -----------------------------
    for (x, y, w, h) in faces:

        face_samples.append(
            image_numpy[y:y + h, x:x + w]
        )

        labels.append(label)

print("\nFaces detected:", len(face_samples))
print("Labels:", set(labels))

# -----------------------------
# Check Training Data
# -----------------------------
if len(face_samples) == 0:

    print("\nERROR: No faces found in dataset.")
    print("Check your dataset images.")
    exit()

# -----------------------------
# Train
# -----------------------------
recognizer.train(
    face_samples,
    np.array(labels)
)

# -----------------------------
# Save Model
# -----------------------------
recognizer.write(
    os.path.join(
        trainer_path,
        "trainer.yml"
    )
)

print("\nTraining completed successfully!")
print("Model saved to: trainer/trainer.yml")

# -----------------------------
# Display Mapping
# -----------------------------
print("\nStudent mapping:")

for i, sid in enumerate(student_ids, start=1):

    print(
        f"Label {i} -> ID: {sid}, "
        f"Name: {student_names[i - 1]}"
    )