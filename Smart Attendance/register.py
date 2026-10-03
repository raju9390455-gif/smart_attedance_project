import cv2
import os
import csv

# -----------------------------
# Load Face Detector
# -----------------------------

cascade_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml"
)

print("Cascade path:", cascade_path)
print("File exists:", os.path.isfile(cascade_path))

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Face detector XML could not be loaded.")
    exit()

print("Face detector loaded successfully.")


# -----------------------------
# Get Student Details
# -----------------------------

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")
students_file = "students.csv"

# Create students.csv if it doesn't exist
if not os.path.exists(students_file):

    with open(students_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["ID", "Name"])


# Save student details
with open(students_file, "a", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        student_id,
        student_name
    ])

print("Student details saved.")


# -----------------------------
# Create Dataset Folder
# -----------------------------

os.makedirs("dataset", exist_ok=True)


# -----------------------------
# Start Camera
# -----------------------------

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("ERROR: Could not open camera.")
    exit()

count = 0

print("\nCamera started.")
print("Look at the camera.")
print("Move your face slightly left and right.")
print("Press Q to stop.\n")


# -----------------------------
# Capture Face Images
# -----------------------------

while True:

    ret, frame = cam.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        count += 1

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Save face image
        filename = f"dataset/User.{student_id}.{count}.jpg"

        cv2.imwrite(
            filename,
            gray[y:y + h, x:x + w]
        )

        # Display image count
        cv2.putText(
            frame,
            f"Images: {count}/30",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Student Registration", frame)

    key = cv2.waitKey(100) & 0xFF

    # Press Q to stop
    if key == ord("q"):
        break

    # Automatically stop after 30 images
    if count >= 30:
        break


# -----------------------------
# Close Camera
# -----------------------------

cam.release()
cv2.destroyAllWindows()

print("\nFace registration completed.")
print("Total images captured:", count)
print("Student ID:", student_id)
print("Student Name:", student_name)