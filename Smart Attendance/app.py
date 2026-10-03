from flask import Flask, render_template, request, jsonify
import cv2
import os
import csv
import base64
import numpy as np
from datetime import datetime

app = Flask(__name__)

# =================================================
# PATHS
# =================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENTS_FILE = os.path.join(
    BASE_DIR,
    "students.csv"
)

CASCADE_FILE = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "trainer",
    "trainer.yml"
)

ATTENDANCE_FOLDER = os.path.join(
    BASE_DIR,
    "attendance"
)

ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "attendance.csv"
)

# Create attendance folder
os.makedirs(
    ATTENDANCE_FOLDER,
    exist_ok=True
)


# =================================================
# LOAD STUDENTS
# =================================================

students = {}

if os.path.exists(STUDENTS_FILE):

    with open(
        STUDENTS_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        label = 1

        for row in reader:

            student_id = row["ID"].strip()
            student_name = row["Name"].strip()

            students[label] = {
                "id": student_id,
                "name": student_name
            }

            label += 1


print("================================")
print("Students loaded:")

for label, data in students.items():

    print(
        f"Label {label} -> "
        f"ID: {data['id']}, "
        f"Name: {data['name']}"
    )

print("================================")


# =================================================
# CREATE ATTENDANCE CSV
# =================================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Name",
            "Date",
            "Time",
            "Status"
        ])


# =================================================
# MARK ATTENDANCE
# ONE STUDENT = ONE ATTENDANCE PER DAY
# =================================================

def mark_attendance(student_id, name):

    today = datetime.now().strftime(
        "%d-%m-%Y"
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )


    # ---------------------------------------------
    # CHECK EXISTING ATTENDANCE
    # ---------------------------------------------

    if os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                existing_id = row["ID"].strip()
                existing_date = row["Date"].strip()

                if (
                    existing_id == str(student_id)
                    and existing_date == today
                ):

                    print(
                        "Already Present Today:",
                        student_id,
                        name
                    )

                    return False


    # ---------------------------------------------
    # ADD NEW ATTENDANCE
    # ---------------------------------------------

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            student_id,
            name,
            today,
            current_time,
            "Present"
        ])


    print(
        "Attendance Marked:",
        student_id,
        name
    )

    return True


# =================================================
# LOAD HAAR CASCADE
# =================================================

face_detector = cv2.CascadeClassifier(
    CASCADE_FILE
)

if face_detector.empty():

    print(
        "ERROR: Haar Cascade could not be loaded."
    )

else:

    print(
        "Haar Cascade loaded successfully."
    )


# =================================================
# LOAD TRAINED MODEL
# =================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_loaded = False

if os.path.exists(MODEL_FILE):

    try:

        recognizer.read(
            MODEL_FILE
        )

        model_loaded = True

        print(
            "Trained model loaded successfully."
        )

    except Exception as e:

        print(
            "Model loading error:",
            e
        )

else:

    print(
        "ERROR: trainer.yml not found."
    )


# =================================================
# HOME PAGE
# =================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        students=students
    )


# =================================================
# FACE RECOGNITION API
# =================================================

@app.route(
    "/recognize",
    methods=["POST"]
)
def recognize():

    if not model_loaded:

        return jsonify({

            "success": False,

            "message":
            "Trained model not found."
        })


    try:

        # -----------------------------------------
        # GET JSON DATA
        # -----------------------------------------

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "message":
                "No data received."
            })


        if "image" not in data:

            return jsonify({

                "success": False,

                "message":
                "Image data not received."
            })


        image_data = data["image"]


        # -----------------------------------------
        # REMOVE BASE64 HEADER
        # -----------------------------------------

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        # -----------------------------------------
        # DECODE IMAGE
        # -----------------------------------------

        image_bytes = base64.b64decode(
            image_data
        )

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({

                "success": False,

                "message":
                "Invalid image."
            })


        # -----------------------------------------
        # CONVERT TO GRAYSCALE
        # -----------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # -----------------------------------------
        # DETECT FACES
        # -----------------------------------------

        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.3,

            minNeighbors=5,

            minSize=(100, 100)
        )


        if len(faces) == 0:

            return jsonify({

                "success": True,

                "recognized": False,

                "message":
                "No face detected."
            })


        results = []


        # -----------------------------------------
        # PROCESS EACH FACE
        # -----------------------------------------

        for (
            x,
            y,
            w,
            h
        ) in faces:


            face = gray[
                y:y + h,
                x:x + w
            ]


            # -------------------------------------
            # FACE RECOGNITION
            # -------------------------------------

            predicted_label, confidence = (

                recognizer.predict(face)

            )


            print("--------------------------------")
            print(
                "Predicted Label:",
                predicted_label
            )
            print(
                "LBPH Distance:",
                confidence
            )
            print(
                "Students:",
                students
            )
            print("--------------------------------")


            # -------------------------------------
            # FIND STUDENT FROM LABEL
            # -------------------------------------

            student = students.get(
                predicted_label
            )


            # -------------------------------------
            # RECOGNITION THRESHOLD
            # -------------------------------------

            if (
                student is not None
                and confidence < 70
            ):


                student_id = student["id"]

                student_name = student["name"]


                # ---------------------------------
                # MARK ATTENDANCE
                # ---------------------------------

                marked = mark_attendance(

                    student_id,

                    student_name

                )


                if marked:

                    status = (
                        "Attendance Marked"
                    )

                else:

                    status = (
                        "Already Marked Today"
                    )


                results.append({

                    "id":
                    student_id,

                    "name":
                    student_name,

                    "confidence":
                    round(
                        confidence,
                        2
                    ),

                    "status":
                    status
                })


            else:


                # ---------------------------------
                # UNKNOWN FACE
                # ---------------------------------

                results.append({

                    "id":
                    "",

                    "name":
                    "Unknown",

                    "confidence":
                    round(
                        confidence,
                        2
                    ),

                    "status":
                    "Face not recognized"
                })


        # -----------------------------------------
        # SEND RESULT TO WEBSITE
        # -----------------------------------------

        return jsonify({

            "success": True,

            "recognized": True,

            "results": results

        })


    except Exception as e:

        print(
            "Recognition error:",
            e
        )


        return jsonify({

            "success": False,

            "message":
            str(e)

        })


# =================================================
# RUN FLASK
# =================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )