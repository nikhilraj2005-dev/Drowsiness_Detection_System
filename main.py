
import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import threading

# Path of the downloaded face landmark model
MODEL_PATH = "face_landmarker.task"

# Eye landmark positions
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Blink detection settings
EAR_THRESHOLD = 0.20

# Blink counter
blink_count = 0
eyes_closed = False
CLOSED_EYE_TIME = 2.0

eye_closed_start = None
alarm_active = False


# Calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )


# Calculate Eye Aspect Ratio
def calculate_ear(eye_points):
    vertical_1 = calculate_distance(
        eye_points[1], eye_points[5]
    )

    vertical_2 = calculate_distance(
        eye_points[2], eye_points[4]
    )

    horizontal = calculate_distance(
        eye_points[0], eye_points[3]
    )

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# Configure MediaPipe
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

def play_alarm():
    global alarm_active

    while alarm_active:
        winsound.Beep(1000, 500)
        time.sleep(0.1)

# Start webcam
camera = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as detector:

    print("EAR Detection Started!")

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Unable to capture video")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(time.monotonic() * 1000)

        results = detector.detect_for_video(
            mp_image, timestamp
        )

        if results.face_landmarks:

            for face_landmarks in results.face_landmarks:

                # Convert left eye landmarks to pixel coordinates
                left_points = [
                    (
                        face_landmarks[i].x * width,
                        face_landmarks[i].y * height
                    )
                    for i in LEFT_EYE
                ]

                # Convert right eye landmarks to pixel coordinates
                right_points = [
                    (
                        face_landmarks[i].x * width,
                        face_landmarks[i].y * height
                    )
                    for i in RIGHT_EYE
                ]

                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_points)
                right_ear = calculate_ear(right_points)

                # Calculate average EAR
                average_ear = (left_ear + right_ear) / 2.0

                # Blink detection
                if average_ear < EAR_THRESHOLD:
                    if not eyes_closed:
                        eyes_closed = True
                else:
                    if eyes_closed:
                        blink_count += 1
                        eyes_closed = False

                # Continuous eye closure detection

                if average_ear < EAR_THRESHOLD:

                    if eye_closed_start is None:
                        eye_closed_start = time.time()

                    closed_duration = time.time() - eye_closed_start

                    if closed_duration >= CLOSED_EYE_TIME:

                        cv2.putText(
                            frame,
                            "WARNING! EYES CLOSED",
                            (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        if not alarm_active:
                            alarm_active = True

                            threading.Thread(
                                target=play_alarm,
                                daemon=True
                            ).start()

                else:
                    eye_closed_start = None
                    alarm_active = False

                # Display eye landmarks
                for point in left_points + right_points:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(
                        frame, (x, y), 3, (0, 255, 0), -1
                    )

                # Display EAR value
                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display eye status and blink count
        status = "Closed" if eyes_closed else "Open"
        cv2.putText(
            frame,
            f"Eye Status: {status}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Eye Aspect Ratio Detection", frame)

        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

