import os
import torch
import cv2
import time
import threading
import queue
from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import pyttsx3  # TTS library for audio feedback
import pygame  # For playing MP3 files

# Setup Flask
app = Flask(__name__)

# Load YOLOv5 Model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

# Initialize TTS Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speaking rate for clarity

# Initialize Pygame mixer for MP3 playback
pygame.mixer.init()

# Define a cooldown period (in seconds)
cooldown_period = 5  # Adjust as necessary

# Track the last time and type of feedback given
last_feedback_time = 0
last_detected_type = None

# Fungsi untuk memutar suara berdasarkan label
def play_sound(label):
    if label == "Casual":
        pygame.mixer.music.load('static/assets/sounds/casual.mp3')
        pygame.mixer.music.play()
    elif label == "Formal":
        pygame.mixer.music.load('static/assets/sounds/formal.mp3')
        pygame.mixer.music.play()
    elif label == "":
        pygame.mixer.music.load('static/assets/sounds/notfound.mp3')
        pygame.mixer.music.play()

# Fungsi untuk memberi feedback suara melalui TTS
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Fungsi untuk memproses file gambar atau video
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(filepath)

            # Check file extension
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            # Process image (JPG)
            if file_extension == 'jpg':
                print(f"Processing image: {f.filename}")
                results = model(filepath)
                detected_labels = [label['name'] for label in results.pandas().xyxy[0].to_dict(orient="records")]
                print(f"Detected labels: {detected_labels}")
                
                # Provide audio feedback for detected labels
                for label in detected_labels:
                    if "Formal" in label:  # Example: trigger formal sound
                        print(f"Formal detected: {label}")
                        play_sound("Formal")
                        # speak(f"Formal item detected: {label}")
                    elif "Casual" in label:  # Example: trigger casual sound
                        print(f"Casual detected: {label}")
                        play_sound("Casual")
                        # speak(f"Casual item detected: {label}")
                    elif "" in label:
                        play_sound("")

                results.save()  # Save detection results
                return display(f.filename)

            # Process video (MP4)
            elif file_extension == 'mp4':
                print(f"Processing video: {f.filename}")
                cap = cv2.VideoCapture(filepath)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame)
                    res_plotted = results.render()[0]
                    out.write(res_plotted)

                    # Provide audio feedback if formal or casual detected
                    detected_labels = [label['name'] for label in results.pandas().xyxy[0].to_dict(orient="records")]
                    print(f"Detected labels in video frame: {detected_labels}")
                    for label in detected_labels:
                        if "Formal" in label:
                            print(f"Formal detected in video: {label}")
                            play_sound("Formal")
                        elif "Casual" in label:
                            print(f"Casual detected in video: {label}")
                            play_sound("Casual")
                        elif "" in label:
                            play_sound("")
                            

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()

        return render_template('index.html')

    return render_template('index.html')


@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    return send_from_directory(directory, filename)

# Function to capture and process webcam frames
def get_frame():
    global last_feedback_time, last_detected_type

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam!")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame!")
            break

        # Perform detection on the frame
        results = model(frame)
        res_plotted = results.render()[0]

        # Extract detected labels
        detected_labels = [label['name'] for label in results.pandas().xyxy[0].to_dict(orient="records")]
        print(f"Label terdeteksi di webcam: {detected_labels}")

        # Get current time
        current_time = time.time()

        # Check for cooldown
        for label in detected_labels:
            if ("formal" in label.lower() or "casual" in label.lower()):
                # Only give feedback if cooldown period has passed or if the label has changed
                if current_time - last_feedback_time > cooldown_period or last_detected_type != label:
                    last_feedback_time = current_time
                    last_detected_type = label

                    # Play sound based on detection type
                    if "formal" in label.lower():
                        print(f"Formal terdeteksi di webcam: {label}")
                        play_sound("Formal")
                    elif "casual" in label.lower():
                        print(f"Casual terdeteksi di webcam: {label}")
                        play_sound("Casual")
                    elif "" in label:
                        play_sound("")
        
        # Encode and yield the frame for streaming
        ret, jpeg = cv2.imencode('.jpg', res_plotted)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(0.1)  # Small delay between frames to manage frame rate

    cap.release()
# Video streaming route
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
