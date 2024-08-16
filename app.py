from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the webcam or video file
camera = cv2.VideoCapture(0)

# Load the YOLO model
model = YOLO('fire8-best.pt')

# Variable to track the color state for flashing
color_state = True

def generate_frames():
    global color_state
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model.predict(source=frame, conf=0.25)
            predictions = results[0]

            # Toggle the color state for flashing effect
            color_state = not color_state
            box_color = (0, 0, 255) if color_state else (0, 255, 0)

            # Draw bounding boxes on the frame
            for box, conf, cls in zip(predictions.boxes.xyxy, predictions.boxes.conf, predictions.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, box_color, 3)
                
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
