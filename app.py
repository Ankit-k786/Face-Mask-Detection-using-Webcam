from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import webbrowser
from keras.models import load_model
from utils import process_frames
app = Flask(__name__)


def load_models():
    prototxtPath = "./models/deploy.prototxt"
    weightsPath = "./models/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("./models/model.h5")
    return faceNet, maskNet


def feed_frames():
    global camera
    global faceNet
    global maskNet
    while True:
        try:
            success, frame = camera.read()
            if success:
                frame = process_frames(frame,faceNet,maskNet)
                frame=cv2.resize(frame,(640,480))
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + black_frame + b'\r\n')
    
        except:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + black_frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(feed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        global switch
        global camera
        if request.form.get('stop') == 'Stop/Start':
            if(switch == 1):
                switch = 0
                if camera is not None:
                    camera.release()
                    cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


# main driver function
if __name__ == '__main__':
    global switch
    global faceNet
    global maskNet
    global camera
    global black_frame
    camera=None
    black_frame=np.zeros((480,640))
    __, black = cv2.imencode('.jpg', black_frame)
    black_frame = black.tobytes()
    switch=0
    faceNet, maskNet=load_models()
    url = 'http://127.0.0.1:5000'
    webbrowser.open_new(url)
    app.run()
