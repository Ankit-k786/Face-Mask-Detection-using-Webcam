from keras.applications.mobilenet_v2 import preprocess_input
import keras.utils as image
import numpy as np
import cv2


def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    print(detections)
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
		
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = image.img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    if len(faces) > 0:
        for f in faces:
            pred = maskNet.predict(f)
            preds.append(pred[0])
        
    return (locs, preds)






def process_frames(frame,faceNet, maskNet):
    frame = cv2.resize(frame,(1080,(frame.shape)[1]))
    (locs,preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (startX+20, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX+15,startY-10), cv2.FONT_HERSHEY_DUPLEX , 1, color)
    return frame















