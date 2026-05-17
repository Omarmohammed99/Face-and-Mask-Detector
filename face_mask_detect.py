import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Fixed the path of the haarcascade_frontalface_alt2.xml file
cascPath = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
faceCascade = cv2.CascadeClassifier(cascPath)

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        super().__init__(**kwargs)

class CustomDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)


model = load_model("my_mask_detector.h5", custom_objects={
    'BatchNormalization': CustomBatchNormalization,
    'Dense': CustomDense
})
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=3,
                                         minSize=(50, 50),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)

    # Moved the prediction outside of the for loop so that it is only called once per frame
    if len(faces_list) > 0:
        faces_array = np.vstack(faces_list)
        preds = model.predict(faces_array)
    else:
        preds = []

    for (x, y, w, h), pred in zip(faces, preds):
        (mask_weared_incorrect, with_mask, without_mask) = pred
        
        if (with_mask > without_mask and with_mask > mask_weared_incorrect):
            label = "Mask Worn Properly :)"
            color = (0, 255, 0) 
        elif (without_mask > with_mask and without_mask > mask_weared_incorrect):
            label = "No Mask! (please wear)"
            color = (0, 0, 255)    
        else:
            label = "Wear Mask Properly!"
            color = (255, 140, 0)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label,
                                     max(with_mask, mask_weared_incorrect, without_mask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy the windows
cap.release()
cv2.destroyAllWindows()