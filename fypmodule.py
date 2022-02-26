# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FileVideoStream
import face_recognition
import pickle
from centrioddetector.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from time import sleep
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-x", "--maskmodel", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
counter = 0

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []
	
	for i in range(0, detections.shape[2]):
		if detections[0, 0, i, 2] > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)


	objects = ct.update(rects)

	
	for (objectID, centroid) in objects.items():
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		if counter == objectID:
			cv2.putText(frame, 'Capturing image', (0,0+20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			print('Taking Screenshot and value of counter is {}'.format(counter))
			sleep(1)
			myimage= vs.read()
			path = '/home/pi/Downloads/Colab Notebooks/Face detector/fypimages'
			cv2.imwrite(os.path.join(path , 'image{}.jpg'.format(objectID)), myimage)
			counter+= 1
		else:
			continue
		
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(args["prototxt"], args["model"])
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["maskmodel"])
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

images = []
imagePaths = '/home/pi/Downloads/Colab Notebooks/Face detector/fypimages/'
images = load_images_from_folder(imagePaths)
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
net.setInput(blob)
detections = net.forward()
imagePaths = sorted(list(paths.list_images(imagePaths)))

for (i, p) in enumerate(imagePaths[0:]):
	# load the image from disk
	print("Opening {}".format(p))
	image = cv2.imread(p)
	if image is None:
		break
	orig = image.copy()
	(h, w) = image.shape[:2]
	print("[INFO] computing face detections...")
	

	blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
	net.setInput(blob)
	detections = net.forward()
	#image = cv2.imread(p)

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))    
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			(mask, withoutMask) = maskNet.predict(face)[0]
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			#print("I am here")
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(image, width=750)
	r = image.shape[1] / float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	print("Comparing Faces")
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
	
		names.append(name)
	for ((top, right, bottom, left), name) in zip(boxes, names):
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		cv2.rectangle(image, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
		cv2.imwrite('/home/pi/Downloads/Colab Notebooks/Face detector/outimage/image{}.jpg'.format(counter),image)
		counter = counter-1
		#cv2.destroyAllWindows()
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)
	

import smtplib
import imghdr
from email.message import EmailMessage
EMAIL_ADDRESS = 'bce2k1721@gmail.com'
EMAIL_PASSWORD = 'bce12345'
msg = EmailMessage()
msg['Subject'] = 'Embedded-System Based Monitoring through DNNs'
msg['From'] = EMAIL_ADDRESS
msg['To'] = EMAIL_ADDRESS
msg.set_content('Challan this person')

with open('outimage/image0.jpg','rb') as f:
	file_data = f.read()
	file_type = imghdr.what(f.name)
	file_name = f.name
          
msg.add_attachment(file_data, maintype='image', subtype = file_type, filename = file_name)
with smtplib.SMTP_SSL('smtp.gmail.com', 587) as smtp:
	smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
	smtp.send_message(msg)
    
cv2.destroyAllWindows()
vs.stop()