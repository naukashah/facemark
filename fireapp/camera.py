from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
from datetime import datetime
import os.path
import pyrebase

# from recognition import extract_embeddings
# from recognition import train_model
# load our serialized face detector model from disk
# protoPath = os.path.sep.join([settings.BASE_DIR, "face_detection_model/deploy.prototxt"])
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
# modelPath = os.path.sep.join([settings.BASE_DIR,"face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
# load the actual face recognition model along with the label encoder
recognizer = "output/recognizer.pickle"
le = "output/le.pickle"
try:
	recognizer = pickle.loads(open(recognizer, "rb").read())
	le = pickle.loads(open(le, "rb").read())
except:
	pass

dataset = "dataset"
user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

today = datetime.today()
# /Users/naukadhabalia/git/library-management/
attendance_csv = 'Attendance' + today.strftime(
    "%m_%d_%y") + '.csv'

config = {
    "apiKey": "AIzaSyBY3X-PB5Iziv3YE-S62jcthPgZbrDpHnc",
    "authDomain": "facemark-a0bef.firebaseapp.com",
    "databaseURL": "https://facemark-a0bef-default-rtdb.firebaseio.com/",
    "projectId": "facemark-a0bef",
    "storageBucket": "facemark-a0bef.appspot.com",
    "messagingSenderId": "1086140254646",
    "appId": "1:1086140254646:web:edfa35b1d2a6d5217d1917",
    "measurementId": "G-VD5E8YXKDJ"
}


firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()

class FaceDetect(object):

	def __init__(self):
		#extract_embeddings.embeddings()
		#train_model.model_train()
		# initialize the video stream, then allow the camera sensor to warm up
		self.vs = VideoStream(src=0).start()
		# start the FPS throughput estimator
		self.fps = FPS().start()

	def __del__(self):
		self.fps.stop()
		cv2.destroyAllWindows()

	def markAttendance(self,name):
		if os.path.isfile(attendance_csv):
			access = 'r+'
		else:
			access = 'w+'
		with open(attendance_csv, access) as f:
			myDataList = f.readlines()
			nameList = []
			for line in myDataList:
				entry = line.split(',')
				nameList.append(entry[0])
			if name not in nameList:
				now = datetime.now()
				dtString = now.strftime('%m/%d/%y %H:%M:%S')
				f.writelines(f'\n{name},{dtString}')
				data = database.child('data').child('attendance').get().val()
				newData = {"id": len(data), "name": name, "timestamp": dtString,
						"type": "Student"}
				database.child('data').child('attendance').child(len(data)).set(newData)	

	def get_frame(self):
		# grab the frame from the threaded video stream
		frame = self.vs.read()
		frame = cv2.flip(frame,1)

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()


		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.8:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				# print("predict", recognizer.predict(vec)[0])
				# print("decision", recognizer.decision_function(vec)[0])
				preds = recognizer.predict_proba(vec)[0]
				# print("preds --> ",preds)
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]


				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				if(proba * 100 > 90):
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 0, 255), 2)
					cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
					self.markAttendance(name)

		# update the FPS counter
		self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
	
	
