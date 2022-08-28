import cv2
import numpy as np
import face_detect as face_detect
import training_data as training_data

label = []
def predict(test_img):
	img = cv2.imread(test_img).copy()
	print("\n\n\n")
	print("Face Prediction Running -\-")
	face, rect, length = face_detect.face_detect(test_img)
	print(len(face), "faces detected.")
	for i in range(0, len(face)):
		labeltemp, confidence = face_recognizer.predict(face[i])
		label.append(labeltemp)
	return img, label

faces, labels = training_data.training_data("training-data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


# Read the test image.
test_img = "test-data/test1.jpg" 	# change test image here <-----------
predicted_img , label= predict(test_img)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
print("Recognized faces = ", label)

#webCam
# faceCascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)

# # For each person, enter one numeric face id
# face_id = input('\n enter user id end press <return> ==>  ')
# print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# count = 0

# while True:
# 	ret, frame = cap.read()
# 	frame = cv2.flip(frame, 1)
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 	faces = faceCascade.detectMultiScale(
#         gray,     
#         scaleFactor=1.3,
#         minNeighbors=5,     
#         minSize=(20, 20)
#     )

# 	for (x,y,w,h) in faces:
# 		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
# 		roi_gray = gray[y:y+h, x:x+w]
# 		roi_color = frame[y:y+h, x:x+w]
# 		count += 1

#         # Save the captured image into the datasets folder
# 		cv2.imwrite("dataset/User." + str(face_id) + '.' +  
#                     str(count) + ".jpg", gray[y:y+h,x:x+w])

# 	cv2.imshow('video', frame)

# 	k = cv2.waitKey(100) & 0xff
# 	if k == 27:
# 		break
# 	elif count >= 30:
# 		break

# print("\n [INFO] Exiting Program and cleanup stuff")
# cap.release()
# cv2.destroyAllWindows()
