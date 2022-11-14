import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from twilio.rest import Client
from playsound import playsound
from decouple import config

  
message_sent = False

model = load_model("./model.h5")

video = cv2.VideoCapture("fire.mp4")

name = ["No fire", "Fire Detected"]


def send_message():
	account_sid = config("ACCOUNT_SID")
	auth_token = config("AUTH_TOKEN")
	
	client = Client(account_sid, auth_token)
	message = client.messages.create(
		body="Forest Fire detected , Stay safe!!!",
		from_=config("FROM"),
		to=config("TO")
	)
	print(message.sid)
	print("Fire Detected")
	print("SMS Sent!")
    
playsound("./beep.mp3")	


while True:
	success, frame = video.read()
	cv2.imwrite("image.jpg", frame)
	img = image.load_img("image.jpg", target_size=(128, 128))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	pred = model.predict(x)
	p = int(pred[0][0])
	cv2.putText(frame, str(name[p]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
	
	if p == 1:
		if not message_sent:
			send_message()
			message_sent = True
		print("Fire Detected , stay safe!!!")
	else:
		print("No Fire Detected")
	
	cv2.imshow("Image", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('x'):
		break

video.release()
cv2.destroyAllWindows()