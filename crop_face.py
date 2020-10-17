import cv2
import os
import sys
import dlib


def extract_frames(video_Path='actor_1.mp4',save_path='Face_dir/PersonA'):
	cap = cv2.VideoCapture(video_Path)
	n=0
	# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	# out = cv2.VideoWriter('liu_out.avi', fourcc, 10, (frame_width, frame_height))
	while(cap.isOpened()):
		try:
			ret, frame = cap.read()

			#frame = frame.reshape(frame.shape[1],frame.shape[0],3)
			if n%2==0:
				frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
				print("extracting from",save_path.split('/')[-1])
				save_images = os.path.join(save_path, str(n)+'.jpg')
				cv2.imwrite(save_images, frame)
			n = n +1
		except Exception as r:
			print(r)
			print("error in ",video_Path)

	cap.release()
	out.release()
	cv2.destroyAllWindows()

def rotate(img):
	rows,cols,_=img.shape
	M=cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
	dst=cv2.warpAffine(img, M, (cols, rows))

	return dst

def extract_face(Images_Folder,OutFace_Folder):

	Images_Path = os.path.join(os.path.realpath('.'), Images_Folder)
	pictures = os.listdir(Images_Path)
	detector=dlib.get_frontal_face_detector()
	print(pictures)

	for f in pictures:
		try:
			img=cv2.imread(os.path.join(Images_Path,f), cv2.IMREAD_COLOR)
			b,g,r=cv2.split(img)
			img2=cv2.merge([r,g,b])
			img=rotate(img)
			dets=detector(img,1)
			
			for idx, face in enumerate(dets):
				# print('face{}; left{}; top {}; right {}; bot {}'.format(idx, face.left(). face.top(), face.right(), face.bottom()))

				left = face.left()
				top = face.top()
				right = face.right()
				bot = face.bottom()
				#print(left, top, right, bot)
				#cv2.rectangle(img, (left, top), (right, bot), (0, 255, 0), 3)
				#print(img.shape)
				crop_img = img[top:bot, left:right]
				#cv2.imshow(f, img)
				#cv2.imshow(f, crop_img)
				cv2.imwrite(OutFace_Folder+f[:-4]+"_face.jpg", crop_img)
				#k = cv2.waitKey(1000)
		except Exception as e:
			#cv2.destroyAllWindows()
			print(f)
			print(e)



# extract_frames()
# extract_frames('actor_2.mp4','Face_dir/PersonB')
extract_face('Face_dir/PersonA','Face_dir/PersonA/face/')
extract_face('Face_dir/PersonB','Face_dir/PersonB/face/')