import numpy as np
import cv2
import tensorflow as tf, sys
import requests
import time



#				initialize openCV stuff 
cam = cv2.VideoCapture("IN_cam 08_31_09-08_34_52.mp4")

# 				initialize tensorflow classifier

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels1.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph1.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

#my_list=[]
#now all we need to do is give tensorflow the image data
##first lets create the image data and then we shall use the tensorflow code to run it on the dataset
sampleNum = 0
while (True):
    
	ret, img = cam.read()
	if sampleNum%5==0:
   
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	       
		cv2.imwrite("car/rep."+str(id)+"."+str(sampleNum)+".jpg", gray)
		samleNum=sampleNum+1   
		
		#the images have been stores in dataSet, now we shall get the images and do a classification
		image_path = "car/rep."+str(id)+"."+str(sampleNum)+".jpg"
		#todo: import images from the dataSet fil
		image_data = tf.gfile.FastGFile(image_path, 'rb').read()
		with tf.Session() as sess:
			# Feed the image_data as input to the graph and get first prediction
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			#for node_id in top_k:
			animal_string = label_lines[top_k[0]]
			score = predictions[0][top_k[0]]
			if ((score >= 0.9) or (score>=0.8)) :
				print('%s (score = %.5f)' % (animal_string, score))
				#font = cv2.FONT_HERSHEY_SIMPLEX
				#cv2.putText(img, "Alert! "+animal_string, (0,200), font, 4, (0,0,255), 20)
			elif (score <=0.3):
				print("unknown_reptiles") 
			cv2.imwrite("unknown/rep."+str(id)+"."+str(sampleNum)+".jpg", gray)
			
        	
        
	#print(my_list)    
	cv2.namedWindow('Car_detection', cv2.WINDOW_NORMAL)
	cv2.imshow('_detection',img);
	k = cv2.waitKey(1)
	if k == ord('x'):
		break
    
cam.release()
cv2.destroyAllWindows()

	
	
	
	


























