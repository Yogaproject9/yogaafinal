# # Importing all required libraries
# import mediapipe as mp
# import pickle
# import pandas as pd
# import cv2
# import os
# import numpy as np
# import math
# import time
# import multiprocessing
# from time import sleep
# from playsound import playsound

# wait = 15
# # wait = 5
# wait2 = 10
# t_wait = time.time()
# t_wait2 = time.time()
# # fial treepose flag
# fi_fl = 0
# c_knee = 0
# c_shoulder = 0
# c_elbow = 0
# posture_output = 0 


# # Mediapipe Drawing helpers
# mp_drawing = mp.solutions.drawing_utils 
# mp_holistic = mp.solutions.holistic 
# mp_pose = mp.solutions.pose

# # Reading data from .pkl file
# with open('yoga/yoga_posture_changed.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Giving value to the posture
# if posture_output <=3:
#     posture="vrukshasana_treepose"
# elif posture_output <=6:
#     posture="tricoaasana"
# # Function for calculating the angle
# def calc_angle(a,b,c):
#     a = np.array(a) # First
#     b = np.array(b) # Mid
#     c = np.array(c) # End
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = round(np.abs(radians*180.0/np.pi),2)
    
#     if angle >180.0:
#         angle = 360-angle
        
#     return angle


# # Function for calculating the distance
# def calculateDistance(x1,y1,x2,y2):
#     dist1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     dist = (dist1*100)
#     return dist

# # cap = cv2.VideoCapture(0)
# class VideoCamera(object):

#     def __init__(self):
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_holistic = mp.solutions.holistic
#         self.video = cv2.VideoCapture(0)
#        # self.video = cv2.VideoCapture('yoga/trim1_2.0x.mp4')

#     def __del__(self):
#         self.video.release()


#     def get_gray(self):
#         with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             global wait
#             global wait2
#             global t_wait
#             global t_wait2
#             global fi_fl
#             global c_knee
#             global c_shoulder
#             global c_elbow
#             global posture_output
#             # print("hello")
#             success, image = self.video.read()
               
#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#             results = holistic.process(image)
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             self.mp_drawing.draw_landmarks(
#                 image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
#             self.mp_drawing.draw_landmarks(
#                 image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
#             self.mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                
#                 # Export coordinates
#             try:
#                     # Extract Pose landmarks
#                     pose = results.pose_landmarks.landmark
#                     pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()) 
            
#                     # Concate rows
#                     row = pose_row
                    
#                     # Make Detections
#                     X = pd.DataFrame([row])
#                     yoga_posture_class = model.predict(X)[0]
#                     yoga_posture_prob = model.predict_proba(X)[0]

#                     # finding probability of posture
#                     proba= int(round(yoga_posture_prob[np.argmax(yoga_posture_prob)],2)*100)            
                    
#                     # to change the threshold of posture
#                     if ((yoga_posture_class==posture) and (proba>= 30)): 
#                         # Get status box
#                         cv2.rectangle(image, (0,0), (len(yoga_posture_class)*22,50), (245, 117, 16), -1)
                    
#                         # Display Class
#                         cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                         cv2.putText(image, yoga_posture_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
#                         # Display Probability
#                         cv2.putText(image, 'PROB.', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                         cv2.putText(image, str(round(yoga_posture_prob[np.argmax(yoga_posture_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                         #--------------------------------------------
#                         landmarks1 = results.pose_landmarks.landmark

#                         # Get coordinate
#                         right_hip = [landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value].y] 
#                         right_knee = [landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#                         right_ankle = [landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
#                         left_knee = [landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#                         left_ankle = [landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
#                         left_hip = [landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                         right_elbow = [landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                         right_shoulder = [landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                         left_elbow = [landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                         left_shoulder = [landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                         right_wrist = [landmarks1[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks1[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#                         left_wrist = [landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
#                         # Calculate angle
#                         angle_rknee = calc_angle(right_hip,right_knee,right_ankle)
#                         angle_lknee = calc_angle(left_hip,left_knee,left_ankle)
#                         angle_rshoulder = calc_angle(right_elbow,right_shoulder,right_hip)
#                         angle_lshoulder = calc_angle(left_elbow,left_shoulder,left_hip)
#                         angle_relbow = calc_angle(right_shoulder,right_elbow,right_wrist)
#                         angle_lelbow = calc_angle(left_shoulder,left_elbow,left_wrist)
                        
#                         #left foot start and end point 
#                         lf_st = tuple(np.multiply(left_knee,[640,480]).astype(int))
#                         lf_en = tuple(np.multiply(left_ankle,[640,480]).astype(int))

#                         #left shoulder start and end point 
#                         ls_st = tuple(np.multiply(left_shoulder,[640,480]).astype(int))
#                         ls_en = tuple(np.multiply(left_elbow,[640,480]).astype(int))

#                         #right shoulder start and end point 
#                         rs_st = tuple(np.multiply(right_shoulder,[640,480]).astype(int))
#                         rs_en = tuple(np.multiply(right_elbow,[640,480]).astype(int))

#                         #left elbow start and end point 
#                         lb_st = tuple(np.multiply(left_elbow,[640,480]).astype(int))
#                         lb_en = tuple(np.multiply(left_wrist,[640,480]).astype(int))

#                         #right elbow start and end point 
#                         rb_st = tuple(np.multiply(right_elbow,[640,480]).astype(int))
#                         rb_en = tuple(np.multiply(right_wrist,[640,480]).astype(int))
                        
#                         # Caluculateing distance between both hand
#                         h_dist = calculateDistance(left_wrist[0],left_wrist[1],right_wrist[0],right_wrist[1])
                        
#                         # Visual Output
#                         if angle_lknee >=74:
#                                     if angle_lknee >=132.0:                                
#                                         image = cv2.circle(image, lf_st, 30, (0, 0, 255), 3)
#                                         image = cv2.line(image,lf_st, lf_en, (0,0,255), 3)
#                                     if angle_lknee >=110.0 and angle_lknee <= 129.0:
#                                         image = cv2.circle(image, lf_st, 30, (0, 165, 255), 3)
#                                         image = cv2.line(image,lf_st, lf_en, (0,165,255), 3)
#                         if (angle_lshoulder <= 173.0 and angle_rshoulder <=173.0):  
#                                     if (angle_lshoulder >=8.0 and angle_lshoulder<=74.0 ):
#                                         image = cv2.circle(image, ls_st, 15, (0, 0, 255), 3)
#                                         image = cv2.line(image,ls_st, ls_en, (0,0,255), 3)
#                                     if (angle_rshoulder >=8.0 and angle_rshoulder <=74.0):
#                                         image = cv2.circle(image, rs_st, 15, (0, 0, 255), 3)
#                                         image = cv2.line(image,rs_st, rs_en, (0,0,255), 3)
#                                     if (angle_lshoulder >=75 and angle_lshoulder <= 173.0):
#                                         image = cv2.circle(image, ls_st, 15, (0, 165, 255), 3)
#                                         image = cv2.line(image,ls_st, ls_en, (0,165,255), 3)
#                                     if (angle_rshoulder >=75 and angle_rshoulder <=173.0):
#                                         image = cv2.circle(image, rs_st, 15, (0, 165, 255), 3)
#                                         image = cv2.line(image,rs_st, rs_en, (0,165,255), 3)
#                         if (h_dist>4):
#                                     if (angle_lshoulder >=173.0 and angle_rshoulder >=173.0):
#                                         image = cv2.circle(image, lb_en, 15, (0,165,255), 3)
#                                         image = cv2.line(image,lb_st, lb_en, (0,165,255), 3)
#                                         image = cv2.circle(image, rb_en, 15, (0,165,255), 3)
#                                         image = cv2.line(image,rb_st, rb_en, (0,165,255), 3)
#                                     else:
#                                         image = cv2.circle(image, lb_en, 15, (0,0,255), 3)
#                                         image = cv2.line(image,lb_st, lb_en, (0,0,255), 3)
#                                         image = cv2.circle(image, rb_en, 15, (0,0,255), 3)
#                                         image = cv2.line(image,rb_st, rb_en, (0,0,255), 3)
                        
#                         # Instructions based on angles
#                         if time.time()> t_wait:
#                                 # print("B")
#                                 t_wait+=wait
#                                 while (fi_fl==0):
#                                             if (c_knee==0):
                                                        
#                                                         if angle_lknee >=132.0 and angle_lknee <=150.0:
                                                           
#                                                             p = multiprocessing.Process(target=playsound, args=("yoga/bendknee.mp3",))
#                                                             p.start()
#                                                             p.terminate
#                                                             print('bend knee')
#                                                             break
#                                                         if angle_lknee >=110.0 and angle_lknee <= 129.0:
#                                                         # if angle_lknee >=85.0 and angle_lknee <= 129.0:    
#                                                             q = multiprocessing.Process(target=playsound, args=("yoga/Foot_up.mp3",))
#                                                             q.start()
#                                                             q.terminate
#                                                             break
#                                                         if angle_lknee < 109.0:
#                                                             c_knee=1
#                                                             print("correct knee")
                                                            
#                                             if (c_shoulder==0) and (c_knee==1): 
#                                                         if (angle_lshoulder >=80.0 and angle_lshoulder<=70.0 ) and (angle_rshoulder >=80.0 and angle_rshoulder <=70.0):
#                                                         # if (angle_lshoulder >=8.0 and angle_lshoulder<=70.0 ) and (angle_rshoulder >=8.0 and angle_rshoulder <=70.0):
#                                                             a = multiprocessing.Process(target=playsound, args=("yoga/parells.mp3",))
#                                                             a.start()
#                                                             a.terminate
#                                                             break
#                                                         if (angle_lshoulder >=71.0 and angle_lshoulder <= 160.0) and (angle_rshoulder >=71.0 and angle_rshoulder <=160.0):
#                                                             b = multiprocessing.Process(target=playsound, args=("yoga/Straigs.mp3",))
#                                                             b.start()
#                                                             b.terminate
#                                                             break
#                                                         if angle_lshoulder >=170.0 and angle_rshoulder >=170.0:
#                                                             c_shoulder = 1
#                                                             print("correct shoulder")
                                                            
#                                             if (c_elbow==0) and (c_shoulder==1):
#                                                         if(h_dist>4):
#                                                             f = multiprocessing.Process(target=playsound, args=("yoga/J_hands.mp3",))
#                                                             f.start()
#                                                             f.terminate
#                                                             break
#                                                         else:
#                                                             print("correct hand")
#                                                             c_elbow = 1
#                                                             posture_output= 1
                                                            
                                                
#                                             break
                        
#                         # Recheck if user breaksout of posture
#                         if (posture_output != 3):
#                             if (angle_lknee >=110):
#                             # if (angle_lknee >=90):    
#                                 c_knee = 0
#                                 fi_fl= 0
#                             if (angle_lshoulder <= 160.0 and angle_rshoulder <=160.0):
#                                 c_shoulder = 0
#                                 fi_fl = 0
#                             if (h_dist>5):
#                                 fi_fl= 0
#                                 c_elbow = 0

#                         c_total = (c_knee+c_shoulder+c_elbow)

                    
#                         if time.time()>t_wait2:
#                             t_wait2+=wait2
#                             while (proba >=60.0)and (c_total==3):
#                             # while (proba >=85.0)and (c_total==3):    
#                                 fi_fl=1
#                                 if (posture_output == 1):
#                                     print("wait 10 sec")
#                                     posture_output = 2
#                                     d = multiprocessing.Process(target=playsound, args=("yoga/10sec.mp3",))
#                                     d.start()
#                                     d.terminate
#                                     break
#                                 if (posture_output == 2):
#                                     print("completed")
#                                     posture_output = 3
#                                     x = multiprocessing.Process(target=playsound, args=("yoga/comp_pos.mp3",))
#                                     x.start()
#                                     x.terminate
#                                     break
                                
#                         # if (posture_output == 3):
#                         #     final_msg = "Close the window by pressing Q"
#                         #     cv2.rectangle(image, (100,310), (610,270), (0,0,255), -1)
#                         #     image = cv2.putText(image, final_msg, (100,300),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                                    

#             except:
#                     pass
                                

#             _, jpeg = cv2.imencode('.jpg', image)
#             return [jpeg.tobytes(), posture_output]