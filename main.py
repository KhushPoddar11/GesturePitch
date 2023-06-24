import cv2, os
import cvzone
import mediapipe as mp
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.HandTrackingModule import HandDetector
import math

width, height = 1280, 720
folderPath = "Presentation"
imgNo=0
hs, ws = int(120*1), int(213*1)
gestureThreshold=550
buttonPressed = False
buttonCounter = 0
buttonDelay = 15
annotations = [[]]
annotationNumber = -1
annotationStart = False


#corelation between hand co-ordinates and distance in cms. - distance
x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y =[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff = np.polyfit(x,y,2)

#camera setup
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

##real-time background replacement
cap.set(cv2.CAP_PROP_FPS, 40)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("BackgroundImages")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'BackgroundImages/{imgPath}')
    imgList.append(img)


#get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
# print(pathImages)

#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

##new code
indexImg = 0
#

while True:
    #Import Images
    success, img=cap.read()
    img=cv2.flip(img,1)  
    ##new code
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(RGB)
    mask = results.segmentation_mask
    condition = np.stack(
    (results.segmentation_mask,) * 3, axis=-1) > 0.5
    bg_image = cv2.resize(imgList[3], (width, height))
    output_image = np.where(condition, frame, bg_image)

    pathFullImage = os.path.join(folderPath,pathImages[imgNo])
    currentImg = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img, flipType=False) 
    cv2.line(img,(0, gestureThreshold),(width,gestureThreshold),(0,255,0), 5)

    threshold = 70

    #adding distance formula
    if hands:
        lmList = hands[0]['lmList']
        x,y,w,h = hands[0]['bbox']
        x1,y1 = lmList[5][:2]
        x2,y2 = lmList[17][:2]
        indexFinger = lmList[8][0], lmList[8][1]
        ####
        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        A, B, C = coff
        distanceCM = A*distance**2 + B*distance + C
        # print(distanceCM, distance)
        
        if int(distanceCM)<threshold:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x,y-50))
            if hands and buttonPressed is False:
                hand = hands[0]
                fingers = detector.fingersUp(hand)
                cx, cy = hand['center']
                # print(fingers)

                if cy<=gestureThreshold: #if hand is at the height of the face

                    #Gesture 1 - Left
                    if fingers == [1,0,0,0,0]:
                        print('left')
                        if imgNo>0:
                            imgNo -= 1
                            buttonPressed = True
                            annotations = [[]]
                            annotationNumber = -1
                            annotationStart = False

                    #Gesture 2 - Right
                    if fingers == [0,0,0,0,1]:
                        print('right')
                        if imgNo < len(pathImages)-1:
                            buttonPressed = True
                            imgNo +=1
                            annotations = [[]]
                            annotationNumber = -1
                            annotationStart = False
                #Gesture 3 - Show Pointer
                if fingers == [0, 1, 1, 0, 0]:
                    cv2.circle(currentImg, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                #Gesture 4 - Draw with pointer
                if fingers == [0, 1, 0, 0, 0]:
                    if annotationStart is False:
                        annotationStart = True
                        annotationNumber += 1
                        annotations.append([])
                    print(annotationNumber)
                    annotations[annotationNumber].append(indexFinger)
                    cv2.circle(currentImg, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                else:
                    annotationStart = False
 
                if fingers == [0, 1, 1, 1, 0]:
                    if annotations:
                        annotations.pop(-1)
                        annotationNumber -= 1
                        buttonPressed = True
    else:
        annotationStart=False
    # if hands and buttonPressed is False:
    #     hand = hands[0]
    #     fingers = detector.fingersUp(hand)
    #     cx, cy = hand['center']
    #     # print(fingers)

    #     if cy<=gestureThreshold: #if hand is at the height of the face

    #         #Gesture 1 - Left
    #         if fingers == [1,0,0,0,0]:
    #             print('left')
    #             if imgNo>0:
    #                 imgNo -= 1
    #                 buttonPressed = True

    #         #Gesture 2 - Right
    #         if fingers == [0,0,0,0,1]:
    #             print('right')
    #             if imgNo < len(pathImages)-1:
    #                 buttonPressed = True
    #                 imgNo +=1
    
    #Button pressed iterations
    if buttonPressed:
        buttonCounter +=1
        if buttonCounter> buttonDelay:
            buttonPressed = False
            buttonCounter = 0
    ##new code:
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(currentImg, annotation[j - 1], annotation[j], (0, 0, 200), 12)
    ###
    
    #adding webcam img on slide
    imgSmall = cv2.resize(output_image, (ws,hs))
    h, w, _=currentImg.shape
    currentImg[0:hs,w-ws:w] = imgSmall
    # cv2.imshow("Image",img)

    cv2.imshow("Image",output_image)

    cv2.imshow("Slide", currentImg)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


            
