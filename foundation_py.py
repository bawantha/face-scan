import cv2
import numpy as np
import dlib
from PIL import ImageColor
import cv2
import numpy as np
import dlib
from PIL import ImageColor
from datetime import datetime
 
webcam = False
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./static/shape_predictor_68_face_landmarks.dat")
img_path = "./static/eye.jpg"
 
def Hex2rgb(hex):

    if hex.startswith("#"):
        rgb = ImageColor.getcolor(hex, "RGB")
    else:
        hex = "#"+hex
        rgb = ImageColor.getcolor(hex, "RGB")

    r, g, b = rgb[0], rgb[1], rgb[2]
    return  b, g, r
# def empty(a):
#     pass
# cv2.namedWindow("BGR")
# cv2.resizeWindow("BGR",640,240)
# cv2.createTrackbar("Blue","BGR",153,255,empty)
# cv2.createTrackbar("Green","BGR",0,255,empty)
# cv2.createTrackbar("Red","BGR",137,255,empty)


def createBox(img,points,scale=5,masked= False,cropped= True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)
        # cv2.imshow('mASKEMask',mask)
 
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        cv2.imwrite("NEWmask.jpg",imgCrop)
        return imgCrop
    else:
        return mask

def eyebrow_changer(image_path, ColorHex):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(0,0),None,0.6,0.6)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector(imgOriginal)
    myPoints =[]
    
    forheadPoints = []
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        # imgOriginal=cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)

        def landmar2circle(n, scale=0, COLOR=(50,50,255), isPoint=False, isforheadPoints = False):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            if isPoint == True:
                myPoints.append([x,y+scale])
            if isforheadPoints == True:
                forheadPoints.append([x,y+scale])
            # cv2.circle(imgOriginal, (x, y+scale), 2, COLOR ,cv2.FILLED)

        for n in range(1, 27, 1):
            if n >= 17 and n <= 27:
                if n == 18:
                    landmar2circle(n, scale=-20, COLOR=(255, 0, 0), isPoint=False, isforheadPoints=True)
                elif n == 21:
                    landmar2circle(n, scale=-30, COLOR=(255, 0, 0), isPoint=False, isforheadPoints=True)
                    # pass
                elif n == 23:
                    landmar2circle(n, scale=-25, COLOR=(255, 0, 0), isPoint=False, isforheadPoints=True)
                    # pass
                elif n == 25:
                    landmar2circle(n, scale=-20, COLOR=(255, 0, 0), isPoint=False, isforheadPoints=True)
                    # pass


                else:
                    # landmar2circle(n, scale=-30, COLOR=(255, 0, 0))
                    pass
            else:
                landmar2circle(n, isPoint=True)

        for i in range(len(forheadPoints)):
            myPoints.append(forheadPoints[-(i+1)])
        myPoints = np.array(myPoints)
        pts = myPoints # + forheadPoints[0]
        pts = pts.reshape((-1, 1, 2))
        maskLips = createBox(img, pts ,masked = True,cropped=False)
        imgColorLips = np.zeros_like(maskLips)

        b, g, r = Hex2rgb(ColorHex)

        imgColorLips[:] = b,g,r


        imgColorLips = cv2.bitwise_and(maskLips,imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),10)

        # imgOriginalGray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
        # imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgOriginal ,1,imgColorLips,0.4,0)

        # cv2.imshow('BGR', imgColorLips)
        whole_face = imgColorLips

        # cv2.imshow("show", imgOriginal)
        # cv2.waitKey(0)

        lipsPoints =[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lipsPoints.append([x,y])

        lipsPoints = np.array(lipsPoints)
        #pts = pts.reshape((-1, 1, 2))
        maskLips = createBox(img, lipsPoints[48:61],masked = True,cropped=False)
        # cv2.imshow("only lips", maskLips)
        imgColorLips = np.zeros_like(maskLips)
        hexcode = '000000'
        b, g, r = Hex2rgb(hexcode)
        imgColorLips[:] =  b,g,r
        imgColorLips = cv2.bitwise_and(maskLips, imgColorLips)
        #imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),10)

        imgOriginalGray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(whole_face ,1,imgColorLips,0.4,0)

        return imgColorLips


def save_image(image_path, hexcode):
    print("PATH OF IMATE IN SAVEIMAGE"+image_path);
    print(hexcode)
    img = eyebrow_changer(image_path, hexcode)
    indx = np.random.randint(10000)
    # print(datetime.now().strftime('%h'));
    path = "./static/LipsImage/"+ datetime.now().strftime('%m%S%M')+".jpg"
    cv2.imwrite(path,img)
    print("saved")


if __name__ == '__main__':
    save_image()

cap.release()
cv2.destroyAllWindows()
