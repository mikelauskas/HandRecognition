'''
Created on Nov 8, 2018

@author: mikel
'''

#from cv2 import cv2
import cv2
import numpy as np

from os import walk
from os.path import join

x1 = 150
x2 = 450
y1 = 100
y2 = 400

class Image(object):

    def __init__(self, params):
        pass
        
    @staticmethod
    def captureImage():
        cam = cv2.VideoCapture(0)
        # cv2.namedWindow("test")

        _, back = cam.read()

        img_counter = 0
        ret, lastpic = cam.read()
        while True:
            ret, frame = cam.read()


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.imshow("live", frame)
            difference = cv2.absdiff(back, frame)
            cv2.imshow("diff", difference)

            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                difference = cv2.absdiff(back, frame)
                roi = difference[y1:y2, x1:x2]
                cv2.imwrite(img_name, roi)
                print("{} written!".format(img_name))
                img_counter += 1
                lastpic = roi
            elif k % 256 == 98:
                # b pressed
                back = frame
                print("Background selected.")
        cam.release()
        cv2.destroyAllWindows()

        return lastpic

    def captureImageBW():
        cam = cv2.VideoCapture(0)

        _, back = cam.read()

        img_counter = 42
        ret, lastpic = cam.read()
        while True:
            ret, frame = cam.read()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imshow("live", frame)

            bw_frame = Image.imgtobw(back, frame)
            cv2.rectangle(bw_frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 3)
            cv2.imshow("bw", bw_frame)

            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "bw_{}.png".format(img_counter)
                # select part of the pic
                roi = bw_frame[y1:y2, x1:x2]
                cv2.imwrite(img_name, roi)
                print("{} written!".format(img_name))
                img_counter += 1
                lastpic = roi
            elif k % 256 == 98:
                # b pressed
                back = frame
                print("Background selected.")
        cam.release()
        cv2.destroyAllWindows()

        return lastpic

    @staticmethod
    def readImage(filename):
        return cv2.resize(cv2.imread(filename),(500,500))
        #return cv2.imread(filename)
    
    @staticmethod
    def displayImage(img):
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE )
        
        cv2.imshow('img',img)
        while True:
            k = cv2.waitKey(10)
            if k == 27:
                break

    @staticmethod
    def imgtobw(back, img):
        # substract the background
        difference = cv2.absdiff(img, back)
        # to grayscale
        gray = Image.rgb2gray(difference)
        # Gaussian blur
        blur = Image.blur(gray)
        # binary treshold
        _, bw = Image.thresholdBW(blur)
        return difference

    @staticmethod
    def rgb2gray(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def blur(img):
        blurred=cv2.GaussianBlur(img,(5,5),0)
        return blurred
    
    @staticmethod
    def thresholdBW(img):
        ret, thresh1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh1 = cv2.erode(thresh1, kernel, iterations=1)
        thresh1 = cv2.dilate(thresh1, kernel, iterations=1)

        return ret, thresh1

    @staticmethod
    def threshold(img):
        ret, thresh1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh1

    @staticmethod
    def truncate(img):
        ret,thresh=cv2.threshold(img,230,240,cv2.THRESH_TOZERO_INV)
        return ret,thresh
        
    @staticmethod
    def contours(thresh):
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return im2,contours,hierarchy
    
    @staticmethod
    def hull(contours,thresh,hierarchy):
        # create hull array for convex hull points
        hull = []
         
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
        
        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
         
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)
        
        return drawing
    
    @staticmethod
    def detect_skin(frame):
        lower = np.array([5, 5, 70 ], dtype = "uint8")
        upper = np.array([150, 255, 255], dtype = "uint8")
        
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
 
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
     
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5,5), 100)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        
        Image.displayImage(skin)
        
        return skin


    def load_images(file, flatten=True):
        X = []
        Y = []
        for (dirpath, dirnames, filenames) in walk(file):
            for filename in filenames:
                # collect, resize and flatten the image
                img = cv2.imread(join(dirpath, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (50, 50))
                if flatten:
                    img = img.flatten()
                # collect the class of the image
                y = int(dirpath.split("\\")[-1])

                X.append(np.reshape(img, (50,50,1)))
                Y.append(y)

        return X, Y

            
