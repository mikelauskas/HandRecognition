'''
Created on Nov 8, 2018

@author: mikel
'''

from cv2 import cv2
import numpy as np

from os import walk
from os.path import join

class Image(object):

    def __init__(self, params):
        pass
        
    @staticmethod
    def captureImage():
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("test")
        
        img_counter = 0
        ret, lastpic = cam.read()
        while True:
            ret, frame = cam.read()
            
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
        
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
                lastpic=frame
        cam.release()
        cv2.destroyAllWindows()
        
        return lastpic

    @staticmethod
    def captureBW_interface(path='..\\..\\data', x1=150, y1=100, width=300,
                            resize=True, nb_shots=50, per_frame=3):
        """
        Launch interface to collect the data.

        * Press B to select a background. Default is the first frame captured
          by the camera.
        * Press 0, 1, 2, 3, 4 or 5 to select the label of the picture. Default
          is -1.
        * Press S to save a frame.
        * Press SPACE to start a series of 5O shots every 3 frames.
        * Press L or R to move the capture window to the Left or Right.
        * Press P or M to increase or decrease the width of the capture window.
        * Press ESC to close the interface.
        """
        cam = cv2.VideoCapture(0)
        _, back = cam.read()

        label = -1
        shots = 0
        img_counters = [0, 0, 0, 0, 0, 0, 0]
        count_loop = 0

        ret, lastpic = cam.read()
        while True:
            count_loop = count_loop % per_frame
            # Live frame
            ret, frame = cam.read()
            # Draw the rectangle to put the hand
            cv2.rectangle(frame, (x1-3, y1-3), (x1+width+3, y1+width+3),
                          (0, 0, 255), 3)
            if label >= 0:
                cv2.putText(frame, "Collecting label: {}".format(label), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            if shots != 0:
                cv2.putText(frame, "{} pics to take!".format(shots), (15, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            cv2.imshow("live", frame)

            # Black and white frame
            bw_frame = Image.imgtobw(back, frame)
            # Draw the rectangle to put the hand
            cv2.rectangle(bw_frame, (x1-3, y1-3), (x1+width+3, y1+width+3),
                          (255, 255, 255), 3)
            cv2.imshow("bw", bw_frame)

            if not ret:
                break

            # k = cv2.waitKey(1)
            k = cv2.waitKey(33)
            if k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            elif k == 115 or k == 83:
                # S pressed : save image
                img_name = "bw_{}_{}.png".format(label, img_counters[label])
                # select part of the pic (and resize)
                roi = bw_frame[y1:y1+width, x1:x1+width]
                if resize:
                    roi = cv2.resize(roi, (50, 50))
                # save the pic
                cv2.imwrite(join(path, img_name), roi)
                print("{} written!".format(img_name))
                img_counters[label] += 1
                lastpic = roi

            elif k == 32:
                # Space pressed : continuous shots
                shots = nb_shots

            elif k == 98 or k == 66:
                # B pressed : select the background
                back = frame
                print("Background selected.")
            elif k == 108 or k == 76:
                # L pressed : move the red square to the left
                x1 = x1 - 10
            elif k == 114 or k == 82:
                # R pressed : move the red square to the right
                x1 = x1 + 10
            elif k == 112 or k == 80:
                # P pressed : increase the width of the red square
                width = width + 10
            elif k == 109 or k == 77:
                # M pressed : decrease the width of the red square
                width = width - 10

            elif k == 48:
                # 0 pressed : change the label to zero
                label = 0
            elif k == 49:
                # 1 pressed : change the label to one
                label = 1
            elif k == 50:
                # 2 pressed : change the label to two
                label = 2
            elif k == 51:
                # 3 pressed : change the label to three
                label = 3
            elif k == 52:
                # 4 pressed : change the label to four
                label = 4
            elif k == 53:
                # 5 pressed : change the label to five
                label = 5

            elif k == -1:
                pass
            else:
                print(k)

            if shots != 0 and count_loop == 0:
                img_name = "bw_{}_{}.png".format(label, img_counters[label])
                # select part of the pic (and resize)
                roi = bw_frame[y1:y1+width, x1:x1+width]
                if resize:
                    roi = cv2.resize(roi, (50, 50))
                # save the pic
                cv2.imwrite(join(path, img_name), roi)
                print("{} written!".format(img_name))
                img_counters[label] += 1
                shots -= 1
                lastpic = roi

            count_loop += 1
        cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def readImage(filename):
        #return cv2.resize(cv2.imread(filename),(500,500))
        return cv2.imread(filename)

    @staticmethod
    def displayImage(img):
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', img)
        while True:
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
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
        return bw

    @staticmethod
    def rgb2gray(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def blur(img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        return blurred

    @staticmethod
    def threshold(img):
        ret, thresh1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh1

    @staticmethod
    def thresholdBW(img):
        ret, thresh1 = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        # Erode and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh1 = cv2.erode(thresh1, kernel, iterations=1)
        thresh1 = cv2.dilate(thresh1, kernel, iterations=1)
        return ret, thresh1

    @staticmethod
    def truncate(img):
        ret,thresh=cv2.threshold(img, 210,240,cv2.THRESH_TOZERO_INV)
        return ret,thresh

    @staticmethod
    def contours(thresh):
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return im2, contours, hierarchy

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
        lower = np.array([110, 20, 70], dtype="uint8")
        upper = np.array([150, 255, 255], dtype="uint8")

        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 100)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        Image.displayImage(skin)

        return skin

    @staticmethod
    def load_images(folder, flatten=True):
        """
        Load the images from a folder.

        Return a numpy.array of (flatten) images and  a numpy array of their
        labels (given by the name of the folder they are in).
        """
        X = []
        Y = []
        for (dirpath, dirnames, filenames) in walk(folder):
            for filename in filenames:
                # collect, resize (and flatten) the image
                img = cv2.imread(join(dirpath, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (50, 50))
                if flatten:
                    img = img.flatten()
                else:
                    img = np.reshape(img, (50, 50, 1))
                # collect the class of the image
                y = int(dirpath.split("\\")[-1])

                X.append(img)
                Y.append(y)

        return np.array(X), np.array(Y)
