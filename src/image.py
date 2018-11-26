from cv2 import cv2
import numpy as np

from os import walk
import os.path
from os.path import join


class Image(object):
    """Collection of method for collecting, processing and loading images."""

    def __init__(self, params):
        """Init of the class."""
        pass

    @staticmethod
    def captureImage(model=None, x1=150, y1=100, width=300, resize=True):
        """Capture an image and predict the number of fingers if a model is provided.

        * Press B to select a background. Default is the first frame captured
          by the camera.
        * Press SPACE to save a frame to return.
        * Press L or R to move the capture window to the Left or Right.
        * Press P or M to increase or decrease the width of the capture window.
        * Press ESC to close the interface.
        """
        cam = cv2.VideoCapture(0)
        _, back = cam.read()

        ret, lastpic = cam.read()
        while True:
            # Live frame
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            # Draw the rectangle to put the hand
            cv2.rectangle(frame, (x1-3, y1-3), (x1+width+3, y1+width+3),
                          (0, 0, 255), 3)

            # Black and white frame
            bw_frame = Image.imgtobw(back, frame)
            # Draw the rectangle to put the hand
            cv2.rectangle(bw_frame, (x1-3, y1-3), (x1+width+3, y1+width+3),
                          (255, 255, 255), 3)

            # Use a trained model to predict the number of fingers inside the red square
            if model is not None:
                img = cv2.resize(bw_frame[y1:y1+width, x1:x1+width], (50, 50))
                predict = model.predict(np.array([img.flatten()])).argmax()

                cv2.putText(frame, "predict : {}".format(predict), (15, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            cv2.imshow("live", frame)
            cv2.imshow("bw", bw_frame)

            if not ret:
                break

            # k = cv2.waitKey(1)
            k = cv2.waitKey(33)
            if k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            elif k == 32:
                # SPACE pressed : save image
                # select part of the pic (and resize)
                roi_bw = bw_frame[y1:y1+width, x1:x1+width]
                if resize:
                    roi_bw = cv2.resize(roi_bw, (50, 50))
                # save the pic
                lastpic = roi_bw

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

            elif k == -1:
                pass
            else:
                print(k)

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
            frame = cv2.flip(frame, 1)

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
                name_bw = "bw_{}_{}.png".format(label, img_counters[label])
                name_frm = "frm_{}_{}.png".format(label, img_counters[label])
                # select part of the pic (and resize)
                roi_bw = bw_frame[y1:y1+width, x1:x1+width]
                roi_frm = frame[y1:y1+width, x1:x1+width]
                if resize:
                    roi_bw = cv2.resize(roi_bw, (50, 50))
                    roi_frm = cv2.resize(roi_frm, (50, 50))
                # save the pic
                cv2.imwrite(join(path, name_bw), roi_bw)
                # uncomment to save rgb picture
                # cv2.imwrite(join(path, name_frm), roi_frm)
                print("{} and {} written!".format(name_bw, name_frm))
                img_counters[label] += 1

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

            # S pressed and frame multiple of 3
            if shots != 0 and count_loop == 0:
                name_bw = "bw_{}_{}.png".format(label, img_counters[label])
                name_frm = "frm_{}_{}.png".format(label, img_counters[label])
                # select part of the pic (and resize)
                roi_bw = bw_frame[y1:y1+width, x1:x1+width]
                roi_frm = frame[y1:y1+width, x1:x1+width]
                if resize:
                    roi_bw = cv2.resize(roi_bw, (50, 50))
                    roi_frm = cv2.resize(roi_frm, (50, 50))
                # save the pic
                cv2.imwrite(join(path, name_bw), roi_bw)
                cv2.imwrite(join(path, name_frm), roi_frm)
                print("{} and {} written!".format(name_bw, name_frm))
                img_counters[label] += 1
                shots -= 1

            count_loop += 1
        cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def readImage(filename, resize=True):
        """Read a (resized) imaage from a file."""
        img = cv2.imread(filename)
        if resize:
            img = cv2.resize(img, (500, 500))
        return img

    @staticmethod
    def displayImage(img):
        """Display the image in a new window.

        * Press ESC to close the window.
        """
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', img)
        while True:
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                break

    @staticmethod
    def imgtobw(back, img):
        """Remove the background from an image and return a b&w image."""
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
        """Transform rgb image to gray image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def blur(img):
        """Apply a Gausian blur on the image."""
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        return blurred

    @staticmethod
    def thresholdBW(img):
        """Transform image in B&W.

        First apply a binary treshold and the erode and dilate the image
        to suppress noise.
        """
        ret, thresh1 = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        # Erode and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh1 = cv2.erode(thresh1, kernel, iterations=1)
        thresh1 = cv2.dilate(thresh1, kernel, iterations=1)
        return ret, thresh1

    @staticmethod
    def load_images(folder, flatten=True):
        """Load the images from a folder into numpy array.

        The label of the images are given by the name of the folder
        they are in.
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
                label = int(os.path.basename(os.path.normpath(dirpath)))

                X.append(img)
                Y.append(label)

        return np.array(X), np.array(Y)
