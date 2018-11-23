from image import Image


if __name__ == '__main__':
    #pic=Image.captureImage()
    pic=Image.readImage('opencv_frame_0.png')
    
    pic=Image.detect_skin(pic)
    gray=Image.rgb2gray(pic)
    blur = Image.blur(gray)
    res,thres0=Image.truncate(blur)
    Image.displayImage(thres0)
    res,thresh=Image.threshold(thres0)
    Image.displayImage(thresh)
    im2, contours, hierarchy=Image.contours(thresh)
    drawing=Image.hull(contours, thresh, hierarchy)
    
    Image.displayImage(drawing)
    
    pass