from image import Image
from background import MyClass as bg


if __name__ == '__main__':
    pic=Image.captureImage()
    #pic=Image.readImage('photo_2018-11-08_19-35-56.jpg')
    Image.displayImage(pic)
#    pic=Image.detect_skin(pic)
#    Image.displayImage(pic)
    gray=Image.rgb2gray(pic)
    gray = bg.remover()
    Image.displayImage(gray)
    blur = Image.blur(gray)
    res,thres0=Image.truncate(blur)
    Image.displayImage(thres0)
    res,thresh=Image.threshold(thres0)
    Image.displayImage(thresh)
    im2, contours, hierarchy=Image.contours(thresh)
    drawing=Image.hull(contours, thresh, hierarchy)
    
    Image.displayImage(drawing)
    
    pass