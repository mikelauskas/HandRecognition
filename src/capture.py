from image import Image
import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        img_counter = int(sys.argv[1])
        _ = Image.captureImageBW(img_counter)
    elif len(sys.argv) == 3:
        img_counter = int(sys.argv[1])
        path = sys.argv[2]
        _ = Image.captureImageBW(img_counter, path)
    else:
        _ = Image.captureImageBW()
