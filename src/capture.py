from image import Image
import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
        Image.captureBW_interface(path, width=250)
    else:
        Image.captureBW_interface(width=250)
