from image import Image
import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
        _ = Image.captureImageBW(path, width=200)
    else:
        _ = Image.captureImageBW(width=200)
