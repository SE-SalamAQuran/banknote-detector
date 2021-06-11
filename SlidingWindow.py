import imutils
import cv2 as cv


class Slider():
    def __init__(self, path):
        self.path = path

    def pyramid(self, path, scale=1.5, minSize=(30, 30)):
        image = cv.imread(path)
        # yield the original image
        yield image

        while True:

            # keep looping over the pyramid

            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
