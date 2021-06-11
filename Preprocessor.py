import cv2 as cv
import numpy as np
import random
import os

class DataAugmentor():
    def __init__(self, path): #Constructor
        self.path = path

    def file_lines_to_list(path):
        '''
        ### Convert Lines in TXT File to List ###
        path: path to file
        '''
        with open(path) as f:
            content = f.readlines()
        content = [(x.strip()).split() for x in content]
        return content

    def get_file_name(path):
        '''
        ### Get Filename of Filepath ###
        path: path to file
        '''
        basename = os.path.basename(path)
        onlyname = os.path.splitext(basename)[0]
        return onlyname

    def write_anno_to_txt(boxes, filepath):
        '''
        ### Write Annotation to TXT File ###
        boxes: format [[obj x1 y1 x2 y2],...]
        filepath: path/to/file.txt
        '''
        txt_file = open(filepath, "w")
        for box in boxes:
            print(box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4]), file=txt_file)
        txt_file.close()

    def randomcrop(self, path, scale=0.5):
        '''
        ### Random Crop ###
        img: image
        gt_boxes: format [[obj x1 y1 x2 y2],...]
        scale: percentage of cropped area
        '''
        img = cv.imread(path)
        # Crop image
        height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
        x = random.randint(0, img.shape[1] - int(width))
        y = random.randint(0, img.shape[0] - int(height))
        cropped = img[y:y + height, x:x + width]
        resized = cv.resize(cropped, (img.shape[1], img.shape[0]))


        return resized



    def colorjitter(self , path, cj_type="b"):
        '''
        ### Different Color Jitter ###
        img: image
        cj_type: {b: brightness, s: saturation, c: constast}
        '''
        img = cv.imread(path)
        jitter = img.copy()
        if cj_type == "b":
            # value = random.randint(-50, 50)
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv.cvtColor(jitter, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            if value >= 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = np.absolute(value)
                v[v < lim] = 0
                v[v >= lim] -= np.absolute(value)

            final_hsv = cv.merge((h, s, v))
            jitter = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            return jitter

        elif cj_type == "s":
            # value = random.randint(-50, 50)
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            hsv = cv.cvtColor(jitter, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            if value >= 0:
                lim = 255 - value
                s[s > lim] = 255
                s[s <= lim] += value
            else:
                lim = np.absolute(value)
                s[s < lim] = 0
                s[s >= lim] -= np.absolute(value)

            final_hsv = cv.merge((h, s, v))
            jitter = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            return jitter

        elif cj_type == "c":
            brightness = 10
            contrast = random.randint(40, 100)
            dummy = np.int16(jitter)
            dummy = dummy * (contrast / 127 + 1) - contrast + brightness
            dummy = np.clip(dummy, 0, 255)
            img = np.uint8(dummy)
            return img

    def noisy(self, path, noise_type="gauss"):
        '''
        ### Adding Noise ###
        img: image
        cj_type: {gauss: gaussian, sp: salt & pepper}
        '''
        img = cv.imread(path)
        if noise_type == "gauss":
            image = img.copy()
            mean = 0
            st = 0.7
            gauss = np.random.normal(mean, st, image.shape)
            gauss = gauss.astype('uint8')
            image = cv.add(image, gauss)
            return image

        elif noise_type == "sp":
            image = img.copy()
            prob = 0.05
            if len(image.shape) == 2:
                black = 0
                white = 255
            else:
                colorspace = image.shape[2]
                if colorspace == 3:  # RGB
                    black = np.array([0, 0, 0], dtype='uint8')
                    white = np.array([255, 255, 255], dtype='uint8')
                else:  # RGBA
                    black = np.array([0, 0, 0, 255], dtype='uint8')
                    white = np.array([255, 255, 255, 255], dtype='uint8')
            probs = np.random.random(image.shape[:2])
            image[probs < (prob / 2)] = black
            image[probs > 1 - (prob / 2)] = white
            return image

    def filters(self, path, f_type="blur"):
        '''
        ### Filtering ###
        img: image
        f_type: {blur: blur, gaussian: gaussian, median: median}
        '''
        img = cv.imread(path)
        if f_type == "blur":
            image = img.copy()
            fsize = 9
            return cv.blur(image, (fsize, fsize))

        elif f_type == "gaussian":
            image = img.copy()
            fsize = 9
            return cv.GaussianBlur(image, (fsize, fsize), 0)

        elif f_type == "median":
            image = img.copy()
            fsize = 9
            return cv.medianBlur(image, fsize)

class CascadeGenerator():
    def __init__(self, path):
        self.path = path

    def generate_negative_description_file(n):
        with open('data/neg/' + str(n) + '.txt', 'w') as f:
            for filename in os.listdir('data/images/n' + str(n)):
                print('data/images/n/' + str(n) + '/' + filename + '\n')
                f.write('data/images/n/' + str(n) + '/' + filename + '\n')

    def generate_positive_description_file(n):
        with open('data/pos/' + str(n) + '.txt', 'w') as f:
            for filename in os.listdir('data/images/p/' + str(n)):
                f.write('images/p/' + str(n) + '/' + filename + '\n')