# import the necessary packages
import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d

class ImagePreprocessor:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def preprocess(self, images, verbose=-1):
        # initialize the list of features and labels
        data = []

        num_image = images.shape[0]
        
        # loop over the input images
        for i in np.arange(num_image):
            image = images[i,:,:,:]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    image = image.astype('uint8')
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, num_image))

        # return a tuple of the data and labels
        return (np.array(data))


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)


class ResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # dimension
        if w < h:
            scale_rate = w / self.width
            image = cv2.resize(image, (self.width, int(h / scale_rate)), interpolation=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
        else:
            scale_rate = h / self.height
            image = cv2.resize(image, (int(w / scale_rate), self.height), interpolation=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing
        # the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
    

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue
        # channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
    
        # merge the channels back together and return the image
        return cv2.merge([B, G, R])
    

class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the image with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]
        
class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store the target image width, height, whether or not
        # horizontal flips should be included, along with the
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter
        
    def preprocess(self, image):
        # initialize the list of crops
        crops = []
        
        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]
        
        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])
        
        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)
            
        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops] 
            crops.extend(mirrors)
            
        # return the set of crops
        return np.array(crops)