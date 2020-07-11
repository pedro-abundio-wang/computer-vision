# import the necessary packages
import cv2
import os
import glob
import matplotlib.pyplot as plt

input = '../datasets/breaking-captcha'
output = '../output/breaking-captcha'

# grab the image paths then initialize the dictionary of character counts
imagePaths = glob.glob('../datasets/breaking-captcha/solved-captchas/*.png', recursive=True)
counts = {}

# loop over the image paths
for(i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    print("[INFO] processing image {}".format(imagePath))
    
    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # threshold the image to reveal the digits
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # find contours in the image, keeping only the four largest ones
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # loop over the contours
        for c in contours:
            # compute the bounding box for the contour then extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            print("[INFO] contour location {}, {}, {}, {}".format(x, y, w, h))
            
            # display the character, making it larger enough for us
            # to see, then wait for a keypress
            cv2.imshow("ROI", cv2.resize(roi, (28, 28)))
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # if the ’‘’ key is pressed, then ignore the character
            if key == ord("-"):
                print("[INFO] ignoring character")
                continue
    
            # grab the key that was pressed and construct the path
            # the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([output, key])
    
            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
    
            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
    
            # increment the count for the current key
            counts[key] = count + 1
    
    # we are trying to control-c out of the script, so break from the
    # loop (you still need to press a key for the active window to
    # trigger this)
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break