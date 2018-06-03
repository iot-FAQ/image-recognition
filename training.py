import numpy as np
import cv2
import mahotas


import os

path, dirs, files = next(os.walk("./samples/"))
file_count = len(files)
path, dirs, files = next(os.walk("./images/"))
images_count = len(files)

print file_count
# OPEN TRAINING IMAGE FOR PROCESSING------------------------------------------------------------------------------------
samples =  np.empty((0, 100))
responses = []
for filename in os.listdir('./samples/'):
    print filename
    image = cv2.imread('samples/'+filename,0)
    image=cv2.resize(image,(50,50))
#DETECTION THRESHOLD----------------------------------------------------------------------------------------------------
    T= mahotas.thresholding.otsu(image)
    for k in range(1,50,1):
        for z in range(1,50,1):
            color=image[k,z]
            if (color>T):
                image[k,z]=0
            else:
                image[k,z]=255
    thresh2=image.copy()
    keys = [i for i in range(48, 58)]
    roi_small = cv2.resize(thresh2, (10, 10))
    cv2.destroyWindow('norm')
    cv2.imshow('Numero', image)
    key = cv2.waitKey(0) & 0xFF
    print "key=", key
    #print int("4")
    print 'You pressed %d (0x%x), 2LSB: %d (%s)' % (key, key, key % 2**16,
    repr(chr(key%256)) if key%256 < 128 else '?')
    if key == 27:
       cv2.destroyAllWindows()
    elif key in keys:
        sample = roi_small.reshape((1,100))
        samples = np.append(samples,sample,0)
	print int(chr(key))
        responses.append(int(chr(key)))
print "training complete"
np.savetxt('data/general_samples.data', samples)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('data/general_responses.data', responses)
