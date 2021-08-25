# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:30:25 2021

@author: giuli
"""

import os
import sys
import pickle
import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model
from imutils.object_detection import non_max_suppression


def selective_search(image, method="fast"):

	# initialize OpenCV's selective search implementation and set the
	# input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == "fast":
	    ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
	    ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    print("[INFO] Processing Selective Search...")
    rects = ss.process()
    # return the region proposal bounding boxes
    return rects

def preprocess_image(image, netinfo, netname):
    # resize to model dimensions
    image = cv2.resize(image, (netinfo[netname]["size"], netinfo[netname]["size"]))

    # mean subtraction
    if netname == 'vggnet_finetune':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the image to a floating point data type and perform mean
        # subtraction
        image = image.astype("float32")
        mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
        image -= mean
    else:
        # scale the pixel values to [0, 1]
        image = image.astype("float") / 255.0
        
    # check to see if we should flatten the image and add a batch
    # dimension
        
    if netinfo[netname]["flatten"]:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
        # otherwise, we must be working with a CNN -- don't flatten the
        # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    return image

def main():
    
    # speed-up using multithreads
    cv2.setUseOptimized(True)   
    cv2.setNumThreads(4)

    cwd=os.getcwd()
    test=os.path.join(cwd,"test")
    
    # load the model and label binarizer
    netname = 'vggnet_finetune'
    netinfo  = {'simplenn'         :{"pickle":"simple_nn_lb.pickle",'model':'simple_nn.model'  ,'size':32,'flatten':True},
                'vggnet'           :{"pickle":"smallvggnet.pickle" ,'model':'smallvggnet.model','size':64,'flatten':False},
                'vggnet_finetune'  :{"pickle":"vggnet_finetune_wnegative_iou02.pickle" ,'model':'vggnet_finetune_wnegative_iou02.model','size':224,'flatten':False}}
   
    
    print("[INFO] loading network and label binarizer...")
    model = load_model(os.path.join(cwd,"output",netinfo[netname]["model"]))
    lb = pickle.loads(open(os.path.join(cwd,"output",netinfo[netname]["pickle"]), "rb").read())

    # prepare a dictionary to store prediction informations
    dict_prediction = dict.fromkeys(lb)
    
    image_number = 0
    for f in os.listdir(test):
        
        # for each image reset dicitionary 
        # with boxes and probabilities
        for key in dict_prediction.keys(): 
            dict_prediction[key] = {"box":[],"proba":[]}
        
        print("[INFO] Reading :"+f)
        image_number+= 1
        
        # read input image
        #f="7.jpg"
        image=cv2.imread(os.path.join(test,f))
        output=image.copy()

        # reshape image maintaining aspect ratio
        # should speed-up selective search
        (H,W) = image.shape[:2]
        newH = 200
        newW = int(W*200/H)
        image = cv2.resize(image, (newW, newH))
        
        proposals,boxes=[],[]
        prediction_number = 0

        rects = selective_search(image)
        print('[INFO] Running Predictions on Proposed Regions ... ')
        # loop over the region proposal bounding box 
        for (x, y, w, h) in rects[0:100]:

            prediction_number+=1
            #print('predicting {}/{} proposals'.format(prediction_number,len(rects)),end = "\r")

            # ignore if the width or height of the region is less than 10% of the
            # image width or height
            if w / float(newW) < 0.1 or h / float(newH) < 0.1:
                continue
            # extract the region from the input image, and preprocess it
            roi = image[y:y + h, x:x + w]
            roi = preprocess_image(roi,netinfo,netname)
	     
            # update our proposals and bounding boxes lists
            #proposals.append(roi)
            #boxes.append((x, y, w, h))

            # make predictions on the roi
            # append prediction with max proba
            # append the corresponding label
            prediction = model.predict(roi)
            prediction_proba = prediction.max(axis=1)[0]

            # skip the roi if it's a negative class
            label = lb[prediction.argmax(axis=1)[0]]
            if label == "not_shark":
                continue

            if prediction_proba>=0.99:

                # draw rectangles with labels and confidence
                # on each selected roi
                c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                clone = image.copy()
                cv2.rectangle(clone, (x, y), (x+w, y+h),c, 2)
                text = "{}: {:.2f}%".format(label, prediction_proba * 100)
                cv2.putText(clone, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,c, 2)

                # append prediction information to dictionary
                dict_prediction[label]['proba'].append(prediction_proba)
                dict_prediction[label]['box'].append((x, y, x + w, y + h))

                # print predictions
                print('{} - proba: {}'.format(label,prediction_proba))

                # draaw roi with prediction
                cv2.imshow("Before_{}".format(prediction_number), clone)
        
     
        # start non max suppression 
        for label in dict_prediction.keys():
            if len(dict_prediction[label]["proba"]) > 0:
                box   = np.array(dict_prediction[label]["box"])
                proba = np.array(dict_prediction[label]["proba"])

                # apply non-max subpression
                boxes = non_max_suppression(box, proba)

                # loop over all bounding boxes that were kept after applying
                # non-maxima suppression
                for (startX, startY, endX, endY) in boxes:
                    # draw the bounding box and label on the image
                    cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


        cv2.imshow("After", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if image_number==100: break
        #print("\n",preds)


        """
        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb[i]
        print("[INFO] prediction {}: ".format(preds))
        
        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        
        # show the output image
        cv2.imshow("Image", output)
        cv2.waitKey(0)
        """
        
if __name__ == "__main__":
   main()