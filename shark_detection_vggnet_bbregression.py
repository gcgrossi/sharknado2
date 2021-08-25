from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import pickle

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

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

####################################################################################

# build path to test dataset
test = os.path.join(os.getcwd(),'test')

# load trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(os.path.join(os.getcwd(),"output","vggnet_bbregression.model"))

# load the classification model and label binarizer
netname = 'vggnet_finetune'
netinfo  = {'simplenn'         :{"pickle":"simple_nn_lb.pickle",'model':'simple_nn.model'  ,'size':32,'flatten':True},
            'vggnet'           :{"pickle":"smallvggnet.pickle" ,'model':'smallvggnet.model','size':64,'flatten':False},
            'vggnet_finetune'  :{"pickle":"vggnet_finetune_wnegative_iou02.pickle" ,'model':'vggnet_finetune_wnegative_iou02.model','size':224,'flatten':False}}
   
    
print("[INFO] loading classification model and label binarizer...")
classification = load_model(os.path.join(os.getcwd(),"output",netinfo[netname]["model"]))
lb = pickle.loads(open(os.path.join(os.getcwd(),"output",netinfo[netname]["pickle"]), "rb").read())

# load the contents of the CSV annotations file
print("[INFO] loading test dataset...")
annotations_csv = os.path.join(test,'test_annotation.csv')
rows = open(annotations_csv).read().strip().split("\n")

# initialize the lists
ious,accuracy = [],[]

# loop over the rows
for r,row in enumerate(rows):
    # break the row into the filename and bounding box coordinates
    row = row.split(",")
    (filename, label, true_startX, true_startY, true_endX, true_endY) = row

    # load the input image (in Keras format) from disk and preprocess
    # # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(os.path.join(test,label,filename), target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # make bounding box predictions on the input image
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
	
    # load the input imageand grab its dimensions
    image = cv2.imread(os.path.join(test,label,filename))
    (h, w) = image.shape[:2]
	
    # scale the predicted bounding box coordinates based on the image
	# dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    true_startX = int(true_startX)
    true_startY = int(true_startY)
    true_endX = int(true_endX)
    true_endY = int(true_endY)

    # compute intersection over union metric
    # and add to list 
    pred_box = (startX, startY, endX, endY)
    true_box=  (true_startX, true_startY, true_endX, true_endY)
    iou = compute_iou(true_box,pred_box)
    ious.append(compute_iou(true_box,pred_box))

    # extract the region from the input image, and preprocess it
    roi = image[startY:startY + endY, startX:startX + endX]
    roi = preprocess_image(roi,netinfo,netname)

    # make predictions on the roi
    # get prediction with max proba
    # get the corresponding label
    prediction = classification.predict(roi)
    prediction_proba = prediction.max(axis=1)[0]
    prediction_label = lb[prediction.argmax(axis=1)[0]]
    # append a list for accuracy based on the comparison
    # between predicted label and true label
    accuracy.append(1) if prediction_label == label else accuracy.append(0)
	
    #  draw the predicted and true bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
    cv2.rectangle(image, (true_startX, true_startY), (true_endX, true_endY),(0, 0, 255), 2)

    # draw label and confidence
    h = image.shape[0]
    cv2.putText(image, "Predicted", (10, int(h*0.05)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)
    cv2.putText(image, "True", (10, int(h*0.1)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)
    cv2.putText(image, "IOU: {:.2f}".format(iou), (10, int(h*0.15)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (3, 194, 252), 2)
    cv2.putText(image, "{} {:.1f}%".format(prediction_label,prediction_proba*100), (10, int(h*0.20)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (3, 194, 252), 2)
	
    # show the output image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
     # write roi to disk
    filename = "{}.png".format(r)
    outputPath = os.path.join(os.getcwd(),"assets",filename)
    cv2.imwrite(outputPath, image)

# print mean intersection over union
print("Mean IOU: {}".format(sum(ious)/len(ious)))
print("Accuracy: {}".format(sum(accuracy)/len(accuracy)))