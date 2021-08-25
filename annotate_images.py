import os
import cv2
import random
from csv import writer

from gvision import GVisionAPI
import pandas as pd


file_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
keep_labels     = ['great_white_shark','hammerhead_shark','whale_shark']

def list_files(indir=os.getcwd(),valid_extensions=file_extensions,valid_labels=keep_labels):
    for (rootdir,dirs,files) in os.walk(indir):
        for filename in files:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            
            # check to see if the file is an image and should be processed
            if valid_extensions is None or ext.endswith(valid_extensions):
                
                # construct the path to the image and yield it
                imagePath = os.path.join(rootdir, filename)
                
                # yield the path if the label should not be dropped 
                if imagePath.split(os.path.sep)[-2] in valid_labels:
                    yield imagePath
            
    return

def add_annotation(annotation_csv,annotation_list):

    # list of column names 
    field_names = ['image','top_left_x','top_left_y','bottom_right_x','bottom_right_y']

    with open(annotation_csv, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(annotation_list)

def load_annotation(annotation_file):

    header_list = ["image","label","tlx","tly","brx","bry"]

    # check if the annotation file exists
    if os.path.isfile(annotation_file) : 
        # if yes -> read as DataFrame
        df = pd.read_csv(annotation_file,names=header_list)
    else:
        # if no -> create an empty DataFrame
        df = pd.DataFrame(columns = header_list)
    
    return df

def main():
    
    # speed-up using multithreads
    cv2.setUseOptimized(True)   
    cv2.setNumThreads(4)

    # init path to files
    cwd=os.getcwd()
    dataset_path=os.path.join(cwd,"sharks")
    annotation_csv = os.path.join(dataset_path,"shark_annotation.csv")

    # load annotation as pandas dataframe
    df_annotation = load_annotation(annotation_csv)
    annotated_images = df_annotation['image'].to_list()

    # initialize google vision api
    gvision_auth=os.path.join(cwd,"google_vision","gvision_auth.json")
    gvis = GVisionAPI(gvision_auth)
    
    #obtain image paths and ramdomize it
    image_paths = list(list_files(dataset_path))
    random.seed(42)
    random.shuffle(image_paths)

    max_images = 1000
    image_count = 0    

    print("[INFO] Reading images from disk. This may take a while ... ")    
    for i in image_paths:

        # get image name and label from path
        name = i.split(os.path.sep)[-1]
        label = i.split(os.path.sep)[-2]
        
        # if already annotated -> skip image
        if name in annotated_images: continue

        #read the image
        image = cv2.imread(i)
        image_count+=1
        
        #copy image for drawing
        clone = image.copy()

        #perform a request to the API
        gvis.perform_request(image,'object detection')
        headers,objs = gvis.objects()

        # check if image contains a shark
        shark_detected = 0
        for obj in objs: 
            if obj[0] in ['Animal','Shark']:
                
                shark_detected+= 1
                
                # draw the true rectangle
                # and print class
                cv2.rectangle(image,(int(obj[2][0]),int(obj[2][1])),(int(obj[4][0]),int(obj[4][1])),(0,255,0), 2)
                print("{} - Class: {}".format(i,obj[0]))
                cv2.imshow("tue", image)

                while(1):

                    # if keyword is 's' (save) ->
                    # add annotation
                    if cv2.waitKey(1) & 0xFF == ord('s'): 

                        annotation_list = [name,label,int(obj[2][0]),int(obj[2][1]),int(obj[4][0]),int(obj[4][1])]
                        add_annotation(annotation_csv,annotation_list)

                        cv2.destroyAllWindows()
                        break


                    # if keyword is 'a' (abort) -> skip
                    elif cv2.waitKey(1) & 0xFF == ord('a'): 
                        cv2.destroyAllWindows()
                        break

                    # if keyword is 'e' (end) -> end
                    elif cv2.waitKey(1) & 0xFF == ord('e'): 
                        image_count=max_images
                        cv2.destroyAllWindows()
                        break

                   

        if image_count>=max_images: 
            # save csv
            break


    return

    
       
        
if __name__ == "__main__":
   main()