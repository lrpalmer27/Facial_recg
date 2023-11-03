##this file will find faces in images, then remove all other pixles from the image before saving the new image, mask coordinates in VGG format so the faces can be associated with a name.

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
from mrcnn import utils
import cv2
import os
import random
import numpy
import skimage.draw
from skimage.io import imsave
import matplotlib as MPL
import matplotlib.pyplot as plt

_TestDir="\\.PeopleData\\"
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "2stepfacial"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

def init():
    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(mode="inference", 
                                config=SimpleConfig(),
                                model_dir=os.getcwd())

    # Load the weights into the model.
    model.load_weights(filepath=ROOT_DIR+"\\mask_rcnn_coco.h5", 
                    by_name=True)
    
    return model

def detect(model,filename):
    image = cv2.imread(ROOT_DIR+_TestDir+'test\\'+filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=0)

    # Get the results for the first image.
    r = r[0]
    
    classids = r['class_ids']
    print(classids.shape) #print shape of array shape[0] is the num of objects detected.
    
    # #print class names below; 
    # obj=classids.tolist()
    # for i in obj: 
    #     print(CLASS_NAMES[i])
    
    if len(classids.shape)>1:
        if classids.shape[1] != '':
            print("bad shape?")
 
    return [r,image]

def visualize(n,img,save=False):
    # Visualize the detected objects -- this is good for trouble shooting: not used natively.
    
    if save:
        #saves a b/w image with the 
        imsave(ROOT_DIR+_TestDir+'test.png',img)
    
    mrcnn.visualize.display_instances(image=img, 
                                    boxes=n['rois'], 
                                    masks=n['masks'], 
                                    class_ids=n['class_ids'], 
                                    class_names=CLASS_NAMES, 
                                    scores=n['scores'])

def save_people(img,instance,original_filename,out,nsplash,save_annotations=False):
    #img is the cropped original image, where nsplash is the cropped original, with the mask and ROI overlaid
    
    if save_annotations:
        print("call fcn to generate and save annotation here")
    
    imsave(ROOT_DIR+_TestDir+'bodies\\'+original_filename+'_'+str(instance)+'.png',img)

def crop(r,image,filename):  
    """
     Returns a list of dicts, one dict per image. The dict contains:
        we need to rebuild the results from model.detect for the single instance detected: 
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, ``W, N]
    """
    masks=r['masks']
    boxes=r['rois']
    c = r['class_ids']
    class_ids=c.tolist()
    scores=r['scores'] 
     
    image_shape=image.shape[0:2]
    N = boxes.shape[0] #number of instances

    for n in range(N):
        if class_ids[n]==1:
            if not numpy.any(boxes[n]):
                continue
            y1, x1, y2, x2 = boxes[n]
            roi=boxes[n]
            
            newimg=image[y1:y2,x1:x2]
            crop=(y1, x1, newimg.shape[0], newimg.shape[1])
            
            #this mask work here doesnt consider multiple masks in an image!
            nmask = (numpy.sum(masks, -1, keepdims=True) >= 1)
            nmask=utils.resize_mask(nmask, 1, [(0, 0), (0, 0), (0, 0)], crop=crop) #resize coordinates for the mask after cropping
            
            shift=25 #shift boundary by n number of px so we can see it.
            nroi=[0+shift,0+shift,newimg.shape[0]-shift*2,newimg.shape[1]-shift*2]
            
            ngray = skimage.color.gray2rgb(skimage.color.rgb2gray(newimg)) * 255
            nsplash = numpy.where(nmask, newimg, ngray).astype(numpy.uint8)

            # rebuild output, for this instance
            out={"rois":numpy.array([nroi]),
                "class_ids":numpy.array([class_ids[n]]),
                "scores":scores,
                "masks":nmask}
            
            # visualize(out,nsplash) #this is used to troubleshoot things
            print("Added: ", CLASS_NAMES[class_ids[n]])
            
            save_people(newimg,n,filename,out,nsplash,False)
            
        else:
            print("Skipped:",CLASS_NAMES[class_ids[n]])
            continue
    
    # visualize(r,image) ##troubleshooting
    return
   

if __name__=='__main__': 
    m=init()
    filenames=Test_img_list=os.listdir(ROOT_DIR+_TestDir+'\\test\\')
    # filenames=['_O4A9692-Edit.jpg']
    for file in filenames:
        print('Chosen random file to display with mask predictions: ', file)
        r,img=detect(m,file) #remove filename output from this and make a for loop in this if statement to pick all pics in the dir
        crop(r,img,file)

