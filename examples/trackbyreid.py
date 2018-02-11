import _init_paths
import os, sys
this_dir = os.path.dirname(__file__)

os.environ['TENSORFLOW_MODE']='1'
from reidentifier import trackbyreid_deepsort_tf

from darkflow.darkflow.defaults import argHandler #Import the default arguments
from darkflow.darkflow.net.build import TFNet

import argparse

FLAGS = argHandler()
FLAGS.setDefaults()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tracking by Re-Identification')
    parser.add_argument('--gpu_id', dest='gpu_id', help='GPU id', default='0')
    parser.add_argument('--dir_model', dest='dir_model', help='Directory of the models',
                        default='')
    parser.add_argument('--input_video', dest='input_video', help='input_video',
                        default='')
    parser.add_argument('--output_dir', dest='output_dir', help='output_dir',
                        default='')


    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if not os.path.isfile(args.input_video):
        print("The Input video is not exist!")
        sys.exit()

    if not os.path.exists(args.dir_model):
        print("models are not exist!!")
        sys.exit()

    return args

args = parse_args()

REID = trackbyreid_deepsort_tf(args.gpu_id, gpu_fraction=0.2)
REID.load_model(args.dir_model)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
FLAGS.demo = args.input_video # video file to use, or if camera just put "camera"
FLAGS.model = os.path.join(args.dir_model, "person_detection/darkflow/yolo.cfg") # tensorflow model
FLAGS.load = os.path.join(args.dir_model, "person_detection/darkflow/yolo.weights") # tensorflow weights
FLAGS.config = os.path.join(this_dir, '..', 'IVALIB','darkflow', 'cfg')
#FLAGS.pbLoad = "tiny-yolo-voc-traffic.pb" # tensorflow model
#FLAGS.metaLoad = "tiny-yolo-voc-traffic.meta" # tensorflow weights
FLAGS.threshold = 0.7 # threshold of decetion confidance (detection if confidance > threshold )
FLAGS.gpu = 0.8 #how much of the GPU to use (between 0 and 1) 0 means use cpu
FLAGS.track = True # wheither to activate tracking or not
#FLAGS.trackObj = ['Bicyclist','Pedestrian','Skateboarder','Cart','Car','Bus'] # the object to be tracked
FLAGS.trackObj = ["person"]
FLAGS.saveVideo = False  #whether to save the video or not
FLAGS.BK_MOG = True # activate background substraction using cv2 MOG substraction,
                        #to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
                        # helps only when number of detection < 3, as it is still better than no detection.
FLAGS.tracker = "deep_sort" # wich algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
FLAGS.skip = 3 # how many frames to skipp between each detection to speed up the network
FLAGS.csv = False #whether to write csv file or not(only when tracking is set to True)
FLAGS.display = True # display the tracking or not
FLAGS.summary = None
FLAGS.csv = True
FLAGS.saveBox = True

darknet = TFNet(FLAGS)
REID.process(FLAGS, args.input_video, args.output_dir, darknet)
# camera(darknet)
#tfnet.camera()
exit('Demo stopped, exit.')
