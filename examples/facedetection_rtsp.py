import sys
import os.path as osp
import argparse
import os
import vlc
import time
import cv2
import requests

this_dir = osp.dirname(__file__)
os.environ['TENSORFLOW_MODE']='1'
os.environ['FACE_MODULE']='1'
sys.path.append(osp.join(this_dir, '..', 'IVALIB'))
from detector import face_detection_mtcnn_tf

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='face detection in RTSP')
    parser.add_argument('--gpu_id', dest='gpu_id', help='GPU id', default='-1')
    parser.add_argument('--dir_model', dest='dir_model', help='Directory of the models',
                        default='')
    parser.add_argument('--rtsp_ip', dest='rtsp_ip', help='rtsp ip',
                        default='rtsp://192.168.1.168/0')
    parser.add_argument('--fr_server', dest='fr_server', help='face recognition server',
                        default='http://localhost:3000/smartedge')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if not os.path.exists(args.dir_model):
        print("The model path is not exist!")
        sys.exit()

    return args

args = parse_args()

gpu_id = '-1' # use_cpu
FD = face_detection_mtcnn_tf(args.gpu_id)
FD.load_model(args.dir_model)

# reader = imageio.get_reader('<video0>') #change the mp4 video path with '<video0>' if we use webcam
print(args.rtsp_ip)
player = vlc.MediaPlayer(args.rtsp_ip) 
player.play()
idx = 0

if not os.path.exists('tmp'):
    os.makedirs('tmp')

idx_face = 0
while 1:
    try:
        time.sleep(1)
        player.video_take_snapshot(0, 'tmp/snapshot.png', 0, 0)
        res = FD.process_find_largest_face('tmp/snapshot.png')
        #print(np.array(res).shape)
        if res is not None:
            idx_face +=1
            fnm = 'tmp/face_'+str(idx_face)+'.png'
            cv2.imwrite(fnm, res)
            print('face is detected!')
            url = args.fr_server+"/user_detected?track_id=1&ip=1&image=%s" % fnm + "&timestamp=1"
	    print(url)
	    r = requests.get(url)
	    print r
        else:
            print('no detected face!')
    except:
        print("system error!!!")
        sys.exit(1)
