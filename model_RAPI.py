# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import pandas as pd
import os
import tempfile
import os.path as osp
import cv2

import settings
import helpers
import redis
import time
import json


this_dir = osp.dirname(__file__)

sys.path.append(osp.join(this_dir, 'DLL'))
os.environ['TENSORFLOW_MODE']='1'
os.environ['PERSON_MODULE']='1'
os.environ['BIBNUMBER_MODULE']='1'
os.environ['PYTORCH_MODE']='1'

from recognizer import bibnumber_crnn_pytorch
from detector import person_faster_rcnn_tf, bibnumber_ctpn_tf

model_person = None
model_bibdetect = None
model_bibrecog = None

def get_models():
    global model_person
    global model_bibdetect
    global model_bibrecog

    if model_person is None:
        model_person = person_faster_rcnn_tf('models_zoo','mobilenet', '0')
    if model_bibdetect is None:
        model_bibdetect = bibnumber_ctpn_tf('0')  # bibnumber_ctpn_caffe(args.dir_model, args.gpu_id)
        model_bibdetect.load_model('models_zoo')
    if model_bibrecog is None:
        model_bibrecog = bibnumber_crnn_pytorch('models_zoo', '0')
    return model_person, model_bibdetect, model_bibrecog

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
        port=settings.REDIS_PORT, db=settings.REDIS_DB)

def sync_bibnumber_api(images_list, rfid_record):
    df = pd.read_csv(rfid_record)
    print("Loading model...")
    mperson, mbibdetect, mbibrecog = get_models()
    print("Model loaded!")
    #image_path = os.path.join(UPLOAD_FOLDER, filename)
    #if not os.path.isfile(image_path):
    #    print('file does not exist: ' + image_path)
    #    return {}

    valjsonbatch = []
    print(np.array(images_list).shape)
    for idx, im in enumerate(images_list):
        anno_im ={}
        #im = cv2.imread(image_path)
	print(np.array(im).shape)
        inds, dets = mperson.process(im)

        valjson = []
        for ii in inds:
            bbox_person = dets[ii, :4]
            h1 = int(bbox_person[3] - bbox_person[1])
            w1 = int(bbox_person[2] - bbox_person[0])

            im_scale, text_lines, scale = mbibdetect.process(im, bbox_person)
            h, w = np.array(im_scale).shape[0:2]

            scale_x = 1.0 * w1 / w
            scale_y = 1.0 * h1 / h

            simpreds = mbibrecog.process(im_scale, text_lines)

            for j, simpred in enumerate(simpreds):
                anno = {}
                df1 = df.loc[df['id'].isin([simpred])]
                if not df1.empty:
                    text_lines[j][0] *= scale_x
                    text_lines[j][1] *= scale_y
                    text_lines[j][2] *= scale_x
                    text_lines[j][3] *= scale_y
                    # h2, w2 = np.array(im).shape[0:2]
                    # thick = int((h2 + w2) / 300)
                    # cv2.rectangle(im,
                    #               (int(text_lines[j][0] + bbox_person[0]), int(text_lines[j][1] + bbox_person[1])),
                    #               (int(text_lines[j][2] + bbox_person[0]), int(text_lines[j][3] + bbox_person[1])),
                    #               (255, 0, 0), 2)
                    # cv2.rectangle(im,
                    #               (bbox_person[0], bbox_person[1]),
                    #               (bbox_person[2], bbox_person[3]),
                    #               (255, 0, 0), 2)
                    # cv2.putText(im, simpred, (int(bbox_person[0]), int(bbox_person[1] - 12)),
                    #             0, 1e-3 * h, (255, 0, 0), thick // 3)
                    # cv2.imwrite('test.jpg', im)
                    anno['person_rect'] = bbox_person.tolist()

                    plate_recs = np.zeros(4, dtype=np.int32)
                    plate_recs[0] = int(text_lines[j][0] + bbox_person[0])
                    plate_recs[1] = int(text_lines[j][1] + bbox_person[1])
                    plate_recs[2] = int(text_lines[j][2] + bbox_person[0])
                    plate_recs[3] = int(text_lines[j][3] + bbox_person[1])
                    anno['plate_rect'] = plate_recs.tolist()
                    anno['plate_prediction'] = simpred
                    valjson.append(anno.copy())

        #anno_im['id'] = images_id[idx]
        anno_im['prediction'] = valjson
        valjsonbatch.append(anno_im)
    return valjsonbatch

def detect_process():
    # continually pool for new images to detect
    while True:
        # attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
			settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"],
				settings.IMAGE_DTYPE, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
					settings.IMAGE_CHANS))
            #rfid = q["rfid"]

            # check to see if the batch list is None
            if batch is None:
                batch = image

            # otherwise, stack the data
            else:
                batch = np.vstack([batch, image])

            # update the list of image IDs
            imageIDs.append(q["id"])

        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            # detect the batch
            print("* Batch size: {}".format(batch.shape))
            rfid = './data/tmp/rfid.csv'
            results = sync_bibnumber_api(batch, rfid)
            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, results):
                db.set(imageID, json.dumps(resultSet))

            # remove the set of images from our queue
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    try:
        detect_process()
    except KeyboardInterrupt:
        if model_person is not None:
        	model_person.close()
    	if model_bibdetect is not None:
        	model_bibdetect.close()
    	if model_bibrecog is not None:
        	model_bibrecog.close()

