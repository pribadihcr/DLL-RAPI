import time
from h264module import *
from PIL import Image

decoder = init_h264dec("file.h264")
SAVE_TO = "frames/frame_"
DEFAULT_H = 720
DEFAULT_W = 1280
DEFAULT_C = 3

for i in range(20000):
    start = time.time()
    frame = get_frame(decoder, width=DEFAULT_W,
                      height=DEFAULT_H, channels=DEFAULT_C)
    end = time.time()

    # check if we can continue with the decoding
    if (frame.shape[0]) != DEFAULT_H:
        break

    print("Time taken to decode frame %d was: %4.5f ms" % (i, 1000 * (end - start)))
    print("Frame of dimensions: %d X %d x %d" %
          (frame.shape[0], frame.shape[1], frame.shape[2]))
    # check-image
    j = Image.fromarray(frame, mode='RGB')
    j.save(SAVE_TO+str(i)+".jpeg")

close_h264dec(decoder)

