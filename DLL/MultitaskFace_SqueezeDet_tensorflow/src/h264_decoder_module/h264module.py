'''
    This modules is in charge of linking ffmpeg library
    and dependencies for decoding h264 bitstream files.
    It is an easy and fast implementation to extract RGB frames
    from encoded video with h264-standard format.

    For reference just check inter-process communication
'''

import subprocess as sp
import numpy

# Name of the binary file to be called
FFMPEG_BIN = "ffmpeg"


def init_h264dec(input_file):
    '''
        Command to be used as if we were on terminal
        the order of the arguments are:
            1. Binary file to be called to execute FFMPEG
            2. Input flag for ffmpeg
            3. File to be used as input
            4. Output file where we want the frames written
            5. File name, in our case a Pipe
            6. Pixel flag to tell ffmpeg how do we want the frames
                to be decoded
            7. Pixel format RGB24 -> 8bits per pixel
            8. Decoding flag
            9. Decoder to be used, in this case RAW, no formatting
            10. end of arguments
    '''
    command = [FFMPEG_BIN, '-i', input_file, '-f', 'image2pipe',
               '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']

    '''
        Pipe containing the decoded frames extracted
        from the h264 video sequence
    '''
    pic_pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=9 ** 8)
    return pic_pipe


def close_h264dec(pic_pipe):
    pic_pipe.terminate()


def get_frame(pic_pipe, width=1280, height=720, channels=3):
    # read width*height*channels bytes (= 1 frame)
    rgb_frame = pic_pipe.stdout.read(width * height * channels)

    # transform to numpy array to be used in tensorflow
    frame = numpy.fromstring(rgb_frame, dtype='uint8')

    if frame.shape[0] < (width*height*channels):
        print("No more frames to be decoded from this stream...")
        return frame

    frame = frame.reshape([height, width, channels])

    '''
        we don't need more data from the pipe, so we just
        flush it. This could be info needed by ffmpeg and not
        for us to use.
    '''
    pic_pipe.stdout.flush()

    return frame
