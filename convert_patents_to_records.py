from scipy.misc import imread
from random import shuffle
import time

import tensorflow as tf
from glob import glob
from utils import get_image, colorize
# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 512

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(argv):
    pattern = "/scratch/nharness/Library/RAW/*TIF"
    files = glob(pattern)
    assert len(files) > 0
    assert len(files) < 1000000, len(files)
    shuffle(files)

    outfile = '/home/nharness/idcgans/datasets/patents' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for i, f in enumerate(files):
        print i
        image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        #print image.min(), image.max()
        # from pylearn2.utils.image import save
        # save('foo.png', (image + 1.) / 2.)
        image_raw = image.tostring()
        label = 0
        if i % 1 == 0:
            print i, '\t',label
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    tf.app.run()

