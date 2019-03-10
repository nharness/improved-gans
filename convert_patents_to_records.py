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

#    dirs = glob("/home/ian/imagenet/ILSVRC2012_img_train_t1_t2/n*")
#    assert len(dirs) == 1000, len(dirs)
#    dirs = [d.split('/')[-1] for d in dirs]
#    dirs = sorted(dirs)
#    str_to_int = dict(zip(dirs, range(len(dirs))))


    outfile = '/home/nharness/idcgans/datasets/patents' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for i in enumerate(files):
        print(i[0])
        image = get_image(i[1], IMSIZE, is_crop=True, resize_w=IMSIZE)
#        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE), image.shape
#        image += 1.
#        image *= (255. / 2.)
        image = image.astype('bool')
#        image *= 255
        #print image.min(), image.max()
        # from pylearn2.utils.image import save
        # save('foo.png', (image + 1.) / 2.)
        image_raw = image.tostring()
#       class_str = f.split('/')[-2]
#       label = str_to_int[class_str]
        if i[0] % 10 == 0:
            print(i[0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(1),
            'image_raw': _bytes_feature(image_raw),
 #           'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    tf.app.run()

