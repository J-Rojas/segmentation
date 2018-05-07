import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
import sys
import json
from pydoc import locate

from utilities import label_img_to_color

from model import ENet_model

project_dir = os.path.dirname(os.path.realpath(__file__))

model_id = "sequence_run"

project_dir = os.path.dirname(os.path.realpath(__file__))
config_file = sys.argv[1]

with open(config_file) as f:
    config = json.load(f)

data_dir = config['data_dir']
save_dir = config['save_dir']

# change this to not overwrite all log data when you train the model:
model_name = config["model"]
model_id = config["model_id"]
batch_size = config['batch_size']
img_height = 160
img_width = 320
subimg_x_offset = config['img_x_offset']
subimg_y_offset = config['img_y_offset']
subimg_height = config['img_height']
subimg_width = config['img_width']

model_class = locate(model_name)

model = model_class(config)

no_of_classes = model.no_of_classes

# load the mean color channels of the train imgs:
train_mean_channels = pickle.load(open("datasets/Cityscapes/data/mean_channels.pkl", "rb"))

# load the sequence data:
seq_frames_dir = data_dir
seq_frame_paths = []
frame_names = sorted(os.listdir(seq_frames_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print(step)

    frame_path = seq_frames_dir + frame_name
    seq_frame_paths.append(frame_path)

# compute the number of batches needed to iterate through the data:
no_of_frames = len(seq_frame_paths)
no_of_batches = int(no_of_frames/batch_size)

# define where to place the resulting images:
results_dir = model.project_dir + "/results_on_seq/"
if not(os.path.exists(results_dir)):
    os.makedirs(results_dir)

# create a saver for restoring variables/parameters:
saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    # restore the best trained model:
    saver.restore(sess, config['model_weights'])

    batch_pointer = 0
    for step in range(no_of_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        img_paths = []

        for i in range(batch_size):
            img_path = seq_frame_paths[batch_pointer + i]
            img_paths.append(img_path)

            # read the image:
            print(img_path)

            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (img_width, img_height))
            img = img - train_mean_channels
            batch_imgs[i] = img

        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
                    early_drop_prob=0.0, late_drop_prob=0.0)

        # run a forward pass and get the logits:
        logits = sess.run(model.logits, feed_dict=batch_feed_dict)

        print("step: %d/%d" % (step+1, no_of_batches))

        # save all predicted label images overlayed on the input frames to results_dir:
        predictions = np.argmax(logits, axis=3)
        for i in range(batch_size):
            pred_img = predictions[i]
            pred_img_color = label_img_to_color(pred_img)

            img = batch_imgs[i] + train_mean_channels

            img_file_name = img_paths[i].split("/")[-1]
            img_name = img_file_name.split(".png")[0]
            pred_path = results_dir + img_name + "_pred.png"

            overlayed_img = 0.3*img + 0.7*pred_img_color

            cv2.imwrite(pred_path, overlayed_img)

# create a video of all the resulting overlayed images:
fourcc = cv2.CV_FOURCC("M", "J", "P", "G")
out = cv2.VideoWriter(results_dir + "cityscapes_stuttgart_02_pred.avi", fourcc,
            20.0, (img_width, img_height))

frame_names = sorted(os.listdir(results_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print(step)

    if ".png" in frame_name:
        frame_path = results_dir + frame_name
        frame = cv2.imread(frame_path, -1)

        out.write(frame)
