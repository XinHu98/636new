from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing import image
from nets.ssd import SSD300
from nets.ssd_training import MultiboxLoss,Generator
from utils.utils import BBoxUtility
from keras.optimizers import Adam

import numpy as np
import pickle
import tensorflow as tf
import cv2
import keras

if __name__ == "__main__":
    log_path = "logs/"
    annotation = "train.txt"
    NUM_CLASS = 2
    input_shape = (300,300,3)

    f = open(annotation,"r")
    lines = f.readlines()

    np.random.shuffle(lines)
    split_rate = 0.8
    num_train = int(len(lines) * split_rate)
    num_val = len(lines) - num_train

    model = SSD300(input_shape, NUM_CLASS)
    model.load_weights("model_data/ssd_weights.h5", by_name = True, skip_mismatch=True)

    log = TensorBoard(log_dir = log_path)
    checkpoint = ModelCheckpoint(log_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    batch_size = 16
    priors = pickle.load(open('model_data/prior_boxes_ssd300.pkl','rb'))
    bbox_util = BBoxUtility(NUM_CLASS,priors)

    gen = Generator(bbox_util, batch_size, lines[:num_train],lines[num_train:],(input_shape[0],input_shape[1]), NUM_CLASS, do_crop=True)

    if True:
        model.compile(optimizer=Adam(1e-5),loss=MultiboxLoss(NUM_CLASS,neg_pos_ratio=2.0).compute_loss)
        model.fit_generator(gen.generate(True),
                            steps_per_epoch=num_train/batch_size,
                            validation_data=gen.generate(False),
                            validation_steps=num_val/batch_size,
                            epochs=30,
                            initial_epoch=0,
                            callbacks=[log, checkpoint, reduce_lr, early_stopping])





