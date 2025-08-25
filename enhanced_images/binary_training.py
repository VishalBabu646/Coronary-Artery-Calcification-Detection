<<<<<<< HEAD
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from albumentations import ShiftScaleRotate, Flip, GridDistortion
from skimage import morphology
import os
from glob import glob
import cv2
from binary_model import model
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser( 
    prog='Binary Segmentation Training',
    description='Train U-Net model on coronary X-ray angiograms'
)
parser.add_argument('--path', '-p', default='imgs/train')
args = parser.parse_args()

# --- Data Generator ---
class MyGenerator(Sequence):
    def __init__(self, imgs, msks, weights, batch_size=2, to_fit=True, train=True, shuffle=True):
        self.batch_size = batch_size
        self.idxs = np.arange(len(imgs))
        self.to_fit = to_fit
        self.train = train
        self.shuffle = shuffle
        self.imgs = imgs
        self.msks = msks
        self.class_weights = weights

    def __len__(self):
        return int(np.ceil(len(self.idxs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def load_Xy(self, batch):
        X, y, w = [], [], []
        for i in batch:
            img_path = os.path.join(args.path, 'images', self.imgs[i])
            msk_path = os.path.join(args.path, 'masks', self.msks[i])

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(512, 512))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.squeeze(img).astype(np.uint8)

            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_not = cv2.bitwise_not(img)
            se = np.ones((50, 50), np.uint8)
            wth = morphology.white_tophat(img_not, se)
            raw_minus_topwhite = ((img.astype(int) - wth) > 0) * (img.astype(int) - wth)
            img = clahe.apply(raw_minus_topwhite.astype(np.uint8))
            img = img[:, :, np.newaxis]

            # Load and preprocess mask
            msk = tf.keras.preprocessing.image.load_img(msk_path, color_mode='grayscale', target_size=(512, 512))
            msk = tf.keras.preprocessing.image.img_to_array(msk)

            if self.train:
                # Apply augmentation
                aug = Flip(p=0.5)
                transform = aug(image=img, mask=msk)
                aug = ShiftScaleRotate(always_apply=True, border_mode=2)
                transform = aug(image=transform['image'], mask=transform['mask'])
                aug = GridDistortion(p=0.5)
                transform = aug(image=transform['image'], mask=transform['mask'])
                img, msk = transform['image'], transform['mask']

            img = img / 255.0
            msk = (msk > 0).astype(np.uint8)
            we = np.abs(np.abs(img - 1) - msk)

            X.append(img)
            y.append(msk)
            w.append(we)

        return np.array(X), np.array(y), np.array(w)

    def __getitem__(self, idx):
        batch = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.to_fit:
            return self.load_Xy(batch)

# --- Prepare data ---
image_paths = sorted(glob(os.path.join(args.path, 'images', '*')))
mask_paths = sorted(glob(os.path.join(args.path, 'masks', '*')))

imgs = np.array([os.path.basename(p) for p in image_paths])
msks = np.array([os.path.basename(p) for p in mask_paths])

print(f"Loaded {len(imgs)} images and {len(msks)} masks")

# --- Model compile ---
model = model()
model.compile(loss={'seg': tf.keras.losses.SparseCategoricalCrossentropy()},
              optimizer='Adam')

# --- Data split ---
train_gen = MyGenerator(imgs[:int(0.5 * len(imgs))], msks[:int(0.5 * len(imgs))], weights=[1, 80], batch_size=2)
valid_gen = MyGenerator(imgs[int(0.5 * len(imgs)):int(0.9 * len(imgs))], msks[int(0.5 * len(imgs)):int(0.9 * len(imgs))], weights=[1, 80], train=False)
test_gen = MyGenerator(imgs[int(0.9 * len(imgs)):], msks[int(0.9 * len(imgs)):], weights=[1, 80], train=False, shuffle=False)

# --- Callbacks ---
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# --- Train ---
model.fit(train_gen, validation_data=valid_gen, epochs=20, callbacks=[callback])
=======
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from albumentations import ShiftScaleRotate, Flip, GridDistortion
from skimage import morphology
import os
from glob import glob
import cv2
from binary_model import model
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser( 
    prog='Binary Segmentation Training',
    description='Train U-Net model on coronary X-ray angiograms'
)
parser.add_argument('--path', '-p', default='imgs/train')
args = parser.parse_args()

# --- Data Generator ---
class MyGenerator(Sequence):
    def __init__(self, imgs, msks, weights, batch_size=2, to_fit=True, train=True, shuffle=True):
        self.batch_size = batch_size
        self.idxs = np.arange(len(imgs))
        self.to_fit = to_fit
        self.train = train
        self.shuffle = shuffle
        self.imgs = imgs
        self.msks = msks
        self.class_weights = weights

    def __len__(self):
        return int(np.ceil(len(self.idxs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def load_Xy(self, batch):
        X, y, w = [], [], []
        for i in batch:
            img_path = os.path.join(args.path, 'images', self.imgs[i])
            msk_path = os.path.join(args.path, 'masks', self.msks[i])

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(512, 512))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.squeeze(img).astype(np.uint8)

            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_not = cv2.bitwise_not(img)
            se = np.ones((50, 50), np.uint8)
            wth = morphology.white_tophat(img_not, se)
            raw_minus_topwhite = ((img.astype(int) - wth) > 0) * (img.astype(int) - wth)
            img = clahe.apply(raw_minus_topwhite.astype(np.uint8))
            img = img[:, :, np.newaxis]

            # Load and preprocess mask
            msk = tf.keras.preprocessing.image.load_img(msk_path, color_mode='grayscale', target_size=(512, 512))
            msk = tf.keras.preprocessing.image.img_to_array(msk)

            if self.train:
                # Apply augmentation
                aug = Flip(p=0.5)
                transform = aug(image=img, mask=msk)
                aug = ShiftScaleRotate(always_apply=True, border_mode=2)
                transform = aug(image=transform['image'], mask=transform['mask'])
                aug = GridDistortion(p=0.5)
                transform = aug(image=transform['image'], mask=transform['mask'])
                img, msk = transform['image'], transform['mask']

            img = img / 255.0
            msk = (msk > 0).astype(np.uint8)
            we = np.abs(np.abs(img - 1) - msk)

            X.append(img)
            y.append(msk)
            w.append(we)

        return np.array(X), np.array(y), np.array(w)

    def __getitem__(self, idx):
        batch = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.to_fit:
            return self.load_Xy(batch)

# --- Prepare data ---
image_paths = sorted(glob(os.path.join(args.path, 'images', '*')))
mask_paths = sorted(glob(os.path.join(args.path, 'masks', '*')))

imgs = np.array([os.path.basename(p) for p in image_paths])
msks = np.array([os.path.basename(p) for p in mask_paths])

print(f"Loaded {len(imgs)} images and {len(msks)} masks")

# --- Model compile ---
model = model()
model.compile(loss={'seg': tf.keras.losses.SparseCategoricalCrossentropy()},
              optimizer='Adam')

# --- Data split ---
train_gen = MyGenerator(imgs[:int(0.5 * len(imgs))], msks[:int(0.5 * len(imgs))], weights=[1, 80], batch_size=2)
valid_gen = MyGenerator(imgs[int(0.5 * len(imgs)):int(0.9 * len(imgs))], msks[int(0.5 * len(imgs)):int(0.9 * len(imgs))], weights=[1, 80], train=False)
test_gen = MyGenerator(imgs[int(0.9 * len(imgs)):], msks[int(0.9 * len(imgs)):], weights=[1, 80], train=False, shuffle=False)

# --- Callbacks ---
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# --- Train ---
model.fit(train_gen, validation_data=valid_gen, epochs=20, callbacks=[callback])
>>>>>>> adc04752f570ee933d1e89f2246d06d473d94d7a
