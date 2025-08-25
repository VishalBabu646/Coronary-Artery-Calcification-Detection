import numpy as np
import cv2
import argparse
from keras.models import load_model


parser = argparse.ArgumentParser(
                    prog='Binary Segmentation Inference',
                    description='This program runs a keras model on x-ray coronary angiogram to extract vessel structures')
parser.add_argument('--source', '-s', default="./imgs/train/images/285.png")
parser.add_argument('--weights', '-w', default="./model.h5")
parser.add_argument('--output', '-o', default="out_binary_285.png")
args = parser.parse_args()

model = load_model(args.weights)

img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)
pred = model.predict(np.expand_dims(img, axis=0)/255)
cv2.imwrite(args.output, pred[0].argmax(axis=2)*255)
