import os

import cv2
import numpy as np
import scipy.misc
from PIL import Image
from PIL import ImageDraw, ImageFont


def paintTrainData():
    font = ImageFont.truetype("../font/" + "OCR-B 10 BT.ttf", 64)

    txt = "0123456789X"
    for i in range(11):
        t = txt[i:i + 1]
        cdir = './data/train/' + t
        if not os.path.exists(cdir):
            os.makedirs(cdir)
        image = Image.new("RGBA", (64, 64), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), t, (0, 0, 0), font=font)
        gray = grayImg(np.asarray(image))

        img = np.asarray(image.convert('L'))
        img = crop_image(img, 80)
        square = squareImage(Image.fromarray(img))
        scipy.misc.imsave(cdir + "/0.png", square)
        # image.save(cdir + "/0.png", "PNG")
        tdir = './data/test/' + t
        if not os.path.exists(tdir):
            os.makedirs(tdir)
        scipy.misc.imsave(tdir + "/0.png", square)

        img = crop_image(gray, 80)
        square = squareImage(Image.fromarray(img))
        # scipy.misc.imsave(cdir + "/1.png", square)
        scipy.misc.imsave(tdir + "/1.png", square)


def grayImg(img):
    # 转化为灰度图
    gray = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    retval, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return gray


def crop_image(img, tol=0):
    mask = img < tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def squareImage(image, size=(64, 64)):
    wh1 = image.width / image.height
    wh2 = size[0] / size[1]
    newsize = ((int)(size[1] * wh1), (int)(size[1]))
    if wh1 > wh2:
        newsize = ((int)(size[0]), (int)(size[0] / wh1))

    image = image.resize(newsize, Image.ANTIALIAS)
    img_padded = Image.new("L", size, 255)
    img_padded.paste(image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2)))
    return img_padded


paintTrainData()
