# coding: utf-8

from PIL import Image


def removeBackground(image):
    width, height = image.size
    data = image.load()
    # Iterate through the columns.
    start = -1
    end = -1
    for y in range(height):
        curC = 0
        for x in range(width):
            if data[x, y][0] < 100:
                curC += 1
        if curC > 0:
            if start == -1:
                start = y
        else:
            if end == -1 and start != -1:
                end = y - 1
            if start != -1 and end != -1 and end > start:
                bbox = (0, start, width, end)
                return image.crop(bbox)

    return image

def split_image(image):
    res = []
    width, height = image.size
    data = image.load()
    # Iterate through the columns.
    start = -1
    end = -1
    for x in range(width):
        curC = 0
        for y in range(height):
            if data[x, y][0] < 100:
                curC += 1
        if curC > 0:
            if start == -1:
                start = x
        else:
            if end == -1 and start != -1:
                end = x - 1
            if start != -1 and end != -1 and end > start:
                bbox = (start, 0, end, height)
                res.append(removeBackground(image.crop(bbox)))
                start = -1
                end = -1

    return res

