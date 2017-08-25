# coding:utf-8
# An implementation of Howe binarization for document images.

# References:

# A Laplacian Energy for Document Binarization, N. Howe.  International Conference on Document Analysis and Recognition, 2011.
# http://cs.smith.edu/~nhowe/research/pubs/divseg-icdar.pdf
#
# Document Binarization with Automatic Parameter Tuning, N. Howe.  To appear in International Journal of Document Analysis and Recognition. DOI: 10.1007/s10032-012-0192-x.
# http://cs.smith.edu/~nhowe/research/pubs/divseg-ijdar.pdf
#
# Matlab Code:  http://cs.smith.edu/~nhowe/research/code/

import argparse
import math
import os
import sys

import cv2
import numpy
from scipy import signal, ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import common

version = '1.0'


def binarize(image, sigma=0.6, crange=None, trange=[0.15, 0.6], csearch=False, thin=False):
    if csearch:
        a = 60
        b = 3000
        if crange is not None:
            crange = common.sort_range(crange[0], crange[1])
            a = crange[0]
            b = crange[1]
        clist = numpy.exp(numpy.linspace(numpy.log(a), numpy.log(b), num=25))
    else:
        clist = [300]
    trange = common.sort_range(trange[0], trange[1])
    result, c, thi = common.algorithm2(image, clist=clist,
                                       csearch=csearch,
                                       thilist=trange,
                                       sigma=sigma, iter=20,
                                       thin=thin,
                                       f=binarize_single)
    # print 'c=', c, 'thi=', thi
    return result


# Use convolution to get the difference between every pixel and a
# neighbor specified by the offset. Offset is (y, x) coordinate
def subtract_neighbor(image, offset):
    kernel = numpy.asarray([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
    kernel[1 + offset[0], 1 + offset[1]] = -1
    sign = signal.convolve2d(image, kernel)[1:-1, 1:-1]
    return sign


# Canny edge detection
# Based originally on: http://pythongeek.blogspot.com/2012/06/canny-edge-detection.html
#                      https://github.com/rishimukherjee/Canny-Python
def canny(image, thi=0.5, tlo=0.1, sigma=0.6):
    # Gaussian filter
    smoothed = ndimage.filters.gaussian_filter(image, sigma)

    # Sobel/Scharr convolution
    kernelx = numpy.asarray([[-3, 0, 3],
                             [-10, 0, 10],
                             [-3, 0, 3]])
    kernely = numpy.asarray([[-3, -10, -3],
                             [0, 0, 0],
                             [3, 10, 3]])

    gx = signal.convolve2d(smoothed, kernelx)[1:-1, 1:-1]
    gy = signal.convolve2d(smoothed, kernely)[1:-1, 1:-1]
    gmag = numpy.hypot(gx, gy)
    # Reflect it along the x-access because positive is down
    gdir = -numpy.arctan2(gy, gx)

    # Non-maximum suppression
    is_e = numpy.logical_or(numpy.logical_and(gdir < math.pi / 8,
                                              gdir >= -math.pi / 8),
                            numpy.logical_or(gdir >= 7 * math.pi / 8,
                                             gdir < -7 * math.pi / 8))
    is_ne = numpy.logical_or(numpy.logical_and(gdir < 3 * math.pi / 8,
                                               gdir >= math.pi / 8),
                             numpy.logical_and(gdir >= -7 * math.pi / 8,
                                               gdir < -5 * math.pi / 8))
    is_n = numpy.logical_or(numpy.logical_and(gdir < 5 * math.pi / 8,
                                              gdir >= 3 * math.pi / 8),
                            numpy.logical_and(gdir >= -5 * math.pi / 8,
                                              gdir < -3 * math.pi / 8))
    is_nw = numpy.logical_not(
        numpy.logical_or(is_e, numpy.logical_or(is_ne, is_n)))
    suppress_e = numpy.logical_or(subtract_neighbor(gmag, (0, 1)) < 0,
                                  subtract_neighbor(gmag, (0, -1)) < 0)
    suppress_ne = numpy.logical_or(subtract_neighbor(gmag, (-1, 1)) < 0,
                                   subtract_neighbor(gmag, (1, -1)) < 0)
    suppress_n = numpy.logical_or(subtract_neighbor(gmag, (1, 0)) < 0,
                                  subtract_neighbor(gmag, (-1, 0)) < 0)
    suppress_nw = numpy.logical_or(subtract_neighbor(gmag, (-1, -1)) < 0,
                                   subtract_neighbor(gmag, (1, 1)) < 0)

    suppress = numpy.logical_or(numpy.logical_or(numpy.logical_or(
        numpy.logical_and(is_e, suppress_e),
        numpy.logical_and(is_ne, suppress_ne)),
        numpy.logical_and(is_n, suppress_n)),
        numpy.logical_and(is_nw, suppress_nw))

    # Line tracing
    return common.hysteresis(gmag, suppress, thi, tlo)


def binarize_single(image, thi=0.5, tlo=0.1, sigma=0.6, clist=[100], csearch=True, thin=False):
    # Ensure image is grayscale
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = numpy.float_(image)

    # Compute Laplacian for source/sink weights
    lkernel = numpy.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
    laplacian = signal.convolve2d(image, lkernel)[2:-2, 2:-2]

    # Find edges and exclude them from adjacency weights
    edge_mask = canny(image, thi=thi, tlo=tlo, sigma=sigma)
    dx = image[:-1, 1:] - image[:-1, :-1]
    dy = image[1:, :-1] - image[:-1, :-1]

    if thin:
        hc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dx < 0),
                numpy.logical_and(edge_mask[:-1, 1:], dx >= 0)))[1:-1, 1:-1]
        vc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dy < 0),
                numpy.logical_and(edge_mask[1:, :-1], dy >= 0)))[1:-1, 1:-1]
    else:
        hc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dx > 0),
                numpy.logical_and(edge_mask[:-1, 1:], dx <= 0)))[1:-1, 1:-1]
        vc = numpy.logical_not(
            numpy.logical_or(
                numpy.logical_and(edge_mask[:-1, :-1], dy > 0),
                numpy.logical_and(edge_mask[1:, :-1], dy <= 0)))[1:-1, 1:-1]

    # Find high confidence background pixels
    background_mask = common.find_background_mask(image, threshold=1.5)[1:-1, 1:-1]

    result = []
    for c in clist:
        # Set source/sink weights
        if not csearch:
            weights = ((laplacian < 0) * c * -0.2) + ((laplacian >= 0) * c * 0.2)
        else:
            weights = laplacian
        weights = numpy.where(background_mask, 500, weights)[:-1, :-1]
        source = 1500 - weights
        sink = 1500 + weights

        # Partition the graph
        cut = numpy.int_(common.image_cut(source, sink, hc, vc, c))
        if thin:
            cut = numpy.logical_and(
                numpy.logical_and(cut[:-1, :-1], cut[1:, :-1]), cut[:-1, 1:])

        result.append(cut * 255)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Binarize document images using the Howe algorithm. Will search for the proper thi and c parameters. Optionally, fixed source/sink weights can be used instead of searching for c values to make it run faster. The resulting document will be very slightly smaller than the input document (3 pixels in each direction)')
    parser.add_argument('--version', action='version',
                        version='%(prog)s Version ' + version,
                        help='Get version information')
    parser.add_argument('--find-c', dest='find_c', default=False,
                        action='store_const', const=True,
                        help='Use variable weights for source/sink and search for the appropriate adjacency weight c. This is much slower but may yield better results.')
    parser.add_argument('--thin', dest='thin', default=False,
                        action='store_const', const=True,
                        help='Bias results to thin out letters. Reduces accuracy but improves readability.')
    parser.add_argument('--min-c', dest='min_c', default=60, type=int,
                        help='When searching for c, this is the minimum c value to look for. Defaults to 60')
    parser.add_argument('--max-c', dest='max_c', default=3000, type=int,
                        help='When searching for c, this is the maximum c value to look for. Defaults to 3000')
    parser.add_argument('--sigma', dest='sigma', default=0.6, type=float,
                        help='The level of smoothing done on the image before trying to find edges. Higher values reduce noise but may miss genuine edges. Defaults to 0.6')
    parser.add_argument('--min-thi', dest='min_thi', default=0.15, type=float,
                        help='Lowest thi value to search for during Canny edge detection. In each iteration, tlo is set to 1/3 of this value. Defaults to 0.15. Must be between 0 and 1.')
    parser.add_argument('--max-thi', dest='max_thi', default=0.6, type=float,
                        help='Highest thi value to search for during Canny edge detection. In each iteration, tlo is set to 1/3 of this value. Defaults to 0.6. Must be between 0 and 1.')
    parser.add_argument('input_file',
                        help='Path to input image file.')
    parser.add_argument('output_file',
                        help='Path to output image file.')
    options = parser.parse_args()
    if not os.path.exists(options.input_file):
        sys.stderr.write('howe: File not found: ' + options.input_file)
        exit(1)
    image = cv2.imread(options.input_file)
    result = binarize(image, sigma=options.sigma,
                      crange=[options.min_c, options.max_c],
                      trange=[options.min_thi, options.max_thi],
                      csearch=options.find_c,
                      thin=options.thin)
    cv2.imwrite(options.output_file, result)


if __name__ == '__main__':
    img = cv2.imread("test0.png")
    img2 = binarize(img)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(img2)
    plt.show()
    # main()
