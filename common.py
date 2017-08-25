import cv2
import maxflow
import numpy
from scipy import signal


# Taken from http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def hysteresis(absg, suppress, thi, tlo, allow=None):
    if suppress is not None:
        absg = numpy.where(suppress, 0, absg)
    absmax = numpy.amax(absg[1:-1, 1:-1])
    high = (absg >= absmax * thi)
    low = numpy.logical_and(absg >= absmax * tlo,
                            absg < absmax * thi)
    close_kernel = numpy.asarray([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]])
    close = signal.convolve2d(high, close_kernel)[1:-1, 1:-1]
    seedY, seedX = numpy.nonzero(numpy.logical_and(low, close))
    if allow is not None:
        # high = numpy.logical_and(high, allow)
        low = numpy.logical_and(low, allow)
    for i in range(0, len(seedY)):
        floodfill(seedX[i], seedY[i], high, low)
    return high


def floodfill(startX, startY, dest, src):
    queue = [(startX, startY)]
    while len(queue) > 0:
        centerX, centerY = queue[-1]
        queue = queue[:-1]
        for x in range(centerX - 1, centerX + 2):
            for y in range(centerY - 1, centerY + 2):
                if y >= 0 and x >= 0 and y < src.shape[0] and x < src.shape[1] and src[y, x]:
                    dest[y, x] = 1
                    src[y, x] = 0
                    queue.append((x, y))


def image_cut(source, sink, horizontal, vertical, c):
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes(source.shape)
    hStructure = numpy.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]])
    vStructure = numpy.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]])
    g.add_grid_edges(nodeids, weights=horizontal * c,
                     structure=hStructure,
                     symmetric=True)
    g.add_grid_edges(nodeids, weights=vertical * c,
                     structure=vStructure,
                     symmetric=True)
    g.add_grid_tedges(nodeids, source, sink)
    g.maxflow()
    return g.get_grid_segments(nodeids)


def algorithm1(img, thi=0.5, tlo=0.1, sigma=0.6,
               clist=[80], f=None, csearch=False, thin=False):
    # compute base binarizations and the stability curve
    bsd = numpy.zeros(len(clist))
    bimg = f(img, thi, tlo, sigma, clist, csearch=csearch, thin=thin)
    for ic in range(1, len(clist)):
        bsd[ic - 1] = numpy.sum(numpy.not_equal(bimg[ic], bimg[ic - 1])) / float(bimg[ic].size)

    # smooth stability curve
    if len(clist) > 1:
        d = smooth(bsd[:-1], 5)[2:-2]
    else:
        d = bsd[:-1]

    r = 0
    scr = None
    for i in range(0, d.size - 2):
        for j in range(i + 2, d.size):
            for k in range(i + 1, j):
                v = d[i] + d[j] - 2 * d[k]
                if scr is None or v > scr:
                    q = i
                    r = k
                    s = j
                    scr = v
    print('algorithm1 ' + str(thi) + ' weighted at ' + str(r) + ': ' + str(clist[r]))
    return bimg[r], clist[r]


def algorithm2(img, sigma=0.6, clist=None, tlo=0.1,
               thilist=[0.1, 0.6], f=None, iter=5, csearch=False, thin=False):
    diffs = []
    images = []
    previous = f(img, thilist[0], thilist[0] / 3.0, sigma, clist, csearch=csearch, thin=thin)[0]
    for i in range(1, iter + 1):
        thi = thilist[0] + (thilist[1] - thilist[0]) * i / float(iter)
        tlo = thi / 3.0
        current = f(img, thi, tlo, sigma, clist, csearch=csearch, thin=thin)[0]
        images.append(previous)
        diffs.append(numpy.sum(numpy.not_equal(current, previous)))
        previous = current
    diffs = numpy.asarray(diffs)
    diffs = smooth(diffs, 5)[2:-2]
    index = numpy.argmin(diffs)
    return images[index], clist[0], thilist[0] + (thilist[1] - thilist[0]) * index / float(iter)


#####################################################

def algorithm3(img, sigma=0.6, clist=None, tlo=0.1,
               thilist=[0.25, 0.5], f=None, csearch=False, thin=False):
    if clist is None:
        clist = numpy.exp(numpy.linspace(numpy.log(10),
                                         numpy.log(640), num=15))
    blo, clo = algorithm1(img, thilist[0], tlo, sigma, clist, f=f, csearch=csearch, thin=thin)
    bmid, cmid = algorithm1(img, numpy.mean(thilist), tlo,
                            sigma, clist, f=f, csearch=csearch, thin=thin)
    bhi, chi = algorithm1(img, thilist[1], tlo, sigma, clist, f=f, csearch=csearch, thin=thin)
    dlo = numpy.sum(numpy.not_equal(blo, bmid))
    dhi = numpy.sum(numpy.not_equal(bhi, bmid))

    if dlo < dhi:
        return blo, clo, thilist[0]
    else:
        return bhi, chi, thilist[1]


def find_background_mask(img, threshold=2.0):
    sr = 31
    img2 = (img - numpy.float_(cv2.GaussianBlur(img, (sr, sr), sr * 3, borderType=cv2.BORDER_REFLECT)))
    rms = numpy.sqrt(cv2.GaussianBlur(img2 * img2, (sr, sr), sr * 3, borderType=cv2.BORDER_CONSTANT))
    return ((img2 / (rms + 0.000000001)) > threshold)


def sort_range(low, high):
    if low < high:
        return [low, high]
    else:
        return [high, low]
