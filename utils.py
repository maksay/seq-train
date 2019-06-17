import numpy as np
from munkres import Munkres
from sklearn.metrics.pairwise import pairwise_distances
import cv2
import tensorflow as tf



class Detection(object):
    def __init__(self, bbox, time, confidence=0.):
        """ Detection object featuring bounding box, confidence, and detection time. """
        self.bbox = np.float32(bbox).reshape((1, 4))
        self.time = np.int32(time)
        self.confidence = np.asarray(confidence).reshape((1, -1))
        self.features = None
        self.interpolated = False

    def __hash__(self):
        return hash((self.time,
                     self.bbox[0, 0], self.bbox[0, 1],
                     self.bbox[0, 2], self.bbox[0, 3]))

    def __eq__(self, other):
        return self.time == other.time and\
               np.all(self.bbox == other.bbox)


class Hypothesis(object):
    """ Hypothesis (sequence) object containing list of detections, output (predictions of the model for every step), and associated hypothesis score."""
    def __init__(self, tracklet=[], score=0.0):
        self.tracklet = [detection for detection in tracklet]
        self.score = score
        self.output = None

    def __hash__(self):
        return hash(tuple([hash(detection) for detection in self.tracklet]))

    def __eq__(self, other):
        if len(self.tracklet) != len(other.tracklet):
            return False
        for d1, d2 in zip(self.tracklet, other.tracklet):
            if not d1.__eq__(d2):
                return False
        return True

# IoU for a pair of detections, detection sequences, hypotheses, etc.

def IoU(bbox_a, bbox_b):
    dx = (np.minimum(bbox_a[:, 0] + bbox_a[:, 2], bbox_b[:, 0] + bbox_b[:, 2]) - np.maximum(bbox_a[:, 0], bbox_b[:, 0]))
    dy = (np.minimum(bbox_a[:, 1] + bbox_a[:, 3], bbox_b[:, 1] + bbox_b[:, 3]) - np.maximum(bbox_a[:, 1], bbox_b[:, 1]))
    inter_area = (np.maximum(dx, 0) * np.maximum(dy, 0))
    total_area = bbox_a[:, 2] * bbox_a[:, 3] + bbox_b[:, 2] * bbox_b[:, 3]

    return inter_area / np.float32(total_area - inter_area)

def tracklet_IoU(bbox_a, bbox_b):

    dx = (np.minimum(bbox_a[:, :, 0] + bbox_a[:, :, 2],
                     bbox_b[:, :, 0] + bbox_b[:, :, 2]) -
          np.maximum(bbox_a[:, :, 0], bbox_b[:, :, 0]))
    dy = (np.minimum(bbox_a[:, :, 1] + bbox_a[:, :, 3],
                     bbox_b[:, :, 1] + bbox_b[:, :, 3]) -
          np.maximum(bbox_a[:, :, 1], bbox_b[:, :, 1]))
    inter_area = np.sum(np.maximum(dx, 0) * np.maximum(dy, 0), axis=1)
    area_a = np.sum(bbox_a[:, :, 2] * bbox_a[:, :, 3], axis=1)
    area_b = np.sum(bbox_b[:, :, 2] * bbox_b[:, :, 3], axis=1)

    return inter_area / np.minimum(area_a, area_b)

def hypotheses_output_to_tracklet(h_a):
    tracklet_a = []
    for pos in range(len(h_a.tracklet)):
        if h_a.outputs["labels"][pos] > 0.5 and h_a.tracklet[pos] is not None:
            if len(tracklet_a) == 0:
                tracklet_a.append(Detection(h_a.outputs["bboxes"][pos], pos))
            else:
                tracklet_a += \
                    interpolate_tracklet(tracklet_a[-1],
                                         Detection(h_a.outputs["bboxes"][pos],
                                                   pos))[1:]
    return tracklet_a


def hypotheses_IoU(h_a, h_b):

    tracklet_a = hypotheses_output_to_tracklet(h_a)
    tracklet_b = hypotheses_output_to_tracklet(h_b)

    if len(tracklet_a) == 0 or len(tracklet_b) == 0:
        return 0.0

    total_a_area = sum([max(0, det.bbox[0, 2]) *
                        max(0, det.bbox[0, 3]) for det in tracklet_a])
    total_b_area = sum([max(0, det.bbox[0, 2]) *
                        max(0, det.bbox[0, 3]) for det in tracklet_b])
    bbox_a = []
    bbox_b = []
    for d_a in tracklet_a:
        if tracklet_b[0].time <= d_a.time <= tracklet_b[-1].time:
            d_b = tracklet_b[d_a.time - tracklet_b[0].time]
            bbox_a.append(d_a.bbox)
            bbox_b.append(d_b.bbox)

    if len(bbox_a) == 0 or total_b_area < 1e-6 or total_b_area < 1e-6:
        return 0.0

    bbox_a = np.vstack(bbox_a)
    bbox_b = np.vstack(bbox_b)

    dx = (np.minimum(bbox_a[:, 0] + bbox_a[:, 2], bbox_b[:, 0] + bbox_b[:, 2])
          - np.maximum(bbox_a[:, 0], bbox_b[:, 0]))
    dy = (np.minimum(bbox_a[:, 1] + bbox_a[:, 3], bbox_b[:, 1] + bbox_b[:, 3])
          - np.maximum(bbox_a[:, 1], bbox_b[:, 1]))
    inter_area = np.sum((np.maximum(dx, 0) * np.maximum(dy, 0)))

    try:
        retval = inter_area / min(total_a_area, total_b_area)
    except:
        pass
    return inter_area / min(total_a_area, total_b_area)

# Simple MOTA implementation for local testing.
def MOTA(tr, gt, D):

    if len(tr) == 0:
        return 0, [[0], [sum([len(x) for x in gt])], [0], [0]]

    denom = sum([len(gtrack) for gtrack in gt])
    numer = 0

    time_tr = np.concatenate([x[:,-1] for x in tr])
    time_gt = np.concatenate([x[:,-1] for x in gt])

    time_moments = np.sort(np.unique(np.concatenate([time_gt, time_tr])))

    munk = Munkres()

    fpseq = np.zeros((len(time_moments),))
    mdseq = np.zeros((len(time_moments),))
    isseq = np.zeros((len(time_moments),))
    gtseq = np.zeros((len(time_moments),))


    activity_tr_mask = np.zeros((len(time_moments), len(tr)))
    for idx, track in enumerate(tr):
        common = np.intersect1d(time_moments, track[:,-1])
        posits = np.searchsorted(time_moments, common)
        activity_tr_mask[posits, idx] = 1

    activity_gt_mask = np.zeros((len(time_moments), len(gt)))
    for idx, track in enumerate(gt):
        common = np.intersect1d(time_moments, track[:,-1])
        posits = np.searchsorted(time_moments, common)
        activity_gt_mask[posits, idx] = 1

    def bbox_overlap(nx, ny):
        return IoU(nx.reshape((1, 4)), ny.reshape((1, 4)))[0]

    for idx_now, now in enumerate(time_moments):

        active_tr_idx = np.where(activity_tr_mask[idx_now, :] == 1)[0]
        active_gt_idx = np.where(activity_gt_mask[idx_now, :] == 1)[0]

        active_tr_pos = np.zeros((len(active_tr_idx), 4))
        active_gt_pos = np.zeros((len(active_gt_idx), 4))

        for num, idx in enumerate(active_tr_idx):
            pos = np.where(tr[idx][:,-1] == now)[0]
            active_tr_pos[num, :] = tr[idx][pos,0:4]

        for num, idx in enumerate(active_gt_idx):
            pos = np.where(gt[idx][:,-1] == now)[0]
            active_gt_pos[num, :] = gt[idx][pos,0:4]

        N = len(active_tr_idx)
        M = len(active_gt_idx)

        dist_mat = np.zeros((max(N, M), max(N, M)))
        if N > 0 and M > 0:
            dist_mat[0:N, 0:M] = pairwise_distances(active_tr_pos, \
                                                    active_gt_pos, \
                                                    metric=bbox_overlap)
            dist_mat[dist_mat < D] = 0

        assign = munk.compute(-dist_mat.copy())

        assign = [x for x in assign if dist_mat[x[0],x[1]] >= D]

        assign = [(active_tr_idx[x[0]], active_gt_idx[x[1]]) for x in assign]

        false_pos = N - len(assign)
        missed_dt = M - len(assign)

        id_switch = 0
        if (idx_now == 0):
            assign_old = assign
        else:
            old_map = np.zeros((len(tr),)) - 1
            for match in assign_old:
                old_map[match[0]] = match[1]
            for match in assign:
                if old_map[match[0]] >= 0 and old_map[match[0]] != match[1]:
                    id_switch += 1
            assign_old = assign

        numer += false_pos + missed_dt + id_switch
        fpseq[idx_now] = false_pos
        mdseq[idx_now] = missed_dt
        isseq[idx_now] = id_switch
        gtseq[idx_now] = len(assign) - id_switch

    return 1 - 1. * numer / denom, [fpseq, mdseq, isseq, gtseq]


def save_tracks(tracks, path):
    pass


# Interpolation of tracklet between the detections.
def interpolate_tracklet(d0, d1):
    out = [d0]
    for t in range(d0.time + 1, d1.time):
        dleft = t - d0.time
        dright = d1.time - t
        bbox = (d0.bbox * dright + d1.bbox * dleft) / (dright + dleft)
        out.append(Detection(bbox=bbox,
                             time=t))
        out[-1].interpolated = True
    out.append(d1)
    return out


def crop_out_and_fit_bbox(img, _bbox, image_size):
    bbox = np.int32(_bbox.reshape((4,)))
    if bbox[0] < 0:
        bbox[2] -= bbox[0]
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[3] -= bbox[1]
        bbox[1] = 0
    if bbox[0] + bbox[2] >= img.shape[1]:
        bbox[2] = img.shape[1] - bbox[0]
    if bbox[1] + bbox[3] >= img.shape[0]:
        bbox[3] = img.shape[0] - bbox[1]

    crop = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
    if image_size is None:
        return crop

    scale_factor = max(crop.shape[0] * 1. / image_size,
                       crop.shape[1] * 1. / image_size)
    crop = cv2.resize(crop, (min(image_size, int(crop.shape[1] / scale_factor)),
                             min(image_size, int(crop.shape[0] / scale_factor))))
    full = np.zeros((image_size, image_size, 3), dtype=np.float32)
    dx = (image_size - crop.shape[0]) // 2
    dy = (image_size - crop.shape[1]) // 2
    full[dx:dx + crop.shape[0], dy:dy + crop.shape[1], :] = crop
    return full


def log_histogram(values, bins=1000):
    """Logs the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(0)
    hist.max = float(1)
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins,
    # where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/
    # master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist

# Utilities for splitting / merging tracks.
def split_tracks(tracks, dt=5, diou=0.5):
    out = []
    for track in tracks:
        head = 0
        idx = dt
        while idx < len(track):
            if IoU(track[idx - dt].bbox, track[idx].bbox) < diou:
                if idx - head > dt:
                    out.append(track[head:idx])
                    #print("Split %d" % len(out[-1]))
                else:
                    #print("Cut %d" % (idx - dt - head))
                    pass
                head = idx
                idx += dt
            else:
                idx += 1
        if head < len(track):
            out.append(track[head:])
    return out


def merge_tracks(tracks, dt=5, diou=0.5):
    out = []
    tracks = sorted(tracks, key=lambda x: x[0].time)
    for t2 in tracks:
        merge = False
        for idx in range(len(out)):
            t1 = out[idx]
            if np.abs(t1[-1].time - t2[0].time) <= dt:
                if IoU(t1[-1].bbox, t2[0].bbox) > diou:
                    #print("Merge!")
                    if t2[0].time > t1[-1].time:
                        out[idx] += interpolate_tracklet(t1[-1], t2[0])[1:-1]
                    else:
                        ptr = 0
                        while ptr < len(t2) and t2[ptr].time <= t1[-1].time:
                            ptr += 1
                        if ptr < len(t2):
                            out[idx] += t2[ptr:]
                    merge = True
                    break
        if not merge:
            out.append(t2)
    return out

