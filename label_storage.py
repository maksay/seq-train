from utils import Hypothesis
from utils import IoU
from utils import Detection
from utils import crop_out_and_fit_bbox
import numpy as np
import os
import pickle
import cv2
import time


class LabelStorage(object):
    """ A class to compute the labels.
    Since number of possible sequences is exponentially large, GT must be
    computed on the fly for any given sequence.
    """

    class __LabelStorage(object):
        """ Internal class that works as a singleton. """
        def __init__(self, label_config, data_provider):
            self.storage = {}
            self.label_config = label_config
            self.data_provider = data_provider
            self.det = self.data_provider.detections()
            self.scene_size = self.data_provider.scene_size

            self.last_loaded_image = None
            self.last_loaded_image_time = -1
            self.pca = None

            self.det_by_time = {}
            self.gt_det_by_time = {}
            self.det_to_matching = {}

            for det in self.det:
                if det.time not in self.det_by_time.keys():
                    self.det_by_time[det.time] = []
                self.det_by_time[det.time].append(np.asarray([
                    det.bbox[0, 0] + det.bbox[0, 2] * 0.5,
                    det.bbox[0, 1] + det.bbox[0, 3] * 0.5,
                    det.bbox[0, 0], det.bbox[0, 1],
                    det.bbox[0, 2], det.bbox[0, 3],
                    det.confidence
                ]))

            self.gt = self.data_provider.ground_truth()
            for tid, track in enumerate(self.gt):
                for det in track:
                    if det.time not in self.gt_det_by_time.keys():
                        self.gt_det_by_time[det.time] = []
                    self.gt_det_by_time[det.time].append((tid, det))

            self.min_det_confidence = \
                self.data_provider.detections()[0].confidence
            self.max_det_confidence = \
                self.data_provider.detections()[0].confidence
            for det in self.data_provider.detections():
                self.min_det_confidence = np.minimum(self.min_det_confidence,
                                                     det.confidence)
                self.max_det_confidence = np.maximum(self.max_det_confidence,
                                                     det.confidence)

            self.social = {}

            self.app_feats = {}
            self.last_loaded_image = None
            self.last_loaded_image_time = -1

        def compute_app_embeddings(self, detections):
            """ Given a list of detections, computes the appearance for every
            one of them. If needed, the detections with unknown appearance are
            copied to a dir which is watched by another program which provides
            the appearance. For efficieny, appearance for any detection computed
            in the past is saved in a persistent storage. """
            root = self.label_config["app_dir"]

            # Load any app storages that have not been loaded yet
            for det in detections:
                app_hash_name = np.int32(det.time // 1000)
                if app_hash_name not in self.app_feats.keys():
                    app_storage_file = \
                        os.path.join(root,
                                     'storage',
                                     '{0}.pickle'.format(app_hash_name))
                    if not os.path.exists(app_storage_file):
                        pickle.dump({}, open(app_storage_file, 'wb'))

                    self.app_feats[app_hash_name] = pickle.load(
                        open(app_storage_file, 'rb'))

            # Create a list of new detections
            list_to_compute = []

            for det in detections:
                app_hash_name = np.int32(det.time // 1000)
                if det not in self.app_feats[app_hash_name].keys():

                    list_to_compute.append(det)
                    # Preventing duplicates
                    self.app_feats[app_hash_name][det] = None

            updated_storage = set()

            if len(list_to_compute) > 0:
                # Create a request
                request_dir = os.path.join(root,
                                           'requests',
                                           '{0}'.format(os.getpid()))
                os.makedirs(request_dir, exist_ok=True)

                for did, det in enumerate(list_to_compute):
                    if self.last_loaded_image_time != det.time:
                        self.last_loaded_image_time = det.time
                        impath = self.data_provider.image_path(det)
                        self.last_loaded_image = cv2.imread(impath)
                    crop = crop_out_and_fit_bbox(self.last_loaded_image,
                                                 det.bbox, None)
                    cv2.imwrite("%s/%08d.jpg" % (request_dir, did), crop)

                with open("%s/extract.features" % request_dir, "w") as f:
                    f.write("Go ahead\n")

                # Wait to finish and load
                while "{0}.done".format(os.getpid()) \
                        not in os.listdir(os.path.join(root, 'requests')):
                    time.sleep(1)

                featmat = np.loadtxt(
                    os.path.join(root,
                                 'requests',
                                 '{0}.features'.format(os.getpid())))
                featmat = featmat.reshape((len(list_to_compute), -1))

                os.remove(os.path.join(root,
                                       'requests',
                                       '{0}.features'.format(os.getpid())))
                os.remove(os.path.join(root,
                                       'requests',
                                       '{0}.done'.format(os.getpid())))

                for did, det in enumerate(list_to_compute):
                    app_hash_name = np.int32(det.time // 1000)
                    self.app_feats[app_hash_name][det] = featmat[did, :]
                    updated_storage.add(app_hash_name)

                print("computed %d new appearances" % len(list_to_compute))

            for k, v in self.app_feats.items():
                if len(v) > 0:
                    for k2, v2 in v.items():
                        app_feat_dim = len(v2)
                        break
                    break

            # Update persistent storage
            for app_hash_name in updated_storage:
                app_storage_file = \
                    os.path.join(root,
                                 'storage',
                                 '{0}.pickle'.format(app_hash_name))
                pickle.dump(self.app_feats[app_hash_name],
                            open(app_storage_file, 'wb'))

        def app_feat(self, det):
            """ Returns appearance feature of a detection. """
            app_hash_name = np.int32(det.time // 1000)
            if app_hash_name not in self.app_feats.keys():
                app_storage_file = os.path.join(
                        self.label_config["app_dir"],
                        'storage',
                        '{0}.pickle'.format(app_hash_name))
                self.app_feats[app_hash_name] = pickle.load(open(app_storage_file, 'rb'))
            data = self.app_feats[app_hash_name][det]
            return data

        def feature_dim(self):
            """ Feature dimension needed to construct the model. """
            det = self.data_provider.detections()[0]
            if 'appr' in self.label_config["features"]:
                self.compute_app_embeddings([det])
            self.get_features(det)
            return len(det.features)

        def get_features(self, detection):
            """ Computes the features for a given detection. """
            features = np.zeros((1, 0), dtype=np.float32)
            if 'bdif' in self.label_config["features"]:
                features = np.append(features, detection.bbox)
                features = np.append(features, detection.bbox)
            if 'bbox' in self.label_config["features"]:
                features = np.append(features, detection.bbox)
            if 'brel' in self.label_config["features"]:
                cam_id, _ = self.data_provider.cam_and_time(detection)
                imsize = self.data_provider.cam_size[cam_id]
                tmp = detection.bbox
                tmp[0, 0] /= imsize[0]
                tmp[0, 2] /= imsize[0]
                tmp[0, 1] /= imsize[1]
                tmp[0, 3] /= imsize[1]
                features = np.append(features, tmp)
            if 'conf' in self.label_config["features"]:
                features = np.append(features, detection.confidence)

            social_cnt = 0
            if 'soc1' in self.label_config["features"]:
                social_cnt = 1
            if 'soc3' in self.label_config["features"]:
                social_cnt = 3
            if 'soc5' in self.label_config["features"]:
                social_cnt = 5

            dens = np.zeros((1, 3), dtype=np.flot32)

            if social_cnt > 0 or 'dens' in self.label_config["features"]:
                if detection not in self.social.keys():
                    self.social[detection] = np.zeros((1, 3 * social_cnt))
                    if detection.time not in self.det_by_time.keys():
                        pass
                    else:
                        neighbours = np.asarray(
                            self.det_by_time[detection.time])
                        if len(neighbours) == 0:
                            pass
                        else:
                            dx = neighbours[:, 0] - \
                                 detection.bbox[0, 0] - \
                                 detection.bbox[0, 2] * 0.5
                            dy = neighbours[:, 1] - \
                                 detection.bbox[0, 1] - \
                                 detection.bbox[0, 3] * 0.5
                            dd = dx**2 + dy**2
                            if 'dens' in self.label_config["features"]:
                                dds = sorted(list(dd.reshape((-1,))))
                                if len(dds) < 20:
                                    dds += [0] * (20 - len(dds))
                                dens[0, 0] = dds[0]
                                dens[0, 1] = dds[4]
                                dens[0, 2] = dds[19]

                            for rep in range(min(len(neighbours), social_cnt)):
                                who = np.argmin(dd)
                                self.social[detection][0, 3*rep:3*rep+3] =\
                                np.asarray([dx[who],
                                            dy[who],
                                            neighbours[who, -1]])
                                dd[who] = 1e10

                features = np.append(features, self.social[detection])

            if 'dens' in self.label_config["features"]:
                features = np.append(features, dens)

            if 'intp' in self.label_config["features"]:
                if not hasattr(detection, "interpolated"):
                    detection.interpolated = False
                features = np.append(features,
                                     np.asarray([detection.interpolated]))
            if 'appr' in self.label_config["features"]:
                features = np.append(features, self.app_feat(detection))

            detection.features = features

        def get_hypo_features(self, hypotheses):
            """ Computes features for a given hypothesis (sequence). """
            all_dets = []
            for h in hypotheses:
                for detection in h.tracklet:
                    if detection is not None:
                        if detection.features is None:
                            all_dets.append(detection)

            all_dets = sorted(all_dets, key=lambda x:x.time)

            if 'appr' in self.label_config["features"]:
                self.compute_app_embeddings([det for det in all_dets 
                                             if det.features is None])

            for det in all_dets:
                if det.features is None:
                    self.get_features(det)

            feat_dim = self.feature_dim()
            for idx, h in enumerate(hypotheses):
                if hasattr(h, "features"):
                    continue

                h.features = np.zeros((self.scene_size, feat_dim),
                                      dtype=np.float32)
                h.input_bboxes = np.zeros((self.scene_size, 4),
                                          dtype=np.float32)
                h.input_values = np.zeros((self.scene_size,),
                                          dtype=np.float32)

                h_idx = 0
                while h.tracklet[h_idx] is None: h_idx += 1
                t_idx = len(h.tracklet) - 1
                while h.tracklet[t_idx] is None: t_idx -= 1

                for did, det in enumerate(h.tracklet):
                    if did < h_idx:
                        h.features[did, :] = 0
                        h.input_bboxes[did, :] = h.tracklet[h_idx].bbox
                        h.input_values[did] = 0
                    elif did > t_idx:
                        h.features[did, :] = 0
                        h.input_bboxes[did, :] = h.tracklet[t_idx].bbox
                        h.input_values[did] = 0
                    else:
                        h.features[did, :] = det.features
                        h.input_bboxes[did, :] = det.bbox
                        h.input_values[did] = 1

                if 'bdif' in self.label_config['features']:
                    h.features[1:, 0:4] -= h.features[:-1, 0:4]
                    h.features[0, 0:4] = 0.
                    h.features[:-1, 4:8] -= h.features[1:, 4:8]
                    h.features[-1, 4:8] = 0.

        def score(self, hypothesis, outputs):
            """ Given a hypothesis and model predictions about IoUs with GT and whether GT exists at all, computes a score. """
            top = 0
            bot = 0
            for did, det in enumerate(hypothesis.tracklet):
                if det is not None:
                    bot += 1
                if outputs["labels"][did] > 0.5:
                    bot += 1
                if det is not None and outputs["labels"][did] > 0.5:
                    top += outputs["ious"][did]
            return 2 * top * 1. / bot


        def matching_to_labeled_hypothesis(self, hypothesis, matching):
            """ Helper function for constructing the GT for a given input. """
            # matching is a list of (gt_idx, iou_list, bbox_list, label_list)
            labeled_hypothesis = Hypothesis(tracklet=hypothesis.tracklet)

            if len(matching) == 0:
                labeled_hypothesis.labels =\
                    np.zeros((self.scene_size,), dtype=np.float32)
                labeled_hypothesis.bboxes = \
                    np.zeros((self.scene_size, 4), dtype=np.float32)
                labeled_hypothesis.ious = \
                    np.zeros((self.scene_size,), dtype=np.float32)
                labeled_hypothesis.gt_idx = -1
                return labeled_hypothesis

            idfs = [sum(m[1]) for m in matching]

            best_match = np.argmax(np.asarray(idfs))
            gt_idx = matching[best_match][0]

            labeled_hypothesis.ious = matching[best_match][1]
            labeled_hypothesis.bboxes = matching[best_match][2]
            labeled_hypothesis.labels = matching[best_match][3]
            labeled_hypothesis.gt_idx = gt_idx

            return labeled_hypothesis

        def label_hypotheses(self, hypotheses, mode="one"):
            """ Computes GT (labels) for a list of hypotheses (sequences). """
            self.update_storage_two([hypothesis
                                     for hypothesis in hypotheses
                                     if hypothesis
                                     not in self.storage.keys()])
            return [self.storage[hypothesis][0] for hypothesis in hypotheses]

        def update_storage_two(self, hypotheses):
            """ Helper function to compute labels for hypotheses. For efficiency,
            if during execution a label was already computed for a hypothesis, it            is not computed again."""
            for hypothesis in hypotheses:
                for det in hypothesis.tracklet:
                    if det is None:
                        continue
                    if det not in self.det_to_matching:

                        self.det_to_matching[det] = []
                        if det.time not in self.gt_det_by_time.keys():
                            continue

                        gt_bboxes = np.asarray([gt_det.bbox for gt_id, gt_det
                                                in
                                                self.gt_det_by_time[det.time]]
                                               ).reshape((-1, 4))
                        det_bbox = np.tile(det.bbox, (len(gt_bboxes), 1))
                        ious = IoU(gt_bboxes, det_bbox)

                        for (gt_id, gt_det), iou in zip(
                                self.gt_det_by_time[det.time], ious):
                            if iou > 0:
                                self.det_to_matching[det].append((gt_id,
                                                                  gt_det,
                                                                  iou))

            for hypothesis in hypotheses:
                if hypothesis is self.storage.keys():
                    continue
                cands = []
                for det in hypothesis.tracklet:
                    if det is None:
                        continue
                    for (gt_id, _, _) in self.det_to_matching[det]:
                        cands.append(gt_id)
                cands = np.unique(np.asarray(cands))

                ious = np.zeros((len(cands), len(hypothesis.tracklet)),
                                dtype=np.float32)
                labs = np.zeros((len(cands), len(hypothesis.tracklet)),
                                dtype=np.float32)
                bboxes = np.zeros((len(cands), len(hypothesis.tracklet), 4),
                                  dtype=np.float32)

                times = []
                for did, det in enumerate(hypothesis.tracklet):
                    if det is not None:
                        for did2 in range(len(hypothesis.tracklet)):
                            times.append(det.time + did2 - did)

                for did, det in enumerate(hypothesis.tracklet):
                    for idx, cand in enumerate(cands):
                        if self.gt[cand][0].time <= \
                                times[did] <= \
                                self.gt[cand][-1].time:
                            try:
                                bboxes[idx, did, :] = \
                                    self.gt[cand][times[did] -
                                                  self.gt[cand][0].time].bbox
                                labs[idx, did] = 1
                            except:
                                continue

                    if det is None:
                        continue

                    for (gt_id, gt_det, iou) in self.det_to_matching[det]:
                        idx = np.where(cands == gt_id)[0][0]
                        ious[idx, did] = iou

                matching = []
                for cid, cand in enumerate(cands):
                    matching.append((cand,
                                     ious[cid, :],
                                     bboxes[cid, :, :],
                                     labs[cid, :]))

                self.storage[hypothesis] = \
                    (self.matching_to_labeled_hypothesis(hypothesis, matching),
                     matching)

        def storage(self):
            return self.storage

    """ Needed for LabelStorage to be a singleton. """

    instance = None

    def __new__(cls, label_config, data_provider):
        if LabelStorage.instance is None:
            LabelStorage.instance = LabelStorage.__LabelStorage(label_config,
                                                                data_provider)
        return LabelStorage.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)
