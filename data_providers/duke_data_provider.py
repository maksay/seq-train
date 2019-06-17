from data_providers.base_data_provider import BaseDataProvider
from utils import Detection
from scipy.io import loadmat
import numpy as np
import os
import h5py
import pickle
import subprocess


class DukeDataProvider(BaseDataProvider):
    def __init__(self, config):
        super(DukeDataProvider, self).__init__(config)

        self.hashval = str(self.frames) + "_" + str(self.frequency)
        if len(self.hashval) > 127:
            self.hashval = hash(self.hashval)
        self.frame_list = []
        for group in self.frames.split("),("):
            cam, st, fn = group.replace(")", "").replace("(", "").split(",")
            self.frame_list.append((int(cam), int(st), int(fn)))

    def detections(self):
        storage = 'tmp/duke_det_%s.pickle' % self.hashval
        self.logger.info("Storage should be %s", storage)

        if not os.path.exists(storage):
            self.logger.info("Reading detections")
            detpath="%s/DukeMTMC/detections/camera{0}.mat" % os.getcwd()

            det = []

            for cam, min_frame, max_frame in self.frame_list:

                detmat = h5py.File(detpath.format(cam))['detections'][
                             ()].transpose()[:, [1, 2, 3, 4, 5, -2, -1]]
                time = np.int32(detmat[:, 0] + 1e-9)

                detmat = detmat[time % self.thinning(cam) == 0, :]
                time = time[time % self.thinning(cam) == 0]
                detmat = detmat[time >= min_frame, :]
                time = time[time >= min_frame]
                detmat = detmat[time <= max_frame, :]
                time = time[time <= max_frame]

                detmat[:, 3] -= detmat[:, 1]
                detmat[:, 4] -= detmat[:, 2]

                for idx in range(len(time)):
                    modified_time = time[idx] / self.thinning(cam)
                    modified_time += self._camera_shift * cam
                    detection = Detection(bbox=detmat[idx, 1:5],
                                          time=modified_time,
                                          confidence=detmat[idx, -1:])
                    det.append(detection)

            pickle.dump(det, open(storage, 'wb'))

        return pickle.load(open(storage, 'rb'))

    def ground_truth(self):
        storage = 'tmp/duke_gt_%s.pickle' % self.hashval

        if not os.path.exists(storage):
            self.logger.info("Reading ground truth")
            gtpath = "%s/external/motchallenge-devkit/gt/DukeMTMCT/trainval.mat" % os.getcwd()

            _gtmat = loadmat(gtpath)['trainData']

            gt = []

            for cam, min_frame, max_frame in self.frame_list:

                gtmat = _gtmat[_gtmat[:, 0] == cam, :]
                min_t = max(min_frame, np.min(gtmat[:, 2]))
                max_t = min(max_frame, np.max(gtmat[:, 2]))
                gtmat = gtmat[gtmat[:, 2] >= min_t, :]
                gtmat = gtmat[gtmat[:, 2] <= max_t, :]

                track_count = 0

                for pid in np.unique(gtmat[:, 1]):
                    pos = np.where(gtmat[:, 1] == pid)[0]
                    bbox = gtmat[pos, 3: 7]
                    time = np.int32(gtmat[pos, 2] + 1e-9)
                    pos = np.where(time % self.thinning(cam) == 0)[0]

                    gt.append([])
                    track_count += 1

                    for idx in pos:
                        modified_time = time[idx] / self.thinning(cam)
                        modified_time += self._camera_shift * cam
                        detection = Detection(bbox[idx], modified_time)
                        if len(gt[-1]) == 0 or\
                                        gt[-1][-1].time == detection.time - 1:
                            gt[-1].append(detection)
                        else:
                            gt.append([detection])
            pickle.dump(gt, open(storage, 'wb'))

        return pickle.load(open(storage, 'rb'))

    def cam_and_time(self, detection):
        t = detection.time
        cam = t // self._camera_shift
        t %= self._camera_shift
        t *= self.thinning(cam)
        return cam, t

    def image_path(self, detection):
        cam, t = self.cam_and_time(detection)
        return "{0}/DukeMTMC/frames/camera{1}/{2}.jpg".format(os.getcwd(), cam, t)

    def evaluate(self, tracks):
        self.logger.info("Evaluation")

        if len(tracks) == 0:
            return 0.0, 0.0

        try:

            cam_id = self.cam_and_time(tracks[0][0])[0]

            tracks = self.det_list_to_numpy(tracks,
                                            interpolate=True)
            subprocess.call("mkdir -p %s/external/motchallenge-devkit/res/DukeMTMCT/debug/" % os.getcwd(), shell=True)
            files = ['%s/external/motchallenge-devkit/res/DukeMTMCT/debug/'
                     'debug-%s.txt' % (os.getcwd(), os.getpid())]

            for path in files:
                with open(path, 'w') as f:
                    for tid, track in enumerate(tracks):
                        for row in track:
                            f.write("%d,%d,%d,%d,%d,%d,0,0\n" % (
                                row[4], tid,
                                row[0] * self.imsize(cam_id)[0],
                                row[1] * self.imsize(cam_id)[1],
                                row[2] * self.imsize(cam_id)[0],
                                row[3] * self.imsize(cam_id)[1]
                            ))

            min_t = min([np.min(track[:,4]) for track in tracks])
            max_t = max([np.max(track[:,4]) for track in tracks])

            self.logger.info("Evaluation in cam %d, [%d:%d]",
                             cam_id, min_t, max_t)

            save_file = "%s/tmp/%s.eval" % \
                        (os.getcwd(), os.getpid())

            seq_file = '%s/external/motchallenge-devkit/seqmaps/' \
                       'DukeMTMCT-debug-%s.txt' % (os.getcwd(), os.getpid())
            with open(seq_file, 'w') as f:
                f.write("name\ndebug-%s\n" % os.getpid())

            line = "matlab -nodesktop -nosplash -nodisplay -r \"" \
                   "cd %s/external/motchallenge-devkit;" \
                   "CAM=%d;SID=%d;FID=%d;OUT='%s';" \
                   "seqmap='DukeMTMCT-debug-%s.txt';" \
                   "addpath(genpath('utils/'));" \
                   "compute_IDF_on_duke;exit();\"" % (
                       os.getcwd(), cam_id, min_t, max_t, save_file, os.getpid())

            subprocess.call(line, shell=True)
            with open(save_file, "r") as f:
                line = f.readlines()[0].rstrip().split(' ')

            return float(line[0]), float(line[1])
        except:
            self.logger.info("Did not succeed in running MOTChallenge MOTA/IDF evaluation")
            self.logger.info("Reporting MOTA")
            res = self.MOTA(tracks)
            return res, res

    def imsize(self, cam):
        return (1.0, 1.0)

    def fps(self, cam):
        return 60

    def save_tracks(self, tracks, file):
        cam_id = self.cam_and_time(self.detections()[0])[0]

        tracks = self.det_list_to_numpy(tracks,
                                        interpolate=True)

        with open(file, 'w') as f:
            for tid, track in enumerate(tracks):
                for row in track:
                    f.write("%d %d %d %d %d %d %d 0 0\n" % (
                        cam_id, tid, row[4],
                        row[0] * self.imsize(cam_id)[0],
                        row[1] * self.imsize(cam_id)[1],
                        row[2] * self.imsize(cam_id)[0],
                        row[3] * self.imsize(cam_id)[1]
                    ))

    def read_tracks(self, file):
        mat = np.loadtxt(file)
        tracks = []

        if len(mat) == 0:
            return tracks

        for pid in np.unique(mat[:, 1]):
            pos = np.where(mat[:, 1] == pid)[0]
            bbox = mat[pos, 3: 7]
            time = np.int32(mat[pos, 2] + 1e-9)

            cam = int(mat[pos[0], 0])
            cam_name = str(cam)
            pos = np.where(time % self.thinning(cam_name) == 0)[0]

            tracks.append([])

            for idx in pos:

                modified_time = time[idx] / self.thinning(cam)
                modified_time += self._camera_shift * cam
                detection = Detection(bbox[idx], modified_time)
                tracks[-1].append(detection)

        return tracks

