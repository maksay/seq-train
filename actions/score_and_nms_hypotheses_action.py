from actions.base_action import BaseAction
from utils import Detection
from utils import Hypothesis
from utils import tracklet_IoU
from utils import IoU
from utils import interpolate_tracklet
from utils import hypotheses_IoU
from label_storage import LabelStorage
import numpy as np
from copy import deepcopy


# Multiple hypothesis tracker

class BaseScoreAndNMSHypothesesAction(BaseAction):
    def __init__(self, config):
        super(BaseScoreAndNMSHypothesesAction, self).__init__(config)

    #[Detection], Model, previous batch results -> [Hypothesis]
    # Previous batch results affect which pairwise hypothesis we can create -
    # only those not contradicting previous batch
    def do(self, detections, model, previous_batch):
        raise NotImplementedError()


class SimpleScoreAndNMSHypotheses(BaseScoreAndNMSHypothesesAction):
    def __init__(self, config):
        super(SimpleScoreAndNMSHypotheses, self).__init__(config)

    def do(self, detections, model, previous_batch, scene_start, scene_end, temp=1e-6):
        self.logger.info("Doing action")

        _detections = [det for det in detections]

        # Hypotheses containing only one detection
        unitary = []
        for d in _detections:
            tracklet = [None for _ in range(scene_start, scene_end)]
            tracklet[d.time - scene_start] = d
            h = Hypothesis(tracklet=tracklet)
            unitary.append(h)
        model.score(unitary)


        # During inference we can try to ignore hypothesis with very low scores
        # to speed up the process.
        if self.nms_option.endswith("ignore"):
            ignore_cutoff = float(self.nms_option.split('-')[-2])
        else:
            ignore_cutoff = -1e9

        # That includes ignoring very low quality detections.

        detections = []
        for did in range(len(unitary)):
            if unitary[did].score * (scene_end - scene_start) < ignore_cutoff:
                pass
            else:
                detections.append(_detections[did])

        # Creating pairwise hypotheses by interpolating between pairs of detections.

        pairwise_hypotheses = self.create_pairwise_hypotheses(detections,
                                                              scene_start,
                                                              scene_end)

        # Populating hypotheses array.
        # Hypotheses[1] = unitary detections
        # Hypotheses[l] = pairwise hypotheses between detections l-1 frames apart

        if len(detections) > 0:
            t_min = min([detection.time for detection in detections])
            t_max = max([detection.time for detection in detections])
            max_track_len = t_max - t_min + 1
        else:
            max_track_len = scene_end - scene_start + 1
        hypotheses = [[] for _ in range(max_track_len + 1)]

        for detection in detections:
            tracklet = [None for when in range(scene_start, scene_end)]
            tracklet[detection.time - scene_start] = detection
            hypotheses[1].append(Hypothesis(tracklet))

        for hypothesis in pairwise_hypotheses:
            h_start = min([det.time
                           for det in hypothesis.tracklet
                           if det is not None])
            h_end = max([det.time
                         for det in hypothesis.tracklet
                         if det is not None])
            hypotheses[h_end - h_start + 1].append(hypothesis)

        active_pairwise = []

        all_hypos = []

        for track_len in range(1, len(hypotheses)):
            self.logger.info("Working with track_len %d", track_len)

            # Compute scores for all hypotheses of the current length

            model.score(hypotheses[track_len])
            all_hypos += hypotheses[track_len]
            # Run NMS - having at most 1 hypothesis of every length
            # with every starting point.
            hypotheses[track_len] = self.nms(hypotheses[track_len],
                                             scene_start,
                                             scene_end,
                                             temp)


            # After NMS, let's add the forced hypotheses from the prev batch:
            # This is required to make sure that we can at least somehow link
            # our current hypotheses to the ones in the previous batch.
            push_force = []
            for tracklet in previous_batch:
                if len(tracklet) == track_len:
                    full_tracklet = [None
                                     for when in range(scene_start, scene_end)]
                    for detection in tracklet:
                        full_tracklet[detection.time - scene_start] = detection
                    push_force.append(Hypothesis(full_tracklet))
            model.score(push_force)
            hypotheses[track_len] += push_force

            # Consider pairwise hypotheses which we will use to grow further.
            # In particular, if we have two pairwise hypotheses A---B and B---C,
            # they could be used to grow into A---B---C.
            for hypothesis in hypotheses[track_len]:
                sum_non_intp = 0
                for detection in hypothesis.tracklet:
                    if detection is None: continue
                    if not hasattr(detection, "interpolated"):
                        sum_non_intp += 1
                    elif not detection.interpolated:
                        sum_non_intp += 1
                if sum_non_intp == 2:
                    active_pairwise.append(hypothesis)

            # Create additional candidates for longer hypotheses. We will NMS then
            # when we process the hypotheses of that length in this loop.

            if track_len < max_track_len:
                merged = self.create_merged_hypotheses(hypotheses,
                                                       active_pairwise,
                                                       track_len + 1)
                for hypothesis in merged:
                    hypotheses[track_len + 1].append(hypothesis)

        return [hypothesis
                for track_len in range(1, max_track_len + 1)
                for hypothesis in hypotheses[track_len]], all_hypos

    def create_pairwise_hypotheses(self,
                                   detections,
                                   scene_start,
                                   scene_end):
        # This simply creates pairwise hypotheses for all pairs
        # of detections that are at most dt apart, and between
        # which we can interpolate so that IoU of hypotheses
        # is > 0. This is obviously an overcomplete set.
        self.logger.info("Creating pairwise hypotheses...")

        pairwise_hypothesis = []
        if len(detections) == 0:
            return pairwise_hypothesis

        detections = sorted(detections, key=lambda detection: detection.time)
        detections_by_time = [[detections[0]]]
        for detection in detections[1:]:
            if detection.time == detections_by_time[-1][0].time:
                detections_by_time[-1].append(detection)
            else:
                detections_by_time.append([detection])

        for idx_now, detections_now in enumerate(detections_by_time):
            for detections_nxt in detections_by_time[idx_now + 1:]:
                if len(detections_now) == 0 or len(detections_nxt) == 0:
                    continue
                if detections_nxt[0].time - detections_now[0].time >\
                        self.pairwise_max_dt:
                    break
                for detection_now in detections_now:
                    for detection_nxt in detections_nxt:

                        tracklet = interpolate_tracklet(detection_now,
                                                        detection_nxt)

                        bboxes = np.vstack([detection.bbox
                                            for detection in tracklet])

                        ious = IoU(bboxes[:-1], bboxes[1:])
                        if np.min(ious) >= self.pairwise_min_iou:
                            tracklet_fin = [None for when in range(scene_start,
                                                                   scene_end)]
                            for did, det in enumerate(tracklet):
                                tracklet_fin[det.time - scene_start] = det
                                if did > 0 and did < len(tracklet) - 1:
                                    det.confidence = \
                                        LabelStorage.instance.min_det_confidence
                            pairwise_hypothesis.append(Hypothesis(
                                tracklet=tracklet_fin))

        self.logger.info("Done: total %d", len(pairwise_hypothesis))
        return pairwise_hypothesis

    def create_merged_hypotheses(self, hypotheses, active_pairwise, track_len):
        # Merge hypotheses of current length with all possible pairwise hypotheses
        # to create candidates of hypotheses of longer lengths.
        # To do that, we consider hypotheses A----B, and pairwise hypotheses B--C,
        # to create A---B--C.
        self.logger.info("Creating merged hypotheses")
        out = []

        heads = [{} for _ in range(track_len)]

        for cur_len in range(1, track_len):
            for h in hypotheses[cur_len]:
                ptr = -1
                while h.tracklet[ptr] is None:
                    ptr -= 1
                detection = h.tracklet[ptr]

                if detection not in heads[cur_len].keys():
                    heads[cur_len][detection] = []
                heads[cur_len][detection].append(h)

        for h in active_pairwise:
            ptr = 0
            while h.tracklet[ptr] is None:
                ptr += 1
            detection = h.tracklet[ptr]
            tail_ptr = len(h.tracklet) - 1
            while h.tracklet[tail_ptr] is None:
                tail_ptr -= 1
            need_len = track_len - (tail_ptr - ptr + 1) + 1
            if detection in heads[need_len].keys():
                for head in heads[need_len][detection]:
                    new_tracklet = [det for det in head.tracklet]
                    for did, d in enumerate(h.tracklet):
                        if d is not None:
                            try:
                                new_tracklet[did] = d
                            except:
                                print("Here")
                    out.append(Hypothesis(new_tracklet))

        self.logger.info("Done: total %d", len(out))
        return out

    def nms(self, hypotheses, scene_start, scene_end, temp):
        # Non maximum suppression. Temp parameter allows us to select
        # not the best candidate, but probabilistically.
        self.logger.info("Applying non-maximum supression")

        hashes = {}
        if self.nms_option.endswith("ignore"):
            ignore_cutoff = float(self.nms_option.split('-')[-2])
        else:
            ignore_cutoff = -1e9

        for h in hypotheses:
            head_ptr = 0
            while h.tracklet[head_ptr] is None:
                head_ptr += 1
            tail_ptr = -1
            while h.tracklet[tail_ptr] is None:
                tail_ptr -= 1

            real_length = (len(h.tracklet) + tail_ptr) - head_ptr + 1

            score_ratio = (scene_end - scene_start) * 1. / real_length
            if h.score * score_ratio < ignore_cutoff:
                continue

            if self.nms_option.startswith("start"):
                val = h.tracklet[head_ptr]
            else:
                val = (h.tracklet[head_ptr], h.tracklet[tail_ptr])
            if val not in hashes.keys():
                hashes[val] = h
            elif hashes[val].score < h.score:
                hashes[val] = h
            elif hashes[val].score > h.score and temp > 1e-5: 
                if np.random.uniform() < np.exp((h.score - hashes[val].score) / temp):
                    # Updating by a lower-scored hypotheses, simulated annealing style
                    hashes[val] = h

        self.logger.info("Done: %d remains", len(hashes))
        return list(hashes.values())
