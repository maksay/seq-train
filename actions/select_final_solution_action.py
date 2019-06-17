from actions.base_action import BaseAction
from label_storage import LabelStorage
import numpy as np
from utils import IoU
from utils import hypotheses_IoU
from utils import Detection
from utils import Hypothesis
from utils import interpolate_tracklet
from copy import deepcopy


class BaseSelectFinalSolutionAction(BaseAction):
    def __init__(self, config):
        super(BaseSelectFinalSolutionAction, self).__init__(config)

    #[Hypothesis] -> [[Detection]]
    def do(self, hypotheses, previous_batch=[]):
        raise NotImplementedError()


class GreedyFinalSolution(BaseSelectFinalSolutionAction):
    def __init__(self, config):
        super(GreedyFinalSolution, self).__init__(config)

    def do(self, hypotheses, previous_batch=[]):
        self.logger.info("Doing action")

        good_h = []
        for h in hypotheses:
            new_tracklet = [d for d in h.tracklet if d is not None]
            int_tracklet = [new_tracklet[0]]
            for d in new_tracklet[1:]:
                int_tracklet += interpolate_tracklet(int_tracklet[-1], d)[1:]
            good_h.append(Hypothesis(int_tracklet, h.score))
            good_h[-1].outputs = deepcopy(h.outputs)

        if len(good_h) == 0:
            return [], []

        tracks = []
        hypos = []
        goodhs = []

        scores = np.asarray([hypothesis.score for hypothesis in good_h])
        prev_ctr = 0
        while True:
            # First find the best match to anything in the previous batch
            # as these are the ones that must be there.
            if prev_ctr < len(previous_batch):
                new_scores = []
                for hid, h in enumerate(good_h):
                    matched = True
                    for did, det in enumerate(h.tracklet):
                        if did >= len(previous_batch[prev_ctr]):
                            break
                        if det is None:
                            matched = False
                            break
                        if det != previous_batch[prev_ctr][did]:
                            matched = False
                    if matched:
                        val = h.score
                        for track, hypo in zip(tracks, hypos):
                            if hypotheses_IoU(hypo, hypotheses[hid]) >=\
                                    self.iou_cutoff:
                                val = -1e9
                        new_scores.append(val)
                    else:
                        new_scores.append(-2e9)

                best_idx = np.argmax(new_scores)
                self.logger.debug("Matching previous batch with track %d"
                                 " with score %0.3f",
                                 best_idx, new_scores[best_idx])
                prev_ctr += 1
            else:
                # Otherwise simply pick the one with the best score
                best_idx = np.argmax(scores)
                if scores[best_idx] <= self.score_cutoff:
                    break
                self.logger.info("Selecting track %d with score %0.3f",
                                 best_idx, scores[best_idx])

            tracks.append(good_h[best_idx].tracklet)

            # Now put score of anything which overlaps with the selected
            # track to -infty
            hypos.append(hypotheses[best_idx])
            goodhs.append(good_h[best_idx])

            for hid, hypothesis in enumerate(hypotheses):
                if hid == best_idx or\
                                hypotheses_IoU(hypos[-1], hypothesis) >=\
                                self.iou_cutoff:
                    scores[hid] = -1e9
        return tracks, hypos
