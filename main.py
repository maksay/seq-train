from actions import make_action
from utils import merge_tracks
from utils import split_tracks
from utils import tracklet_IoU
import sys
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import spline
from utils import log_histogram
import gc
from data_providers import make_data_provider
from models import make_model
from label_storage import LabelStorage
import numpy as np
from utils import IoU
from utils import hypotheses_IoU
from utils import interpolate_tracklet
from utils import Hypothesis
import logging
from utils import Detection
import os
import tensorflow as tf
import pickle
import argparse
import time
from tqdm import tqdm
from copy import deepcopy
import json
import glob
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(1)

# List of all flags

base_label_config = {
    "features"           : "bdif-bbox-conf-soc3-intp",
    "score_fun"          : "idf_1",
    "output_fun"         : "iou_list",
    "app_dir"            : ""
}

base_data_provider_config = {
    "name"         : "DukeDataProvider",
    "frames"       : "(1,70000,71000)",
    "frequency"     : 0.33,
    "scene_size"   : 6,
    "_camera_shift": 1000000
}

base_model_config = {
    "name" : "BidiRNNIoUPredictorModel",
    "embedding_size"   : 300,
    "rnn_cell_size"    : 300,
    "dropout"          : 0.0,
    "l2_norm"          : 0.0,
    "batch_size"       : 100,
    "lr"               : 1e-3,
    "samples_per_epoch": 100000,
    "_eval_batch_size" : 100,
    "_save_every_secs" : 6000000,
    "predict_ious"     : 1,
    "predict_labels"   : 1,
    "predict_bboxes"   : 1,
    "layers"           : 1
}

base_final_solution_config = {
    "name"             : "GreedyFinalSolution",
    "score_cutoff"     : 2.5,
    "iou_cutoff"       : 0.30,
    "no_overlap_policy": 1
}

base_nms_config = {
    "name"            : "SimpleScoreAndNMSHypotheses",
    "pairwise_min_iou": 1e-9,
    "pairwise_max_dt": 3,
    "nms_option": "start"
}

TestConfig = {
    "label_config"               : deepcopy(base_label_config),
    "train_data_provider_config" : deepcopy(base_data_provider_config),
    "input_data_provider_config" : deepcopy(base_data_provider_config),
    "model_config"               : deepcopy(base_model_config),
    "final_solution_config"      : deepcopy(base_final_solution_config),
    "nms_config"                 : deepcopy(base_nms_config),
    "mode"                       : "test",
    "logging"                    : "consol",
    "experiment_name"            : "mot17_dpm_but3",
    "hardmine"                   : 1,
    "eval_runner_cnt"            : 0,
    "datapath"                   : "",
    "target_metric"              : "IDF"
}


parser = argparse.ArgumentParser()
for k, v in TestConfig.items():
    if type(v) is dict:
        for k2, v2 in v.items():
            parser.add_argument('--%s.%s' % (k, k2), default=v2, type=type(v2))
    else:
        parser.add_argument('--%s' % k, default=v, type=type(v))
args = parser.parse_args()
for arg in vars(args):
    k = arg
    v = getattr(args, k)
    print(k, "->", v)
    if '.' in k:
        config_name, config_var = k.split('.')
        TestConfig[config_name][config_var] = v
    else:
        TestConfig[k] = v

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(name)s\n'
                              '%(asctime)s: %(message)s')
ch.setFormatter(formatter)
log_path = "runs/%s/logs" % TestConfig["experiment_name"]
if not os.path.exists(log_path):
    os.makedirs(log_path)
fh = logging.FileHandler(os.path.join(log_path, '%s.log' % os.getpid()), "w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(levelname)10s    %(name)60s     '
                                  '%(asctime)30s    %(message)s'))

if "console" in TestConfig['logging'].split("-"):
    logger.addHandler(ch)
if "file" in TestConfig['logging'].split("-"):
    logger.addHandler(fh)


# Main procedure for running the model on the data.

def run_once(model,
             label_config,
             data_provider_config,
             nms_config,
             final_solution_config,
             debug=False):

    data_provider = make_data_provider(data_provider_config)

    detections = data_provider.detections()

    nms_action = make_action(nms_config)
    select_action = make_action(final_solution_config)

    LabelStorage.instance = None
    LabelStorage(label_config, data_provider)
    max_det_confidence = LabelStorage.instance.max_det_confidence

    # Operating under assumption of one continuous scene here!
    start_time = min([detection.time for detection in detections])
    finish_time = max([detection.time for detection in detections])
    final_tracks = []

    # Shifting by 1/3 of the batch length on every step.
    # In particular, in batch [1, 2, 3] we assume that
    # results in the first third (1) are already fixed
    # results in the second third (2) will be fixed after this batch
    # results in the third third (3) will be discarded and recomputed in the next batch.
    batch_len = data_provider.scene_size
    batch_shift = batch_len // 3

    all_observed_hypos = []
    all_selected_hypos = []

    for batch_start in tqdm(range(start_time, finish_time + 1, batch_shift)):

        logger.info("Starting batch @ %d, last @ %d",
                    batch_start, finish_time)

        batch_end = batch_start + batch_len

        # First let's gather all the detection in the current batch.
        # That includes all detection from the detector
        # + all detections obtained as the result of tracking from earlier batches
        # (remember, we shift the detections by the shift that is output by the model)

        batch_detections = [detection for detection in detections
                            if batch_start <= detection.time < batch_end]

        previous_batch = []
        for track in final_tracks:
            if track[-1].time >= batch_start:
                keep_len = min(len(track), track[-1].time - batch_start + 1)
                for det in track[-keep_len:]:
                    bbox = np.zeros((len(batch_detections), 4),
                                    dtype=np.float32)
                    bbox[:, :] = det.bbox
                    dbox = np.asarray([d.bbox
                                       for d
                                       in batch_detections]).reshape((-1, 4))
                    dtime = np.asarray([d.time
                                       for d
                                       in batch_detections])
                    ious = IoU(bbox, dbox)

                    tokeep = np.where(np.bitwise_or(
                        ious < select_action.iou_cutoff,
                        dtime != det.time))[0]
                    batch_detections = [
                        batch_detections[idx]
                        for idx in tokeep]

                if track[-1].time == batch_start + batch_shift - 1:
                    previous_batch.append(track[-keep_len:])

        for track in previous_batch:
            batch_detections.append(track[-1])

        # Generate a set of all hypotheses with MHT tracker

        hypotheses, all_hypos = nms_action.do(detections=batch_detections,
                                              model=model,
                                              previous_batch=previous_batch,
                                              scene_start=batch_start,
                                              scene_end=batch_end)
        all_observed_hypos += all_hypos

        # Greedily select the best set of hypotheses

        tracks, selected_hypos = \
            select_action.do(hypotheses=hypotheses,
                             previous_batch=previous_batch)

        all_selected_hypos += selected_hypos

        if batch_start == start_time and batch_end >= finish_time:
            final_tracks = tracks
            break

        # Merge current tracks with the results obtained from previous batches.
        taken = set()
        for track in final_tracks:
            if track[-1].time == batch_start + batch_shift - 1:
                mid = len(taken)
                taken.add(mid)
                tmp = tracks[mid]
                while len(track) > 0 and track[-1].time >= tmp[0].time:
                    track.pop()
                for det in tmp:
                    if len(track) == 0:
                        track.append(det)
                    else:
                        track += interpolate_tracklet(track[-1], det)[1:]

        # Add new trajectories that were not there in the previous batch.
        for mid in range(len(tracks)):
            if mid not in taken:
                if tracks[mid][0].time < batch_start + 2 * batch_shift or \
                                batch_end >= finish_time:
                    tmp = tracks[mid]
                    final_tracks.append(tmp)

        # Cut at 2/3 of the batch
        if batch_end <= finish_time:
            for track in final_tracks:
                while len(track) > 0 and\
                    track[-1].time >= batch_start + 2 * batch_shift:
                    track.pop()
        else:
            break

        final_tracks = [t for t in final_tracks if len(t) > 0]

        # Add the interpolated detections that were output by the model to the global set of detections.
        for track in final_tracks:
            ptr = len(track) - 1
            while ptr >= 0 and track[ptr].time >= batch_start:
                if track[ptr].time >= batch_start + batch_shift:
                    track[ptr].interpolated = False
                    track[ptr].confidence = max_det_confidence
                    detections.append(track[ptr])
                ptr -= 1

                        
    return [t for t in final_tracks if len(t) > 0],\
           all_observed_hypos,\
           all_selected_hypos


# Simple utility for logging arguments, and the code for logging arguments
def tostring(dic):
    out = ""
    for k in sorted(dic.keys()):
        if not k.startswith("_"):
            if len(out) > 0:
                out += ','
            out += k + '=' + str(dic[k])
    return out


label_json = tostring(TestConfig["label_config"])
model_json = tostring(TestConfig["model_config"])
train_json = tostring(TestConfig["train_data_provider_config"])
valid_json = tostring(TestConfig["input_data_provider_config"])

# Paths to all necessary resources (created if necessary)

gendata_stopfile = os.path.join("runs",
                                TestConfig["experiment_name"],
                                "gendata_stopfile")
ckpt_dir = os.path.join("runs", TestConfig["experiment_name"], "checkpoints")
datagen_ckpt_dir = \
    os.path.join("runs", TestConfig["experiment_name"], "datagen_checkpoints")
best_model_dir = os.path.join("runs", TestConfig["experiment_name"],
                              "best_model")
summ_dir = os.path.join("runs", TestConfig["experiment_name"],
                        "summaries", TestConfig["mode"])
data_dir = os.path.join("runs", TestConfig["experiment_name"], "data")

if TestConfig["datapath"] == "":
    dataset_dir = os.path.join("runs", TestConfig["experiment_name"], "dataset")
else:
    dataset_dir = TestConfig["datapath"]

for dir in [ckpt_dir, summ_dir, best_model_dir,
            data_dir, datagen_ckpt_dir, dataset_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

with open(os.path.join("runs", TestConfig["experiment_name"], "config"), "a")\
        as f:
    line = json.dumps(TestConfig, indent=4, sort_keys=True)
    f.write(line)
    f.write("\n")


# Training code
if TestConfig["mode"] == 'train':

    data_provider = make_data_provider(TestConfig["train_data_provider_config"])
    LabelStorage(label_config=TestConfig["label_config"],
                 data_provider=data_provider)

    # Create a model and possibly load it from the checkpoints
    model = make_model(TestConfig["model_config"], "infer",
                       tf.Session(), ckpt_dir)
    fw = tf.summary.FileWriter(summ_dir, graph=model.sess.graph)
    try:
        model.load_model()
        model.save_model()
    except:
        model.sess.run(tf.global_variables_initializer())
        model.save_model()

    # Read all the data that was prepared for training
    labeled_hypotheses = [[] for _ in range(10)]
    fls = os.listdir(dataset_dir)
    use_data_ptr = False
    if fls[0].endswith("data_ptr"):
        use_data_ptr = True
        print("Reading from data_ptr file")
        with open(os.path.join(dataset_dir, fls[0]), "r") as f:
            fls = [line.rstrip() for line in f.readlines()]
        print("First file %s" % fls[0])


    cur_hypotheses = []
    for path in fls:
        if not use_data_ptr:
            cur_hypotheses = pickle.load(
                open(os.path.join(dataset_dir, path), "rb"))
        else:
            cur_hypotheses = pickle.load(open(path, "rb"))

        for lh in cur_hypotheses:
            gtlen = np.sum(lh.labels)
            sqlen = 0
            mtlen = 0
            for idx in range(len(lh.tracklet)):
                if lh.tracklet[idx] is not None:
                    cam_id, _ = data_provider.cam_and_time(lh.tracklet[idx])
                    sqlen += 1.0
                    if lh.ious[idx] > 0.5:
                        mtlen += 1.0
            idf = 2 * mtlen * 1. / (gtlen + sqlen)
            bucket = max(0, min(9, int(idf * 10)))
            lh.idf = idf

            # clearing features
            delattr(lh, "features")
            for idx in range(len(lh.tracklet)):
                if lh.tracklet[idx] is not None:
                    lh.tracklet[idx].features = None

            labeled_hypotheses[bucket].append(lh)
        print("Loaded %d from %s" % (len(cur_hypotheses), path))

    epoch = 0

    while True:
        np.random.seed(epoch)
        min_bucket_size = min([len(x) for x in labeled_hypotheses])

        # Hardmining to select the balanced dataset
        for idx in range(10):

            # First sort at random
            order = np.arange(len(labeled_hypotheses[idx]))
            np.random.shuffle(order)

            # Then take min_bucket_size x HARDMINE
            # and select min_bucket_size hardest
            to_take = min(len(order), min_bucket_size * TestConfig["hardmine"])
            _, losses = model.train_epoch(labeled_hypotheses[idx][:to_take], 0,
                                          do_train=False)
            order = np.argsort(np.asarray(losses))
            order = order[::-1]

            labeled_hypotheses[idx] = [labeled_hypotheses[idx][idx2]
                                       for idx2 in order]


        cur_dataset = []
        for idx in range(10):
            cur_dataset += [labeled_hypotheses[idx][idx2]
                            for idx2
                            in range(min(min_bucket_size,
                                         len(labeled_hypotheses[idx])))]

        # Train one epoch and wait a bit to ensure that all evaluation runners
        # had the time to run on the current checkpoint (otherwise a large backlog 
        # might be created)
        model.train_epoch(cur_dataset, epoch, do_train=True, do_save=True)
        print("Done epoch %d" % epoch)
        epoch += 1
        time.sleep(120)

# Generating the dataset - MASTER
if TestConfig["mode"] == 'gen_dataset':

    data_provider = make_data_provider(TestConfig["train_data_provider_config"])
    detections = data_provider.detections()
    ground_truth = data_provider.ground_truth()

    all_det_times = np.unique(np.asarray([
        det.time for det in detections]))

    LabelStorage(label_config=TestConfig["label_config"],
                 data_provider=data_provider)

    # Create and possibly load a model
    model = make_model(TestConfig["model_config"], "infer",
                       tf.Session(), datagen_ckpt_dir)
    fw = tf.summary.FileWriter(summ_dir, graph=model.sess.graph)
    try:
        model.load_model()
        model.save_model()
    except:
        model.sess.run(tf.global_variables_initializer())
        model.save_model()
    epoch = 0

    total_deleted_ever = 0

    size_history = []
    labeled_hypotheses = [[] for _ in range(10)]
    while(True):
        gc.collect()
        step = model.sess.run(model.global_step)

        # Read all the data generated by the slaves.
        # The slaves create two filex (X.pickle and X.pickle_flag)
        # As long as there is a pickle_flag, it is safe to read
        # the pickle file (the file is finished and can be read)

        fls = os.listdir(data_dir)
        added_hypotheses = []
        for path_flag in fls:
            if path_flag.endswith("_flag"):
                path = path_flag[:-5]
                cur_hypotheses = pickle.load(
                    open(os.path.join(data_dir, path), "rb"))
                added_hypotheses += cur_hypotheses
                os.remove(os.path.join(data_dir, path))
                os.remove(os.path.join(data_dir, path_flag))

        # computing IDF of each of the read hypotheses
        # to divide them into 10 buckets according to IDF
        for lh in added_hypotheses:
            gtlen = np.sum(lh.labels)
            sqlen = 0
            mtlen = 0
            for idx in range(len(lh.tracklet)):
                if lh.tracklet[idx] is not None:
                    sqlen += 1.0
                    if lh.ious[idx] > 0.5:
                        mtlen += 1.0
            idf = 2 * mtlen * 1. / (gtlen + sqlen)
            bucket = max(0, min(9, int(idf * 10)))
            lh.idf = idf
            labeled_hypotheses[bucket].append(lh)

        # If not enough training data was read, just leeep
        min_bucket_size = min([len(x) for x in labeled_hypotheses])
        if  min_bucket_size == 0:
            print("Not enough training data so far, waiting")
            time.sleep(30)
            continue


        np.random.seed(epoch)
        to_delete = []
        all_losses = []
        for idx in range(10):
            # Compute losses for every data sample and keep only the hardest ones
            _, losses = model.train_epoch(labeled_hypotheses[idx], 0,
                                          do_train=False)
            all_losses.append(losses)
            order = np.argsort(np.asarray(losses))
            order = order[::-1]


            #order = np.arange(len(labeled_hypotheses[idx]))
            #np.random.shuffle(order)

            labeled_hypotheses[idx] = [labeled_hypotheses[idx][idx2]
                                       for idx2 in order]
            # To avoid memory limit exceeded
            if len(labeled_hypotheses[idx]) > 20000:
                to_delete += [h for h in labeled_hypotheses[idx][20000:]]
                labeled_hypotheses[idx] = labeled_hypotheses[idx][:20000]

        min_bucket_size = min([len(x) for x in labeled_hypotheses])

        cur_dataset = []
        for idx in range(10):
            cur_dataset += [labeled_hypotheses[idx][idx2]
                            for idx2
                            in range(min(min_bucket_size,
                                         len(labeled_hypotheses[idx])))]

        # Report the summaries for the data generation process.

        sl = sum([len(x) for x in labeled_hypotheses])

        summary = tf.Summary()
        summary.value.add(tag='data idf',
                          histo=log_histogram(
                              np.asarray([lh.idf
                                          for lhypos in labeled_hypotheses
                                          for lh in lhypos]),
                              bins=10))
        summary.value.add(tag='mean loss',
                          simple_value=np.mean(np.concatenate(all_losses)))
        summary.value.add(tag='min bucket', simple_value=min_bucket_size)
        summary.value.add(tag="train data size", simple_value=len(cur_dataset))
        summary.value.add(tag="active data size", simple_value=sl)
        summary.value.add(tag="deleted data size", simple_value=len(to_delete))
        total_deleted_ever += len(to_delete)
        summary.value.add(tag="observed data size",
                          simple_value=sl + total_deleted_ever)
        fw.add_summary(summary, step)
        fw.flush()

        # Save the data samples
        pickle.dump(to_delete,
                    open(os.path.join(dataset_dir,
                                      "deleted_%d.pickle" % step), "wb"))

        pickle.dump([x for lhypos in labeled_hypotheses for x in lhypos],
                    open(os.path.join(dataset_dir, "remaining.pickle"),
                         "wb"))

        # Train the model on the hardest samples taken.
        model.train_epoch(cur_dataset, epoch, do_train=True, do_save=True)
        print("Done epoch %d" % epoch)
        epoch += 1
        size_history.append(sl + total_deleted_ever)
        # If there was a significant dataset growth, restarting the model training.
        if size_history[-1] >= 1.5 * size_history[0]:
            print("Dataset has grown at least 1.5x, clearing the model")
            step = model.sess.run(model.global_step)
            model.sess.run(tf.global_variables_initializer())
            model.sess.run(model.global_step.assign(step))
            size_history = size_history[-1:]

        # If there was less than 5% growth in 10 steps, stopping the training.
        if os.path.exists(gendata_stopfile) or\
                (len(size_history) > 10 and\
                                 size_history[-11] * 1.05 > size_history[-1]):
            print("Dataset growth less than 5% in 10 steps or stopfile, ending")
            break

# Generating data - SLAVE
if TestConfig["mode"] == 'gen_data':

    data_provider = make_data_provider(TestConfig["input_data_provider_config"])
    detections = data_provider.detections()
    ground_truth = data_provider.ground_truth()

    det_by_time = {}
    for det in detections:
        if det.time not in det_by_time.keys():
            det_by_time[det.time] = []
        det_by_time[det.time].append(det)

    start_time = min([detection.time for detection in detections])
    finish_time = max([detection.time for detection in detections])
    batch_len = data_provider.scene_size

    LabelStorage(label_config=TestConfig["label_config"],
                 data_provider=data_provider)

    model = make_model(TestConfig["model_config"], "infer",
                       tf.Session(), ckpt_dir)

    nms_action = make_action(TestConfig["nms_config"])

    all_hypotheses_ever = set()

    last_step = -1
    while True:
        # Try to load the model from the latest checkpoint
        try:
            model.load_model(datagen_ckpt_dir)
            new_step = model.sess.run(model.global_step)
        except:
            time.sleep(30)
            continue
        # If it was already observed, sleep
        if new_step == last_step:
            time.sleep(30)
            continue

        last_step = new_step
        last_ckpt = tf.train.latest_checkpoint(datagen_ckpt_dir)

        labeled_all_hypotheses = []

        # Precompute batches
        batches = []
        for batch_start in range(start_time, finish_time + 1, batch_len):
            batch_end = batch_start + batch_len
            batches.append((batch_start, batch_end))

        np.random.seed(new_step)
        order = np.arange(len(batches))
        batches = [batches[idx] for idx in order]

        for bid in tqdm(range(len(batches))):

            # Run the model on every batch
            # If the new model checkpoint is available, restart the process

            batch_start, batch_end = batches[bid]
            newest_ckpt = tf.train.latest_checkpoint(datagen_ckpt_dir)
            if newest_ckpt != last_ckpt:
                print("Model was updated, restarting")
                break

            #LabelStorage.instance.storage = {}
            batch_detections = []
            for when in range(batch_start, batch_end):
                if when in det_by_time.keys():
                    batch_detections += det_by_time[when]

            if len(batch_detections) == 0:
                continue

            # Good_hypotheses are the ones selected by MHT
            # All_hypotheses are all hypotheses observed by the MHT
            # We use all hypotheses for training.

            good_hypotheses, all_hypotheses =\
                nms_action.do(detections=batch_detections,
                              model=model,
                              previous_batch=[],
                              scene_start=batch_start,
                              scene_end=batch_end)

            out = []
            for h in all_hypotheses:
                if hash(h) not in all_hypotheses_ever:
                    all_hypotheses_ever.add(hash(h))
                    out.append(h)
            all_hypotheses = out

            # Compute the labels corresponging to all hypotheses

            lhypos = LabelStorage.instance.label_hypotheses(all_hypotheses,
                                                            mode="two")
            LabelStorage.instance.get_hypo_features(lhypos)
            labeled_all_hypotheses += lhypos

            if batch_end >= finish_time:
                break

        # Save all of the observed hypotheses with the corresponding labels
        # for the master to consume

        print("Step %d: + %d hypotheses = %d total" % (
            new_step, len(labeled_all_hypotheses), len(all_hypotheses_ever)))

        path = os.path.join(data_dir, "%d_%s.pickle" % (new_step, os.getpid()))
        pickle.dump(labeled_all_hypotheses, open(path, "wb"))
        with open(path + "_flag", "w") as f:
            f.write("Done creating labeled hypotheses")

# Evaluation process - MASTER
if args.mode.startswith('continuous_eval_joiner'):

    writer = tf.summary.FileWriter(summ_dir)
    best_metric = -100
    while True:
        # Find all the statistics generated by the slaves
        dir = str.replace(summ_dir, "joiner", "runner")

        fls = glob.glob("%s/*.txt" % dir)
        fls.sort(key=os.path.getmtime)

        if len(fls) == 0:
            time.sleep(10)
            continue

        # If there are as many summaries for the oldest checkpoint
        # as there are runners, gather all the statistics, average it,
        # and save to tensorboard
        min_ckpt = fls[0].split("/")[-1].split("_")[0]
        ctr = sum([file.split("/")[-1].split("_")[0] == min_ckpt
                   for file in fls])
        print("Found min_ckpt %s cnt %d" % (min_ckpt, ctr))
        if ctr == args.eval_runner_cnt:
            data = {}
            sum_wght = 0
            for file in fls:
                if not file.split("/")[-1].split("_")[0] == min_ckpt:
                    continue
                with open(file, "r") as f:
                    lines = [line.rstrip() for line in f.readlines()]
                os.remove(file)

                keys = [line.split(" ")[0] for line in lines]
                vals = [float(line.split(" ")[1]) for line in lines]
                for k, v in zip(keys, vals):
                    if k == "GTLEN":
                        sum_wght += v
                        cur_wght = v
                        continue
                    if k not in data.keys():
                        data[k] = 0
                    data[k] += v * cur_wght

            summary = tf.Summary()
            for k, v in data.items():
                summary.value.add(tag=k, simple_value=v / sum_wght)
            writer.add_summary(summary, min_ckpt)
            writer.flush()

            # If the value for a given checkpoint is the best,
            # save the model in the best_model folder
            if data[TestConfig["target_metric"]] > best_metric:
                best_metric = data[TestConfig["target_metric"]]
                for file in glob.glob(r'%s/model-%s.*' % (ckpt_dir, min_ckpt)):
                    shutil.copy(file, best_model_dir)
                with open(os.path.join(best_model_dir, "checkpoint"), "w") as f:
                    f.write("model_checkpoint_path: \"model-%s\"\n" % min_ckpt)
                    f.write("all_model_checkpoint_paths: \"model-%s\"\n" % min_ckpt)
        else:
            # If something happened and there are no results for the oldest checkpoint
            # but it is already very stale, just ignore it and proceed to the newer ones.
            if len(fls) > args.eval_runner_cnt * 3:
                print("But too long list of files, removing tail")
                for file in fls:
                    if not file.split("/")[-1].split("_")[0] == min_ckpt:
                        continue
                    os.remove(file)

        time.sleep(10)
        continue
# Evaluation process - SLAVE
if args.mode.startswith('continuous_eval_runner'):
    final_json = tostring(TestConfig["final_solution_config"])

    old_step = -1
    data_provider = make_data_provider(TestConfig["input_data_provider_config"])

    while True:
        # Read the latest model
        LabelStorage.instance = None
        LabelStorage(label_config=TestConfig["label_config"],
                     data_provider=data_provider)
        tf.reset_default_graph()
        model = make_model(TestConfig["model_config"], "infer",
                           tf.Session(), ckpt_dir)
        try:
            if args.mode.endswith("best"):
                model.load_model(best_model_dir)
                print("Loading from best")
            else:
                model.load_model(ckpt_dir)
            new_step = model.sess.run(model.global_step)
        except:
            logger.info("Waiting for model checkpoint")
            time.sleep(10)
            continue

        if new_step == old_step:
            time.sleep(10)
            continue
        else:
            old_step = new_step
            logger.info("Evaluating @ step %d", new_step)

        # Run inference and save the tracks

        final_tracks, all_hypos, selected_hypos = \
            run_once(model,
                     TestConfig["label_config"],
                     TestConfig["input_data_provider_config"],
                     TestConfig["nms_config"],
                     TestConfig["final_solution_config"])


        # We are only interested in tracks longer than the length of the batch.
        final_tracks = [t for t in final_tracks
                        if len(t) >
                        TestConfig["input_data_provider_config"]["scene_size"]]

        # Evaluate the quality of tracking

        mota, idf = data_provider.evaluate(tracks=final_tracks)

        # To compute the summaries, also take a part of the observed hypotheses
        # and compute summaries on it

        if len(all_hypos) > 100000:
            all_hypos = all_hypos[:100000]
        labeled_hypos = LabelStorage.instance.label_hypotheses(all_hypos)

        summaries, _ = model.train_epoch(labeled_hypos, 0, do_train=False)

        # Save the summaries and the tracking results for evaluation master to gather

        path = os.path.join(summ_dir, "%d_%d.txt" % (new_step, os.getpid()))

        with open(path, "w") as f:
            f.write("GTLEN %d\n" % sum([len(x)
                                        for x
                                        in data_provider.ground_truth()]))
            f.write("MOTA %0.3f\n" % mota)
            f.write("IDF %0.3f\n" % idf)
            for k, v in summaries.items():
                f.write("%s %0.3f\n" % (k, v))

        if args.mode.endswith("best"):
            exit(0)

# Inference
if args.mode.startswith('infer'):

    data_provider = make_data_provider(TestConfig["input_data_provider_config"])
    min_det = min([det.time for det in data_provider.detections()])
    LabelStorage.instance = None
    LabelStorage(label_config=TestConfig["label_config"],
                 data_provider=data_provider)
    tf.reset_default_graph()
    model = make_model(TestConfig["model_config"], "infer",
                       tf.Session(), ckpt_dir)
    model.load_model(best_model_dir)

    final_tracks, all_hypos, selected_hypos = \
        run_once(model,
                 TestConfig["label_config"],
                 TestConfig["input_data_provider_config"],
                 TestConfig["nms_config"],
                 TestConfig["final_solution_config"])

    # We only consider in the final solution tracks that are longer than the length of the batch.
    cur_final_tracks = [t for t in final_tracks
                        if len(t) >
                        TestConfig["input_data_provider_config"]["scene_size"]]
    # Final results are saved in runs/$experiment_name/summaries/infer/tracks_$idx.txt
    data_provider.save_tracks(cur_final_tracks, os.path.join(summ_dir,
                                                             "tracks_%s.txt" % TestConfig["input_data_provider_config"]["frames"]))
