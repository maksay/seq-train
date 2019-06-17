from models.base_model import BaseModel
import tensorflow as tf
import numpy as np
from label_storage import LabelStorage
from tqdm import tqdm
import time
from copy import deepcopy


def sequence_embedding(input_seq,
                       feat_dim,
                       embedding_size,
                       rnn_cell_size,
                       dropout,
                       training,
                       layers):


    batch_size = tf.shape(input_seq)[0]
    trainable = True

    with tf.variable_scope('batch_norm'):
        input_seq_2d = tf.reshape(input_seq,
                                  shape=(-1, feat_dim),
                                  name='input_2d')
        batch_norm = tf.layers.batch_normalization(input_seq_2d,
                                                   trainable=trainable,
                                                   training=training,
                                                   name='batch_norm',
                                                   axis=1)

    with tf.variable_scope('embedding'):
        embedding = tf.layers.dense(batch_norm,
                                    trainable=trainable,
                                    units=embedding_size,
                                    activation=tf.nn.relu,
                                    name='dense')

        embedding = tf.layers.batch_normalization(embedding,
                                                  trainable=trainable,
                                                  training=training,
                                                  name='batch_norm',
                                                  axis=1)

        embedding = tf.layers.dropout(embedding,
                                      rate=dropout,
                                      training=training,
                                      name='dropout')

        embedding = tf.reshape(embedding,
                               (batch_size, -1, embedding_size),
                               name='reshape')

    with tf.variable_scope('rnn'):

        cell_fw = tf.contrib.rnn.BasicLSTMCell(rnn_cell_size)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(rnn_cell_size)
        (output_fw, output_bw), _ = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                            embedding,
                                            dtype=tf.float32)
        rnn_output_3d = tf.stack([output_fw, output_bw], axis=2)

    if layers == 2:
        with tf.variable_scope('rnn2'):

            cell_fw2 = tf.contrib.rnn.BasicLSTMCell(rnn_cell_size)
            cell_bw2 = tf.contrib.rnn.BasicLSTMCell(rnn_cell_size)
            (output_fw2, output_bw2), _ = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2,
                                                tf.reshape(rnn_output_3d,
                                                           (batch_size,
                                                            -1,
                                                            2 * rnn_cell_size)),
                                                dtype=tf.float32)
            rnn_output_3d = tf.stack([output_fw2, output_bw2], axis=2)

    return rnn_output_3d

# Three heads acting on the rnn output of size batchxlengthxoutput_size
# They predict IoU, whether the Gt exists, and the shift to GT bounding box
def iou_prediction_head(rnn_output_3d, output_size):
    batch_size = tf.shape(rnn_output_3d)[0]
    rnn_output = tf.reshape(rnn_output_3d, (-1, output_size))
    output_ious = tf.layers.dense(rnn_output, units=1, name='dense')[:, 0]
    output_ious = tf.reshape(output_ious, (batch_size, -1))
    return output_ious

def label_prediction_head(rnn_output_3d, output_size):
    batch_size = tf.shape(rnn_output_3d)[0]
    rnn_output = tf.reshape(rnn_output_3d, (-1, output_size))
    output_labels = tf.layers.dense(rnn_output,
                                    activation=tf.nn.sigmoid,
                                    units=1,
                                    name='dense')[:, 0]
    output_labels = tf.reshape(output_labels, (batch_size, -1))
    return output_labels

def bbox_shift_head(rnn_output_3d, output_size):
    batch_size = tf.shape(rnn_output_3d)[0]
    rnn_output = tf.reshape(rnn_output_3d, (-1, output_size))
    output_bbox_shifts = tf.layers.dense(rnn_output,
                                         units=4, name='dense')
    output_bbox_shifts = tf.reshape(output_bbox_shifts, (batch_size, -1, 4))
    return output_bbox_shifts

# IoU between two bounding boxes computation in TF
# such that IoU with GT could be optimized.
def bbox_iou(output_bboxes, label_bboxes):
    bbox_lft = tf.maximum(output_bboxes[:, 0], label_bboxes[:, 0])
    bbox_rgt = tf.minimum(output_bboxes[:, 0] + output_bboxes[:, 2],
                          label_bboxes[:, 0] + label_bboxes[:, 2])
    bbox_up = tf.maximum(output_bboxes[:, 1], label_bboxes[:, 1])
    bbox_dn = tf.minimum(output_bboxes[:, 1] + output_bboxes[:, 3],
                         label_bboxes[:, 1] + label_bboxes[:, 3])
    bbox_dx = tf.maximum(0., bbox_rgt - bbox_lft)
    bbox_dy = tf.maximum(0., bbox_dn - bbox_up)
    bbox_inter = bbox_dx * bbox_dy
    bbox_area = tf.add(tf.maximum(0., output_bboxes[:, 2]) *
                       tf.maximum(0., output_bboxes[:, 3]),
                       tf.maximum(0., label_bboxes[:, 2]) *
                       tf.maximum(0., label_bboxes[:, 3]))
    bbox_iou = bbox_inter / (bbox_area - bbox_inter + 1e-6)
    return bbox_iou


class BidiRNNIoUPredictorModel(BaseModel):
    def __init__(self, config, mode, sess, ckpt_dir):
        super(BidiRNNIoUPredictorModel, self).__init__(config, mode,
                                                       sess, ckpt_dir)

        self.mode = mode
        self.training = True if mode == 'train' else False
        self.feat_dim = LabelStorage.instance.feature_dim()
        with tf.variable_scope('inputs'):

            self.input_seq = tf.placeholder(dtype=tf.float32,
                                            shape=[None, None, self.feat_dim],
                                            name='input_seq')
            self.input_bboxes = tf.placeholder(dtype=tf.float32,
                                               name='input_bboxes',
                                               shape=[None, None, 4])
            self.input_values = tf.placeholder(dtype=tf.float32,
                                               name='input_bboxes',
                                               shape=[None, None])

        with tf.variable_scope("labels"):
            self.label_bboxes = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None, 4],
                                               name='label_bboxes')

            self.label_values = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None],
                                               name='label_values')

        with tf.variable_scope("sequence_embedding"):
            self.rnn_output_3d = sequence_embedding(self.input_seq,
                                                    self.feat_dim,
                                                    self.embedding_size,
                                                    self.rnn_cell_size,
                                                    self.dropout,
                                                    self.training,
                                                    self.layers)

        self.outputs = {}
        self.losses = {}
        self.summaries = {}

        if self.predict_ious == 1:

            with tf.variable_scope('predict_ious'):
                with tf.variable_scope('prediction'):
                    self.outputs["ious"] = \
                        iou_prediction_head(self.rnn_output_3d,
                                            2 * self.rnn_cell_size)
                    label_ious_3d = bbox_iou(
                        tf.reshape(self.input_bboxes, (-1, 4)),
                        tf.reshape(self.label_bboxes, (-1, 4)))
                    label_ious = \
                        tf.reshape(label_ious_3d,
                                   (tf.shape(self.label_bboxes)[0], -1))

                with tf.variable_scope("loss"):
                    error_matrix = tf.square(self.outputs["ious"] - label_ious)
                    self.losses["iou_vector"] = \
                        tf.reduce_sum(error_matrix * self.input_values,
                                      axis=1) / \
                        tf.reduce_sum(self.input_values, axis=1)
                    self.losses["iou"] = \
                        tf.reduce_sum(error_matrix * self.input_values) /\
                        tf.reduce_sum(self.input_values)
                    self.summaries["IoU_loss"] = self.losses["iou"]

        if self.predict_labels == 1:
            with tf.variable_scope('predict_labels'):
                with tf.variable_scope('prediction'):
                    self.outputs["labels"] = \
                        label_prediction_head(self.rnn_output_3d,
                                              2 * self.rnn_cell_size)
                with tf.variable_scope("loss"):
                    error_matrix = tf.square(self.outputs["labels"] -
                                             self.label_values)
                    self.losses["label_vector"] = tf.reduce_mean(error_matrix,
                                                                 axis=1)
                    self.losses["label"] = tf.reduce_mean(error_matrix)
                    self.summaries["label_loss"] = self.losses["label"]

        with tf.variable_scope('predict_bboxes'):
            self.outputs["bboxes"] = self.input_bboxes
            if self.predict_bboxes == 1:
                with tf.variable_scope("prediction"):
                    self.outputs["bboxes"] += \
                        bbox_shift_head(self.rnn_output_3d,
                                        2 * self.rnn_cell_size)
            with tf.variable_scope("loss"):
                self.label_ious_3d = bbox_iou(
                    tf.reshape(self.outputs["bboxes"], (-1, 4)),
                    tf.reshape(self.label_bboxes, (-1, 4)))

                self.label_ious = \
                    tf.reshape(self.label_ious_3d,
                               (tf.shape(self.label_bboxes)[0], -1))
                error_matrix = tf.square(1 - self.label_ious)

                self.losses["bbox"] = \
                    tf.reduce_sum(error_matrix * self.input_values) /\
                    tf.reduce_sum(self.input_values)
                self.losses["bbox_vector"] = \
                    tf.reduce_sum(error_matrix * self.input_values, axis=1) / \
                    tf.reduce_sum(self.input_values, axis=1)
                self.summaries["bbox_loss"] = self.losses["bbox"]

        with tf.variable_scope("optimizer"):
            self.loss = 0
            self.loss_vector = 0
            for k, v in self.losses.items():
                if not k.endswith("vector"):
                    self.loss += v
                else:
                    self.loss_vector += v
            self.summaries["loss"] = self.loss

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=
                                                    self.global_step)

    def train_epoch(self, labeled_hypotheses, epoch, do_train=True,
                    do_save=True):
        # Compute all features
        LabelStorage.instance.get_hypo_features(labeled_hypotheses)
        order = np.arange(len(labeled_hypotheses))
        np.random.seed(epoch + 1)
        np.random.shuffle(order)

        summaries = {}
        for k in self.summaries.keys():
            summaries[k] = []

        losses = []

        t_start = time.time()
        batch_range = range(0, len(labeled_hypotheses), self.batch_size)
        for batch_start in tqdm(batch_range) if do_train else batch_range:
            batch_end = min(batch_start + self.batch_size,
                            len(labeled_hypotheses))

            # Arrange data in batches

            cur_input_seq = np.stack([
                labeled_hypotheses[idx].features
                for idx in order[batch_start:batch_end]])
            cur_input_bboxes = np.stack([
                labeled_hypotheses[idx].input_bboxes
                for idx in order[batch_start:batch_end]])
            cur_input_values = np.stack([
                labeled_hypotheses[idx].input_values
                for idx in order[batch_start:batch_end]])
            cur_label_values = np.stack([
                labeled_hypotheses[idx].labels
                for idx in order[batch_start:batch_end]])
            cur_label_bboxes = np.stack([
                labeled_hypotheses[idx].bboxes
                for idx in order[batch_start:batch_end]])

            feed_dict = {
                self.input_seq: cur_input_seq,
                self.input_bboxes: cur_input_bboxes,
                self.input_values: cur_input_values,
                self.label_values: cur_label_values,
                self.label_bboxes: cur_label_bboxes
            }
            if do_train:
                self.sess.run(self.train_op, feed_dict)
                if do_save:
                    if time.time() - t_start > self._save_every_secs:
                        t_start = time.time()
                        self.save_model()
            cur_summaries, lv = self.sess.run([self.summaries,
                                               self.loss_vector],
                                              feed_dict)
            # compute average summaries in batches
            for k, v in cur_summaries.items():
                summaries[k].append(v)
            losses.append(lv)

        losses = np.concatenate(losses)

        for k in summaries.keys():
            summaries[k] = np.mean(np.asarray(summaries[k]))
        if do_train and do_save:
            self.save_model()

        return summaries, losses

    def _score(self, hypotheses):
        # Given a set of hypotheses, feed them to a network in batches.
        self.logger.info("Scoring %d hypotheses", len(hypotheses))
        LabelStorage.instance.get_hypo_features(hypotheses)
        for batch_start in range(0, len(hypotheses), self._eval_batch_size):
            batch_end = min(batch_start + self._eval_batch_size,
                            len(hypotheses))
            cur_input_seq = np.stack([
                hypotheses[idx].features
                for idx in range(batch_start, batch_end)])
            cur_input_bboxes = np.stack([
                hypotheses[idx].input_bboxes
                for idx in range(batch_start, batch_end)])
            cur_input_values = np.stack([
                hypotheses[idx].input_values
                for idx in range(batch_start, batch_end)])
            feed_dict = {
                self.input_seq: cur_input_seq,
                self.input_bboxes: cur_input_bboxes,
                self.input_values: cur_input_values
            }
            outputs = self.sess.run(self.outputs, feed_dict)
            for idx in range(batch_end - batch_start):
                cur_outputs = {}
                for k, v in outputs.items():
                    cur_outputs[k] = \
                        outputs[k][idx]
                hypotheses[batch_start + idx].outputs = cur_outputs
                hypotheses[batch_start + idx].score = \
                    LabelStorage.instance.score(hypotheses[batch_start + idx],
                                                cur_outputs)

    def score(self, hypotheses):
        # Scoring hypotheses with the model.
        self._score(hypotheses)
        if self.mode == "infer":
            # During inference we take input hypothesis.
            # Then we change it accoring to the shifts predicted by the network.
            # Then we score it once again.
            tmp = [deepcopy(h) for h in hypotheses]
            for h, h2 in zip(hypotheses, tmp):
                for did in range(len(h.tracklet)):
                    if h.tracklet[did] is not None:
                        h.outputs["old_ious"] = h.outputs["ious"]
                        h2.tracklet[did].bbox = \
                            h.outputs["bboxes"][did].reshape((1, 4))
                        h2.tracklet[did].features = None
                delattr(h2, "features")
            self._score(tmp)
            for h, h2 in zip(hypotheses, tmp):
                h.outputs["ious"] = h2.outputs["ious"]
