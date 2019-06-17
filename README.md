## Eliminating Exposure Bias and Metric Mismatch in Multiple Object Tracking.
Created by Andrii Maksai at CVLAB, EPFL.
This is an approach for training sequence models for multiple object tracking.

### License

This work is released under the MIT License (refer to the LICENSE file for details).

### Requirements

1. [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/details.html) dataset. Should be placed in `DukeMTMC/` folder.
2. [Motchallenge devkit](https://motchallenge.net/devkit) for computation of IDF metric. Should be placed in `external/motchallenge-devkit` folder, together with files currently present there.
3. [Open-reid](https://github.com/Cysu/open-reid) to use the approach with appearance features. Should be placed in `external/open-reid` folder, together with files currently present there.


### Workflow

0. (Optional)

	-  Train a ReID model in DukeMTMC dataset by running `external/open-reid/train.sh`.

  -  Start process that will answer to requests for computing the appearance model by running `external/open-reid/run.sh`.

  -  Modify `run.sh` according to your needs - see next section.

  -  Start tensorboard on `runs/<experiment_name>` folder to observe statistics related to the experiment. Experiment name can be set in `run.sh`.

1. Start the dataset generation procedure as `run.sh gen_dataset <cam_id>` for cameras with numbers ranging 1 to 8.
2. Start the training procedure on the generated dataset as `run.sh train <cam_id>`.
3. Start the evaluation procedure that will pick best model from the checkpoints generated during training by running `run.sh eval <cam_id>`. This could be done in parallel with training.
4. Start the inference procedure by running `run.sh infer <cam_id>`. Output will be generated in `runs/<experiment_name>/summaries/infer/tracks_*` file in the DukeMTMC benchmark-comparable format.

### Important values that could be modified in `run.sh`

1. **dp\_freq**, **dp\_size**, **dt\_size** (l.9-11) define frequency of frames sampling (0.33 refers to 3 per second), size of the batch for training, and maximum number of missed detections between two detections (to limit the number of pairs of detections that could possibly belong to the same trajectory).

2. **gendata\_step** (l.89). During dataset generation multiple runners in parallel run the latest verion of the model on the parts of the dataset, while one trainer gets all of the combined data. This value describes how many frames are assigned to one runner and affects number of runners and training time.

3. **label\_config.features** (l.115) List of features to be used for the model. When `appr` feature is provided, open-reid is required.

4. **model\_config** (l.150) - parameters of the model.

5. **experiment\_name** (l.187) - name of the experiment. All data related to the run will be located in `runs/<experiment_name>`.

6. **nms\_config.nms\_option** - how to select which hypotheses to keep in multiple hypotheses tracking. `start` corresponds to having at most one hypothesis of length X starting at each detection, and `start-0.3-ignore` additionally filters all hypotheses with IDF < 0.3 (speeds the inference, possibly why reducing accuracy, see paper appendix).

7. **final\_solution\_config** (l.178) - how to select the final set of hypotheses. score_cutoff corresponds to minimum value of IDF to be considered for final solution, and the bounding box overlap of any two solutions should be below iou cutoff.

8. l.266 - size of the batch to use for inference. As mentioned in the paper, it was found beneficial to train with batch of size 6 and infer with size 12.


### Citation \& Contact

If you use the code or compare to the results obtained with it on MOT15, MOT17,
or DukeMTMC dataset (available on [MOTChallenge website](https://motchallenge.net/)), please consider
citing our [paper](https://arxiv.org/abs/1811.10984).

Please contact andrii dot maksai at epfl dot ch for any related queries.
