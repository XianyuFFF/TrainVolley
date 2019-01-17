import cv2
import math
import numpy as np
# from moments import describe_shape, template_feature_match
# import libbgs
import MASK_RCNN.mrcnn.model as modellib
import os
from MASK_RCNN.coco import coco
from MASK_RCNN.volleyball import VolleyballConfig
from Moments import Moments
from BgsModel import BgsModel
from Parabola import Candidate, TrajectoryFilter, TraGenerator


def load_mask_rnn():
    MODEL_DIR = os.path.join("MASK_RCNN", "logs")
    VOLLEYBALL_MODEL_PATH = os.path.join("MASK_RCNN", "mask_rcnn_volleyball.h5")

    class InferenceConfig(VolleyballConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(VOLLEYBALL_MODEL_PATH, by_name=True)
    return model


if __name__ == '__main__':

    Asset_path = "asset"

    learning_stage = 1
    duration = 400

    mask_network = load_mask_rnn()

    video_name = 'cam1.mp4'
    res_video_name = "res_" + video_name
    cap = cv2.VideoCapture(video_name)

    num = 0

    circle_file_dir = os.path.join(Asset_path, "circle.jpg")
    ball_moments = Moments(circle_file_dir)
    bgs = BgsModel()

    frame_candidates = []
    while True:
        ret, frame = cap.read()
        if not ret or num > learning_stage + duration:
            break

        if num == learning_stage:
            print("learning stage is gone")

        frame_ = frame[..., ::-1]
        r = mask_network.detect([frame_], verbose=0)[0]

        rois = r['rois']
        scores = r['scores']

        candidates = []

        for roi, score in zip(rois, scores):
            y1, x1, y2, x2 = roi
            w, h = x2 - x1, y2 - y1
            candidate = Candidate([x1+w/2, y1+h/2], [w, h], score)
            candidates.append(candidate)

        if num >= learning_stage:
            tmp_image_dir = os.path.join('asset/tmp',"{}.png".format(num))
            cv2.imwrite(tmp_image_dir, bgs(frame))
            frame_different_region = ball_moments.find_loc(cv2.imread(tmp_image_dir))
            frame_different_center = frame_different_region[:2]
            frame_different_size = frame_different_region[2:]

            candidates.append(Candidate(frame_different_center, frame_different_size, 0.7))

        frame_candidates.append(candidates)
        num += 1

    print("done")
    print(len(frame_candidates))

    trajectory_filter = TrajectoryFilter(D_frm=10, frame_candidates=frame_candidates)

    not_conflict_trajectorys = trajectory_filter.filter_conflict_trajectorys(trajectory_filter.potential_trajectorys())
    print(len(not_conflict_trajectorys))

    trajectory_generator = TraGenerator(0, num, not_conflict_trajectorys)

    trajectory_generator.drop_useless_trajectory()
    trajectory_generator.head_tail_extend()
    trajectory_generator.interpolate()

    # trajectory_generator.save_result_in_json("result.json")
    trajectory_generator.show_result(video_name,save_result=True, res_video_name=res_video_name)



