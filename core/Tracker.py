# core/Tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou_batch(bb_test, bb_gt):
    """Compute IoU matrix between detections and tracker boxes."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_test = np.asarray(bb_test, dtype=np.float32)
    bb_gt = np.asarray(bb_gt, dtype=np.float32)

    if bb_test.shape[1] > 4:
        bb_test = bb_test[:, :4]
    if bb_gt.shape[1] > 4:
        bb_gt = bb_gt[:, :4]

    bb_gt_exp = np.expand_dims(bb_gt, 0)
    bb_test_exp = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test_exp[..., 0], bb_gt_exp[..., 0])
    yy1 = np.maximum(bb_test_exp[..., 1], bb_gt_exp[..., 1])
    xx2 = np.minimum(bb_test_exp[..., 2], bb_gt_exp[..., 2])
    yy2 = np.minimum(bb_test_exp[..., 3], bb_gt_exp[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area_test = np.maximum(0., (bb_test_exp[..., 2] - bb_test_exp[..., 0])) * np.maximum(0., (bb_test_exp[..., 3] - bb_test_exp[..., 1]))
    area_gt = np.maximum(0., (bb_gt_exp[..., 2] - bb_gt_exp[..., 0])) * np.maximum(0., (bb_gt_exp[..., 3] - bb_gt_exp[..., 1]))
    denom = area_test + area_gt - wh
    return np.where(denom > 0, wh / denom, 0.0)


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox, score=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_score = None if score is None else float(score)

        self.validated_id = None
        self.flow_status = 0
        self.flow_roi = None

    def update(self, bbox, score=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        if score is not None:
            self.last_score = float(score)

    def predict(self):
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        bbox = np.asarray(bbox, dtype=np.float32)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 0.0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(max(float(x[2] * x[3]), 0.0))
        h = float(x[2]) / w if w > 0 else 0.0
        box = np.array([float(x[0] - w / 2.), float(x[1] - h / 2.),
                        float(x[0] + w / 2.), float(x[1] + h / 2.)]).reshape((1, 4))
        if score is None:
            return box
        return np.concatenate([box, np.array([[score]], dtype=np.float32)], axis=1)


class HybridOCSORT:
    def __init__(self,
                 is_lab_scene=False,
                 total_target_count=None,
                 det_thresh=0.4,
                 iou_threshold=0.3,
                 max_age=30,
                 min_hits=3,
                 validator=None,
                 enable_validator=False,
                 occlusion_penalty=0.5):

        if is_lab_scene and (total_target_count is None or int(total_target_count) <= 0):
            raise ValueError("total_target_count must be > 0 when is_lab_scene=True")

        self.is_lab_scene = bool(is_lab_scene)
        self.total_target_count = None if total_target_count is None else int(total_target_count)
        self.det_thresh = float(det_thresh)
        self.iou_threshold = float(iou_threshold)
        self.max_age = 99999 if self.is_lab_scene else int(max_age)
        self.min_hits = int(min_hits)
        self.occlusion_penalty = float(occlusion_penalty)

        self.trackers = []
        self.frame_count = 0

        self.validator = validator
        self.enable_validator = enable_validator
        self.id_mapping = {}

    def update(self, detections, flow_results, image=None):
        """
        Args:
            detections: ndarray of shape (N, 5), [x1, y1, x2, y2, score].
            flow_results: list of dicts with internal tracker id and flow status.
            image: current frame, used only by optional validator.

        Returns:
            list[dict]: each record contains track_id, id, validated_id, box,
            score, is_occluded, and flow_status. This avoids mixing string ids
            with numeric boxes in a single numpy array.
        """
        self.frame_count += 1

        detections = np.asarray(detections, dtype=np.float32) if len(detections) > 0 else np.empty((0, 5), dtype=np.float32)
        if len(detections) > 0:
            detections = detections[detections[:, 4] >= self.det_thresh]

        flow_map = {item['id']: item for item in flow_results}

        to_del = []
        active_tracker_indices = []
        trks_pred = []

        for idx, trk in enumerate(self.trackers):
            pos = trk.predict()[0]

            if trk.id in flow_map:
                f_data = flow_map[trk.id]
                trk.flow_status = int(f_data.get('is_occluded', 0))
                if trk.flow_status == 2 and f_data.get('mask') is not None:
                    y_indices, x_indices = np.nonzero(f_data['mask'])
                    if len(y_indices) > 0:
                        trk.flow_roi = [float(np.min(x_indices)), float(np.min(y_indices)),
                                        float(np.max(x_indices) + 1), float(np.max(y_indices) + 1)]
                    else:
                        trk.flow_roi = pos.astype(float).tolist()
                else:
                    trk.flow_roi = None
            else:
                trk.flow_status = 0
                trk.flow_roi = None

            if np.any(np.isnan(pos)):
                to_del.append(idx)
            else:
                active_tracker_indices.append(idx)
                trks_pred.append(pos)

        for idx in reversed(to_del):
            self.trackers.pop(idx)

        if trks_pred:
            current_trks_state = np.asarray([self.trackers[i].get_state()[0] for i in active_tracker_indices], dtype=np.float32)
        else:
            current_trks_state = np.empty((0, 4), dtype=np.float32)

        matched, unmatched_dets, unmatched_trks = self._associate(detections, current_trks_state, active_tracker_indices)

        for det_idx, trk_idx in matched:
            self.trackers[int(trk_idx)].update(detections[int(det_idx), :4], detections[int(det_idx), 4])

        if self.is_lab_scene:
            can_create = self.total_target_count is not None and len(self.trackers) < self.total_target_count
            if can_create:
                for det_idx in unmatched_dets:
                    if len(self.trackers) >= self.total_target_count:
                        break
                    self.trackers.append(KalmanBoxTracker(detections[int(det_idx), :4], detections[int(det_idx), 4]))
        else:
            for det_idx in unmatched_dets:
                self.trackers.append(KalmanBoxTracker(detections[int(det_idx), :4], detections[int(det_idx), 4]))

        if self.enable_validator and self.validator is not None and image is not None:
            self._run_validator(image)

        results = []
        for idx, trk in enumerate(self.trackers):
            state = trk.get_state()[0]
            is_occluded = 1 if trk.time_since_update > 0 else 0
            should_output = (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)
            if should_output:
                validated_id = self.id_mapping.get(trk.id, None)
                public_id = validated_id if validated_id is not None else trk.id
                results.append({
                    "track_id": int(trk.id),
                    "id": public_id,
                    "validated_id": validated_id,
                    "box": [float(v) for v in state.tolist()],
                    "score": None if trk.last_score is None else float(trk.last_score),
                    "is_occluded": int(is_occluded),
                    "flow_status": int(trk.flow_status)
                })

        for idx in reversed(range(len(self.trackers))):
            if self.trackers[idx].time_since_update > self.max_age:
                self.trackers.pop(idx)

        return results

    def _run_validator(self, image):
        img_h, img_w = image.shape[:2]

        for trk in self.trackers:
            if trk.validated_id is None and trk.time_since_update == 0:
                bbox = trk.get_state()[0]
                x1, y1, x2, y2 = map(int, bbox)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    try:
                        result_id = self.validator(crop)
                        if result_id is not None:
                            print(f"[Validator] Track {trk.id} identified as {result_id}")
                            trk.validated_id = result_id
                            self.id_mapping[trk.id] = result_id
                    except Exception as e:
                        print(f"Validator error for track {trk.id}: {e}")

    def post_process_trajectories(self, trajectory_data):
        print("[Tracker] Running reverse ID update...")

        if not self.id_mapping:
            print("[Tracker] No ID updates found. Returning original data.")
            return trajectory_data

        corrected_count = 0
        for frame_data in trajectory_data:
            for obj in frame_data.get('objects', []):
                internal_id = obj.get('track_id', obj.get('id'))
                if internal_id in self.id_mapping:
                    new_id = self.id_mapping[internal_id]
                    if obj.get('id') != new_id:
                        obj['id'] = new_id
                        obj['validated_id'] = new_id
                        corrected_count += 1

        print(f"[Tracker] Corrected {corrected_count} object instances in history.")
        return trajectory_data

    def _associate(self, detections, trackers, active_tracker_indices):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.asarray(active_tracker_indices, dtype=int)

        iou_matrix = iou_batch(detections[:, :4], trackers)

        for col_idx, trk_idx_real in enumerate(active_tracker_indices):
            trk = self.trackers[trk_idx_real]
            if trk.flow_status == 1:
                # Soft occlusion handling: reduce priority but do not forbid
                # rematching, so a target can reconnect after short occlusion.
                iou_matrix[:, col_idx] *= self.occlusion_penalty
            elif trk.flow_status == 2 and trk.flow_roi is not None:
                roi = trk.flow_roi
                det_centers_x = (detections[:, 0] + detections[:, 2]) / 2
                det_centers_y = (detections[:, 1] + detections[:, 3]) / 2
                in_roi = (det_centers_x >= roi[0]) & (det_centers_x <= roi[2]) & \
                         (det_centers_y >= roi[1]) & (det_centers_y <= roi[3])
                iou_matrix[~in_roi, col_idx] = -1.0

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)), dtype=int)

        unmatched_dets = []
        for d in range(len(detections)):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_dets.append(d)

        unmatched_trks = []
        for local_idx, real_idx in enumerate(active_tracker_indices):
            if len(matched_indices) == 0 or local_idx not in matched_indices[:, 1]:
                unmatched_trks.append(real_idx)

        matches = []
        for det_idx, local_trk_idx in matched_indices:
            real_trk_idx = active_tracker_indices[local_trk_idx]
            if iou_matrix[det_idx, local_trk_idx] < self.iou_threshold:
                unmatched_dets.append(det_idx)
                unmatched_trks.append(real_trk_idx)
            else:
                matches.append([det_idx, real_trk_idx])

        return np.asarray(matches, dtype=int), np.asarray(unmatched_dets, dtype=int), np.asarray(unmatched_trks, dtype=int)
