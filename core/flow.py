# core/flow.py
import cv2
import numpy as np


class OpticalFlowTracker:
    def __init__(self, overlap_threshold=0.3):
        """
        Args:
            overlap_threshold (float): ratio threshold used to mark an object as
            ambiguous/overlapping (status 2).
        """
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
        self.overlap_threshold = overlap_threshold

    def _resize_frame(self, frame, target_size):
        if target_size is None:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR), target_w / w, target_h / h

    def _resize_mask(self, mask, target_size):
        if mask is None:
            return None
        if target_size is None:
            return (mask > 0).astype(np.uint8)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def predict_next_step(self, prev_frame, curr_frame, objects_data, process_size=None):
        """
        Args:
            prev_frame: previous frame image.
            curr_frame: current frame image.
            objects_data: list of dicts with {'id': int, 'mask': np.array,
                'is_occluded': 0/1}. The id should be the internal numeric
                tracker id, not a user-facing validated id.
            process_size: (w, h) optical-flow processing size, e.g. (480, 480).

        Returns:
            list[dict]: [{'id': int, 'mask': np.array | None,
            'is_occluded': 0/1/2}, ...]
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

        prev_small, _, _ = self._resize_frame(prev_gray, process_size)
        curr_small, _, _ = self._resize_frame(curr_gray, process_size)

        h_small, w_small = prev_small.shape
        orig_h, orig_w = prev_frame.shape[:2]

        flow = self.dis.calc(prev_small, curr_small, None)

        grid_x, grid_y = np.meshgrid(np.arange(w_small), np.arange(h_small))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)

        visible_objs = [obj for obj in objects_data if int(obj.get('is_occluded', 0)) == 0 and obj.get('mask') is not None]
        occluded_objs = [obj for obj in objects_data if int(obj.get('is_occluded', 0)) == 1 and obj.get('mask') is not None]

        predicted_cache = {}

        # A. Visible objects: warp a temporary label map. Do NOT use tracker ids
        # directly as label-map values because id=0 would collide with background.
        if visible_objs:
            label_map = np.zeros((h_small, w_small), dtype=np.int32)
            valid_ids = []
            id_to_label = {}
            label_to_id = {}

            for obj in visible_objs:
                track_id = obj['id']
                if track_id in id_to_label:
                    continue
                label_value = len(id_to_label) + 1  # background remains 0
                id_to_label[track_id] = label_value
                label_to_id[label_value] = track_id
                valid_ids.append(track_id)

                m_small = self._resize_mask(obj['mask'], (w_small, h_small))
                if m_small is not None:
                    label_map[m_small > 0] = label_value

            warped_label_map = cv2.remap(label_map.astype(np.float32), map_x, map_y,
                                         interpolation=cv2.INTER_NEAREST).astype(np.int32)

            for track_id in valid_ids:
                label_value = id_to_label[track_id]
                p_mask = np.where(warped_label_map == label_value, 1, 0).astype(np.uint8)
                predicted_cache[track_id] = p_mask

        # B. Previously occluded objects: independently warp the last available mask.
        for obj in occluded_objs:
            track_id = obj['id']
            m_small = self._resize_mask(obj['mask'], (w_small, h_small))
            if m_small is None:
                continue
            warped_m = cv2.remap(m_small.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST)
            p_mask = (warped_m > 0).astype(np.uint8)
            predicted_cache[track_id] = p_mask

        accumulation_map = np.zeros((h_small, w_small), dtype=np.uint16)
        for p_mask in predicted_cache.values():
            accumulation_map += p_mask.astype(np.uint16)

        final_results = []

        for obj in objects_data:
            track_id = obj['id']
            p_mask = predicted_cache.get(track_id, np.zeros((h_small, w_small), dtype=np.uint8))
            pixel_count = cv2.countNonZero(p_mask)

            if pixel_count == 0:
                final_results.append({
                    'id': track_id,
                    'mask': None,
                    'is_occluded': 1
                })
                continue

            overlap_area = np.logical_and(p_mask == 1, accumulation_map > 1)
            overlap_pixels = np.count_nonzero(overlap_area)
            overlap_ratio = overlap_pixels / max(pixel_count, 1)

            status = 2 if overlap_ratio > self.overlap_threshold else 0
            final_mask_small = (p_mask * 255).astype(np.uint8)

            if process_size is not None:
                final_mask = cv2.resize(final_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                final_mask = final_mask_small

            final_results.append({
                'id': track_id,
                'mask': final_mask,
                'is_occluded': status
            })

        return final_results
