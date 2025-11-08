import cv2
import numpy as np
import threading
import cv2.aruco as aruco
from pioneer_sdk import Pioneer, Camera

import core
from core import VisionCfg


class VisionWorker(threading.Thread):
    """Захват кадра, поиск зелёных ворот и ArUco."""

    def __init__(self, pioneer: Pioneer, cam: Camera | None, vcfg: VisionCfg):
        super().__init__(daemon=True)
        self._pioneer = pioneer
        self._cam = cam
        self._vcfg = vcfg
        self._stop = threading.Event()

        # ArUco
        ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        ar_params = aruco.DetectorParameters()
        self._aruco = aruco.ArucoDetector(ar_dict, ar_params)

        # Общие данные, защищённые локом
        self._data_lock = threading.Lock()
        self._gate = None           # tuple(cx, cy, w, h) | None
        self._aruco_centers = {}    # {id: (cx, cy)}
        self._frame_shape = None    # (h, w)

    # ----- интерфейс чтения из других потоков -----

    def snapshot(self):
        with self._data_lock:
            return (self._gate, dict(self._aruco_centers), self._frame_shape)

    # ----- внутреннее -----

    def _detect_green_gate(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb, self._vcfg.green_lo, self._vcfg.green_hi)
        mask = cv2.medianBlur(mask, 7)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, mask

        biggest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self._vcfg.min_area:
            return None, mask

        x, y, w, h = cv2.boundingRect(biggest)
        cx, cy = x + w // 2, y + h // 2

        if self._vcfg.show_windows and core.DEBUG:
            cv2.drawContours(frame_bgr, [biggest], -1, (0, 255, 0), 2)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

        return (cx, cy, w, h), mask

    def _decode_frame(self) -> np.ndarray | None:
        if self._cam is not None:
            return self._cam.get_cv_frame()

        # fallback через Pioneer.get_frame()
        blob = self._pioneer.get_frame()
        if not blob:
            return None
        arr = np.frombuffer(blob, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def run(self):
        while not self._stop.is_set():
            frame = self._decode_frame()
            if frame is None:
                continue

            gate_info, mask = self._detect_green_gate(frame)

            # ArUco
            ids_centers = {}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self._aruco.detectMarkers(gray)
            if ids is not None:
                for i, c in zip(ids.flatten(), corners):
                    cx = int(c[0][:, 0].mean())
                    cy = int(c[0][:, 1].mean())
                    ids_centers[int(i)] = (cx, cy)
                if self._vcfg.show_windows and core.DEBUG:
                    aruco.drawDetectedMarkers(frame, corners, ids)

            with self._data_lock:
                self._gate = gate_info
                self._aruco_centers = ids_centers
                self._frame_shape = frame.shape[:2]

            if self._vcfg.show_windows and core.DEBUG:
                cv2.imshow(self._vcfg.win_frame_name, frame)
                cv2.imshow(self._vcfg.win_mask_name, mask)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    self.stop()

        # безопасно даже если окон не было
        cv2.destroyAllWindows()

    def stop(self):
        self._stop.set()