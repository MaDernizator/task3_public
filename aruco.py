from pioneer_sdk import Camera, Pioneer

import cv2
import numpy as np
import time
import threading
import sys
import math
from collections import defaultdict, deque

# -------------------- НАСТРОЙКИ --------------------

# Выравнивание yaw к нормали ворот
K_YAW_PERP = 1.0          # усиление по «перпендикулярности»
ALIGN_USE_FROM = 0.28     # начиная с какого «размера ворот» включать выравнивание
ALIGN_FULL_FROM = 0.52    # с этого размера – только выравнивание (центрирование по ex почти не влияет)
ALIGN_TOL = 0.18          # допустимая несоосность (нормированная), при которой разрешаем «рывок»
NEAR_SLOW_VX = 0.25       # ограничиваем вперёд, если близко, но ещё «доворачиваемся»

SHOW_DEBUG = True                 # показать окно с оверлеем
CMD_HZ = 30.0                     # частота отправки set_manual_speed_* (Гц)
TAKEOFF_ALT = 2                   # целевая высота после взлёта (м)
BASE_VX = 0.45                    # базовая скорость вперёд (м/с)
BURST_VX = 1                      # рывок при пролёте (м/с)
PASS_SIZE_TH = 0.42               # порог "мы уже у ворот": относит. диагональ к ширине кадра
PASS_BURST_T = 1.5                # длительность рывка сквозь ворота (с)
NO_DET_HOLD_T = 0.7               # если цели не видно дольше этого — зависаем (с)

# ПИД-коэффициенты (простые P-регуляторы)
K_YAW = 1.0                       # рад/с на единицу норм. ошибки по X
K_VY = 0.35                       # м/с на единицу норм. ошибки по X (боковое смещение)
K_VZ = 0.6                        # м/с на единицу норм. ошибки по Y (вверх/вниз)
USE_STRAFE = True                 # кроме поворота, немного подруливать вбок

# Ограничения
MAX_VX = 1.2
MAX_VY = 0.8
MAX_VZ = 0.8
MAX_YAWR = 1.8

# Какие словари ArUco пробуем (берём тот, где больше детекций)
ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
]

# ---------------------------------------------------

class GateDetector:
    """Ищет группы одинаковых ID (четыре угла ворот) и возвращает ближайшие ворота."""
    def __init__(self):
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detectors = []
        for d in ARUCO_DICTS:
            self.detectors.append(cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(d),
                self.detector_params
            ))

    @staticmethod
    def _centroid(corners):
        # corners: (4,2) -> центр маркера
        return corners.mean(axis=0)

    def detect_gates(self, frame):
        h, w = frame.shape[:2]
        best = None
        best_cnt = -1
        # выбираем словарь, давший максимум меток
        for det in self.detectors:
            corners, ids, _ = det.detectMarkers(frame)
            cnt = 0 if ids is None else len(ids)
            if cnt > best_cnt:
                best_cnt = cnt
                best = (corners, ids)

        corners, ids = best
        if ids is None or len(ids) == 0:
            return []  # ничего не нашли

        # группируем по id
        groups = defaultdict(list)  # id -> список центров меток
        raw_groups = defaultdict(list)  # id -> список списков углов
        for cs, i in zip(corners, ids.flatten()):
            c = self._centroid(cs[0])
            groups[int(i)].append(c)
            raw_groups[int(i)].append(cs[0])

        gates = []
        for gid, pts in groups.items():
            if len(pts) < 2:
                continue
            pts = np.array(pts)  # центры меток
            center = pts.mean(axis=0)
            # "размер" ворот — максимум попарных расстояний между центрами углов (похоже на диагональ)
            diag = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d > diag:
                        diag = d
            size_norm = float(diag) / float(w + 1e-6)
            gates.append({
                "id": gid,
                "center": (float(center[0]), float(center[1])),
                "size_norm": size_norm,
                "pts": pts,
                "raw": raw_groups[gid],
            })

        # сортируем по убыванию видимого размера — ближайшие ворота первые
        gates.sort(key=lambda g: g["size_norm"], reverse=True)
        return gates


def _yaw_error_from_marker_areas(raw_quads):
    """
    Оцениваем «несоосность по yaw» как нормированную разницу средних площадей
    правых и левых маркеров: (A_right - A_left)/(A_right + A_left).
    Знак: положительный, если правая сторона визуально ближе.
    """
    if raw_quads is None or len(raw_quads) < 2:
        return 0.0, 0
    centers = [q.mean(axis=0) for q in raw_quads]
    xs = np.array([c[0] for c in centers])
    order = np.argsort(xs)

    n = len(raw_quads)
    k = max(1, n // 2)          # по половине на сторону (1–2 маркера)
    left_idx = order[:k]
    right_idx = order[-k:]

    def mean_area(idxs):
        if len(idxs) == 0:
            return 0.0
        return float(np.mean([abs(cv2.contourArea(raw_quads[i].astype(np.float32))) for i in idxs]))

    A_left = mean_area(left_idx)
    A_right = mean_area(right_idx)
    s = A_left + A_right
    if s < 1e-6:
        return 0.0, n
    e = (A_right - A_left) / (s + 1e-6)
    return e, n



class Controller:
    """Простой P-контроллер наведения на центр ворот + выравнивание yaw к нормали"""
    def __init__(self):
        self.last_det_t = 0.0
        self.burst_until = 0.0
        self.cx_ema = None
        self.cy_ema = None
        self.s_ema = None
        self.alpha = 0.25
        self.current_gate_id = None
        self.last_align_e = 0.0   # для отладки

    def _ema(self, val, prev):
        return val if prev is None else (1.0 - self.alpha) * prev + self.alpha * val

    def update(self, gate, t_now, w, h):
        """Возвращает (vx, vy, vz, yaw_rate)"""
        if t_now < self.burst_until:
            return (BURST_VX, 0.0, 0.0, 0.0)

        if gate is not None:
            self.last_det_t = t_now
            self.current_gate_id = gate["id"]
            cx, cy = gate["center"]
            s = gate["size_norm"]

            # сглаживание
            self.cx_ema = self._ema(cx, self.cx_ema)
            self.cy_ema = self._ema(cy, self.cy_ema)
            self.s_ema = self._ema(s, self.s_ema)

            # ошибки по центру (-1..+1)
            ex = (self.cx_ema - w * 0.5) / (0.5 * w)
            ey = (self.cy_ema - h * 0.5) / (0.5 * h)

            # --- оценка несоосности по yaw по площадям маркеров ---
            align_e, nmarks = _yaw_error_from_marker_areas(gate.get("raw", []))
            self.last_align_e = align_e

            # вес, с которым используем «перпендикулярность» (чем ближе — тем сильнее)
            w_perp = 0.0
            if self.s_ema is not None and nmarks >= 3:
                if self.s_ema > ALIGN_USE_FROM:
                    w_perp = np.clip(
                        (self.s_ema - ALIGN_USE_FROM) / max(1e-6, (ALIGN_FULL_FROM - ALIGN_USE_FROM)),
                        0.0, 1.0
                    )

            # управляющие
            yaw_center = -K_YAW * float(ex)                 # наводка центром
            yaw_perp   =  K_YAW_PERP * float(align_e)       # доворот к нормали
            yaw_rate   = (1.0 - w_perp) * yaw_center + w_perp * yaw_perp

            vy = (K_VY * float(ex)) if USE_STRAFE else 0.0
            vz = -K_VZ * float(ey)
            vx = BASE_VX * (1.0 + 0.4 * max(0.0, float(self.s_ema or 0.0)))

            # если близко, но ещё не выровнялись — не разгоняемся, даём времени довернуть
            if self.s_ema is not None and self.s_ema > PASS_SIZE_TH and abs(align_e) > ALIGN_TOL:
                vx = min(vx, NEAR_SLOW_VX)

            # ограничения
            vx = float(np.clip(vx, 0.0, MAX_VX))
            vy = float(np.clip(vy, -MAX_VY, MAX_VY))
            vz = float(np.clip(vz, -MAX_VZ, MAX_VZ))
            yaw_rate = float(np.clip(yaw_rate, -MAX_YAWR, MAX_YAWR))

            # пролёт: достаточно близко и выровнялись по нормали
            if (self.s_ema is not None and self.s_ema > PASS_SIZE_TH and
                    abs(align_e) <= ALIGN_TOL):
                self.burst_until = t_now + PASS_BURST_T
                return (BURST_VX, 0.0, 0.0, 0.0)

            return (vx, vy, vz, yaw_rate)

        else:
            dt = t_now - self.last_det_t
            if dt < NO_DET_HOLD_T:
                return (0.0, 0.0, 0.0, 0.0)
            else:
                return (BASE_VX * 0.4, 0.0, 0.0, 0.0)


def draw_overlay(img, gate, ctrl, cmd):
    if img is None:
        return
    vis = img.copy()
    h, w = vis.shape[:2]

    # сетка центра
    cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 1)

    if gate is not None:
        # рисуем углы меток
        for quad in gate["raw"]:
            quad = quad.astype(int)
            cv2.polylines(vis, [quad], True, (0, 255, 255), 2)
            c = quad.mean(axis=0).astype(int)
            cv2.circle(vis, tuple(c), 3, (0, 255, 255), -1)

        cx, cy = map(int, gate["center"])
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
        txt = f"Gate id={gate['id']} size={gate['size_norm']:.2f}"
        cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(vis, f"align={getattr(ctrl, 'last_align_e', 0.0):.2f}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)

    vx, vy, vz, yawr = cmd
    cv2.putText(vis, f"CMD vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yawr={yawr:.2f}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Gates ArUco", vis)


def fly_through_gates(pio: Pioneer, cam: Camera):
    det = GateDetector()
    ctrl = Controller()

    # взлёт и набор высоты
    pio.arm(); time.sleep(1.0)
    pio.takeoff(); time.sleep(1.5)
    coord = pio.get_local_position_lps()
    pio.go_to_local_point(coord[0], coord[1], TAKEOFF_ALT, 0.0)
    t0 = time.time()
    while not pio.point_reached():
        time.sleep(0.05)

    print("Takeoff complete. Starting gate-following loop.")
    period = 1.0 / CMD_HZ

    try:
        while True:
            t_now = time.time()
            frame = cam.get_cv_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gates = det.detect_gates(gray)
            target = gates[0] if len(gates) > 0 else None

            h, w = frame.shape[:2]
            vx, vy, vz, yawr = ctrl.update(target, t_now, w, h)
            # команды скоростей в СК дрона — слать постоянно
            pio.set_manual_speed_body_fixed(vy, vx, vz, -yawr)

            if SHOW_DEBUG:
                draw_overlay(frame, target, ctrl, (vx, vy, vz, yawr))
                # выход по ESC
                if cv2.waitKey(1) == 27:
                    break

            # (опционально) критерий завершения миссии: прошло N пролётов или таймер
            # здесь — просто работаем бесконечно, пока не прервут

            # выдержка частоты
            dt = time.time() - t_now
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        print("Landing...")
        pio.land()
        time.sleep(2.0)
        pio.disarm()
        cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("Использование: python gates_aruco.py <mavlink_port> <camera_port>")
        sys.exit(1)

    pioneer = Pioneer(ip=args[0], mavlink_port=int(args[1]))
    camera = Camera(ip=args[0], port=int(args[2]))

    # В этом скрипте поток один — управление и обработка кадров вместе.
    # Если нужно — можно вынести get_cv_frame() в отдельный поток.

    fly_through_gates(pioneer, camera)



if __name__ == "__main__":
    main()
