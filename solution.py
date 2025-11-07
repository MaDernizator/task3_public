# -*- coding: utf-8 -*-
# Прохождение ворот по зелёному свечению рамки (без ArUco)
# Использует твой шаблон с двумя потоками: камера + управление

from pioneer_sdk import Camera
from pioneer_sdk import Pioneer

import time
import cv2
import threading
import sys
import math
import numpy as np
from dataclasses import dataclass, field

SHOW_MASKS = True  # показывать окна "Mask raw" и "Mask proc"

# -------------------- ПАРАМЕТРЫ --------------------

# Частоты и тайминги
CMD_HZ = 30.0                 # частота отправки команд скоростей
NO_DET_HOLD_T = 0.6           # если целей нет дольше этого — стоп/зависание
PASS_BURST_T = 0.65           # длительность "рывка" через ворота, сек

# Ограничения и коэффициенты контроллера
KP_LAT  = 0.9                 # коэффициент боковой коррекции vy
KP_VERT = 0.9                 # коэффициент вертикальной коррекции vz
KP_YAW  = 1.2                 # коэффициент поворота вокруг оси yaw

VX_MAX   = 0.7
VZ_MAX   = 0.7
YAW_MAX  = 1.2                # рад/с

FWD_MIN        = 0.25          # продольная скорость "подлёта", минимальная
FWD_MAX        = 0.9           # продольная скорость "подлёта", максимальная
FWD_GAIN = 1.2                 # чем меньше рамка, тем выше vx = VX_MIN + (1-area)*GAIN
PASS_BURST_VY = 1.3            # скорость во время рывка

# Критерии "готов к пролёту"
CENTER_TOL_X = 0.10           # допуск по центровке (по ширине кадра)
CENTER_TOL_Y = 0.10           # допуск по центровке (по высоте кадра)
CENTER_ROI_FRAC = 0.72  # доля кадра по ширине/высоте, где ищем ворота (центр кадра)
AREA_TO_PASS = 0.04     # порог пролёта по НОРМИРОВАННОЙ площади minAreaRect (оставь как есть)

HSV_H_LO, HSV_H_HI = 45, 85     # зелёный диапазон тона
HSV_S_LO, HSV_V_LO = 70, 60     # отсечь белое/серое

G_MARGIN   = 25                  # G должен быть > R и > B как минимум на этот порог
G_MIN_VAL  = 80                  # минимальный уровень G
CENTER_ROI_FRAC = 0.65           # окно поиска по центру

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
SHOW_MASKS = True
RING_ERODE_K = 11  # размер ядра для «ringness»
MIN_CONTOUR_FRAC = 0.001      # отсечение мелких шумов по площади от кадра

# Фильтрация
EMA_ERR = 0.6                 # эксп. сглаживание ошибок по центру
EMA_AREA = 0.5                # сглаживание площади

# Сколько ворот пройти (None — бесконечно)
MAX_GATES = None

# -------------------- ОБЩЕЕ СОСТОЯНИЕ --------------------

stop_event = threading.Event()

@dataclass
class VisionState:
    """Потокобезопасное хранилище результатов детекции."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    # сырые измерения
    has_target: bool = False
    cx: float = 0.0
    cy: float = 0.0
    angle_rad: float = 0.0
    area_norm: float = 0.0
    last_seen_ts: float = 0.0
    frame_wh: tuple = (0, 0)
    # оверлеи для показа
    last_vis: np.ndarray | None = None

vision = VisionState()

def setup_tuner():
    cv2.namedWindow("Tuner", cv2.WINDOW_AUTOSIZE)
    # расстояние по Hue до зеленого (центр ~60), градусы 0..90
    cv2.createTrackbar("H_dist", "Tuner", 18, 90, lambda x: None)
    # минимальная насыщенность и яркость
    cv2.createTrackbar("S_min",  "Tuner", 40, 255, lambda x: None)
    cv2.createTrackbar("V_min",  "Tuner", 50, 255, lambda x: None)
    # доминирование G над R/B в абсолюте (маржа) и по отношению (g/(r+b))
    cv2.createTrackbar("G_margin",   "Tuner", 15, 100, lambda x: None)
    cv2.createTrackbar("G_ratio_x100","Tuner", 115, 300, lambda x: None)  # 115 => 1.15
    # окно поиска по центру кадра
    cv2.createTrackbar("ROI_%", "Tuner", 65, 100, lambda x: None)

def read_tuner():
    return dict(
        H_dist = max(1,  cv2.getTrackbarPos("H_dist", "Tuner")),
        S_min  =        cv2.getTrackbarPos("S_min",  "Tuner"),
        V_min  =        cv2.getTrackbarPos("V_min",  "Tuner"),
        G_margin =      cv2.getTrackbarPos("G_margin","Tuner"),
        G_ratio  = max(100, cv2.getTrackbarPos("G_ratio_x100","Tuner"))/100.0,
        ROI_frac = max(40, cv2.getTrackbarPos("ROI_%", "Tuner"))/100.0
    )


def pulse_cmd(pioneer, vx, vy, vz, yaw_rate, duration, hz=20.0):
    dt = 1.0 / hz
    t_end = time.time() + duration
    while time.time() < t_end:
        pioneer.set_manual_speed_body_fixed(vx, vy, vz, yaw_rate)  # слать постоянно!
        time.sleep(dt)

def calibrate_yaw_sign(pioneer, get_ex, timeout=2.0):
    """
    Небольшой импульс yaw, чтобы понять где «плюс».
    get_ex() -> текущая горизонтальная ошибка [-1..1] или None, если цели нет.
    Возвращает +1 или -1.
    """
    t0 = time.time()
    ex0 = None
    # ждём устойчивого кадра с воротами
    while time.time() - t0 < timeout:
        ex = get_ex()
        if ex is not None:
            ex0 = ex
            break
        pioneer.set_manual_speed_body_fixed(0, 0, 0, 0)
        time.sleep(0.05)
    if ex0 is None:
        return +1  # дефолт

    # короткий импульс +yaw
    pulse_cmd(pioneer, 0, 0, 0, +0.15, 0.30)  # 0.15 рад/с * 0.3 с = небольшой поворот
    time.sleep(0.05)
    ex1 = get_ex() or ex0

    # если |ex| уменьшилась — «плюс» правильный
    return +1 if abs(ex1) < abs(ex0) else -1


# -------------------- ДЕТЕКЦИЯ --------------------

def detect_green_gate(frame, params):
    """
    Возвращает (det|None, masks):
      det = {cx, cy, angle_rad, norm_area, contour}
      masks = {"huedist", "hsv_basic", "gdom", "or", "proc"}
    """
    H, W = frame.shape[:2]
    roi_frac = float(params["ROI_frac"])

    # --- Hue-distance к «зелёному» (~60 градусов), + пороги по S/V ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # круговая дистанция по тону
    dh = np.abs(h.astype(np.int16) - 60)
    dh = np.minimum(dh, 180 - dh).astype(np.uint8)
    mask_h = (dh <= params["H_dist"]).astype(np.uint8) * 255
    mask_s = (s >= params["S_min"]).astype(np.uint8) * 255
    mask_v = (v >= params["V_min"]).astype(np.uint8) * 255
    hsv_basic = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))

    # --- Доминирование G в BGR (помогает при низкой S из-за блика) ---
    b, g, r = cv2.split(frame)
    ratio = (g.astype(np.float32) + 1.0) / (r.astype(np.float32) + b.astype(np.float32) + 1.0)
    gdom_bool = ((g > r + params["G_margin"]) & (g > b + params["G_margin"])) | (ratio >= params["G_ratio"])
    gdom = (gdom_bool.astype(np.uint8) * 255)

    # --- Комбинация: достаточно одного признака (OR), чтобы не «терять» блеклый зелёный ---
    comb_or = cv2.bitwise_or(hsv_basic, gdom)

    # --- Центральный ROI ---
    cw, ch = int(W * roi_frac), int(H * roi_frac)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    roi = np.zeros_like(comb_or); roi[y0:y0+ch, x0:x0+cw] = 255
    proc = cv2.bitwise_and(comb_or, roi)

    # --- Морфология ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    proc = cv2.medianBlur(proc, 5)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, k, iterations=2)
    proc = cv2.dilate(proc, k, iterations=1)

    # --- Контур и геометрия ворот ---
    cnts, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, {"huedist": mask_h, "hsv_basic": hsv_basic, "gdom": gdom, "or": comb_or, "proc": proc}

    best, best_score = None, -1.0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 0.001 * W * H:  # мусор
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
        if rw < 10 or rh < 10:
            continue
        aspect = min(rw, rh) / max(rw, rh)
        if aspect < 0.6:
            continue
        center_score = 1.0 - min(np.hypot((cx - W/2)/(W/2), (cy - H/2)/(H/2)), 1.0)
        rect_area_norm = (rw * rh) / float(W * H)
        score = 1.4*center_score + 1.0*aspect + 0.5*rect_area_norm
        if score > best_score:
            best_score = score
            best = (cx, cy, rw, rh, ang, cnt, rect_area_norm)

    if best is None:
        return None, {"huedist": mask_h, "hsv_basic": hsv_basic, "gdom": gdom, "or": comb_or, "proc": proc}

    cx, cy, rw, rh, ang, cnt, rect_area_norm = best
    if ang < -45: ang += 90.0
    det = dict(cx=float(cx), cy=float(cy),
               angle_rad=float(np.deg2rad(ang)),
               norm_area=float(rect_area_norm),
               contour=cnt)
    return det, {"huedist": mask_h, "hsv_basic": hsv_basic, "gdom": gdom, "or": comb_or, "proc": proc}



# -------------------- ПОТОК КАМЕРЫ --------------------

def camera_stream(camera: Camera):
    setup_tuner()
    ex_f = ey_f = area_f = None

    while not stop_event.is_set():
        frame = camera.get_cv_frame()
        if frame is None:
            if cv2.waitKey(1) == 27: stop_event.set()
            continue

        params = read_tuner()
        det, masks = detect_green_gate(frame, params)

        h, w = frame.shape[:2]
        with vision.lock:
            vision.frame_wh = (w, h)
            if det is not None:
                ex = (det["cx"] - w/2) / (w/2)
                ey = (det["cy"] - h/2) / (h/2)
                if ex_f is None:
                    ex_f, ey_f, area_f = ex, ey, det["norm_area"]
                else:
                    ex_f = 0.6*ex_f + 0.4*ex
                    ey_f = 0.6*ey_f + 0.4*ey
                    area_f = 0.5*area_f + 0.5*det["norm_area"]

                vision.has_target = True
                vision.cx = det["cx"]; vision.cy = det["cy"]
                vision.angle_rad = det["angle_rad"]
                vision.area_norm = float(area_f)
                vision.last_seen_ts = time.time()
            else:
                vision.has_target = False

        vis = frame.copy()
        cv2.putText(vis, "ESC to stop", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.drawMarker(vis, (w//2, h//2), (255,255,255), cv2.MARKER_CROSS, 24, 2)
        if det is not None:
            cv2.drawContours(vis, [det["contour"]], -1, (0,255,0), 2)
            cv2.putText(vis, f"area={vision.area_norm:.3f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Socket Camera", vis)
        if SHOW_MASKS:
            cv2.imshow("Mask HueDist",  masks["huedist"])
            cv2.imshow("Mask HSV basic",masks["hsv_basic"])
            cv2.imshow("Mask G-dominance", masks["gdom"])
            cv2.imshow("Mask OR (Hue|G)", masks["or"])
            cv2.imshow("Mask proc (ROI+morph)", masks["proc"])

        if cv2.waitKey(1) == 27:
            stop_event.set()

    cv2.destroyAllWindows()

# -------------------- ПОТОК УПРАВЛЕНИЯ --------------------

def pioneer_control(pioneer: Pioneer):
    """ALIGN -> PASS -> ALIGN. Горизонт гасим yaw-ом по ex; x-канал почти не трогаем."""
    # Параметры
    CMD_HZ = 30.0
    FWD_MIN, FWD_MAX, FWD_GAIN = 0.25, 0.9, 1.2      # vy (вперёд)
    KP_YAW_EX, YAW_MAX = 1.2, 1.0                    # yaw_rate = sign * KP_YAW_EX * ex
    VX_MAX = 0.25                                     # ограничим боковой x, чтобы не «улетал вбок»
    PASS_BURST_VY, PASS_BURST_T = 1.0, 0.6
    CENTER_TOL_X, CENTER_TOL_Y = 0.10, 0.10
    AREA_TO_PASS = 0.04
    NO_DET_HOLD_T = 0.6

    # Старт
    pioneer.arm(); time.sleep(1.0)
    pioneer.takeoff(); time.sleep(2.0)

    # Функция для чтения текущего ex из vision
    def get_ex_from_vision():
        with vision.lock:
            if not vision.has_target or vision.frame_wh == (0, 0):
                return None
            w = vision.frame_wh[0]
            return (vision.cx - w/2) / (w/2)

    # Автокалибровка знака yaw (1–2 кадра)
    yaw_sign = calibrate_yaw_sign(pioneer, get_ex_from_vision)

    state, pass_until = "ALIGN", 0.0
    dt = 1.0 / CMD_HZ
    last_seen = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()
            with vision.lock:
                has = vision.has_target
                w, h = vision.frame_wh
                cx, cy = vision.cx, vision.cy
                area = vision.area_norm
                angle = vision.angle_rad  # больше не используем для yaw

            if has and w and h:
                ex = (cx - w/2) / (w/2)
                ey = (cy - h/2) / (h/2)
                last_seen = now
            else:
                ex = ey = 0.0

            if state == "PASS":
                if now < pass_until:
                    # pioneer.set_manual_speed_body_fixed(0.0, PASS_BURST_VY, 0.0, 0.0)
                    time.sleep(dt); continue
                else:
                    state = "ALIGN"

            if has:
                # 1) крутимся так, чтобы gate был по центру по X
                yaw_rate = float(np.clip(yaw_sign * KP_YAW_EX * ex, -YAW_MAX, YAW_MAX))

                # 2) боком почти не едем, только мягкая компенсация остатка
                vx = float(np.clip(+0.2 * ex, -VX_MAX, VX_MAX))  # можно поставить 0.0, если нужно

                # 3) летим вперёд по «размеру ворот» (area_norm из minAreaRect)
                vy = float(np.clip(FWD_MIN + (1.0 - area) * FWD_GAIN, FWD_MIN, FWD_MAX))

                # 4) высоту держим по центру по Y-кадра
                vz = float(np.clip(-0.9 * ey, -0.7, 0.7))

                # критерий «готов к пролёту»
                ready = (abs(ex) < CENTER_TOL_X) and (abs(ey) < CENTER_TOL_Y) and (area >= AREA_TO_PASS)
                if ready:
                    state = "PASS"
                    pass_until = now + PASS_BURST_T
                    # pioneer.set_manual_speed_body_fixed(0.0, PASS_BURST_VY, 0.0, 0.0)
                else:
                    pass
                    # pioneer.set_manual_speed_body_fixed(vx, vy, vz, yaw_rate)
            else:
                # цели нет
                if (now - last_seen) > NO_DET_HOLD_T:
                    pass
                    # pioneer.set_manual_speed_body_fixed(0, 0, 0, 0)
                else:
                    pass
                    # pioneer.set_manual_speed_body_fixed(0, FWD_MIN * 0.5, 0, 0)

            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            pioneer.set_manual_speed_body_fixed(0, 0, 0, 0)
            time.sleep(0.05)
        pioneer.land(); time.sleep(2.0); pioneer.disarm()

# -------------------- MAIN --------------------

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Не переданы необходимые аргументы: порт Пионера и порт камеры. Пример: python main.py 8000 18000")
        exit(1)

    pioneer = Pioneer(ip="127.0.0.1", mavlink_port=int(args[0]))
    pioneer_camera = Camera(ip="127.0.0.1", port=int(args[1]))

    camera_thread = threading.Thread(target=camera_stream, args=(pioneer_camera,), daemon=True)
    pioneer_thread = threading.Thread(target=pioneer_control, args=(pioneer,), daemon=True)

    try:
        camera_thread.start()
        pioneer_thread.start()

        while camera_thread.is_alive() and pioneer_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            pioneer.land()
            time.sleep(1.0)
            pioneer.disarm()
        except Exception:
            pass

if __name__ == "__main__":
    main()
