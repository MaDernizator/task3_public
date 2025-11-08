# -*- coding: utf-8 -*-
"""
Переписанное решение с поддержкой флага --debug:
- Все отладочные принты и окна визуализации показываются только при включённом debug.
- По умолчанию работает «тихо» — без окон и лишних сообщений.
"""

import sys
import time
import cv2
import math
import argparse
import threading
import numpy as np
from dataclasses import dataclass, field
from pioneer_sdk import Pioneer, Camera
import cv2.aruco as aruco


# ------------------------------- Debug helper ------------------------------- #

DEBUG = False  # будет выставлен после парсинга аргументов

def dprint(*args, **kwargs):
    """Печатает только при включённом DEBUG."""
    if DEBUG:
        print(*args, **kwargs)


# ========================= Конфигурация и константы ========================= #

@dataclass
class VisionCfg:
    # Порог по зелёному (работаем в RGB после конвертации из BGR)
    green_lo: np.ndarray = field(default_factory=lambda: np.array((130, 200, 130), dtype=np.uint8))
    green_hi: np.ndarray = field(default_factory=lambda: np.array((255, 255, 255), dtype=np.uint8))
    min_area: int = 200                       # минимальная площадь контура ворот (px^2)
    show_windows: bool = False                # окна cv2.imshow() (включаются флагом --debug)
    win_frame_name: str = "View"
    win_mask_name: str = "GateMask"


@dataclass
class ControlCfg:
    cmd_hz: float = 30.0                      # частота отправки set_manual_speed_* (Гц)
    loop_hz: float = 15.0                     # частота логики управления (Гц)
    gate_ema_beta: float = 0.35               # EMA-сглаживание центра/размера ворот
    dx_deadband_px: int = 18                  # мёртвая зона по центру
    seen_up_frames: int = 2                   # сколько кадров подряд "видим", чтобы засчитать
    seen_down_frames: int = 6                 # сколько кадров подряд "не видим", чтобы потерять
    fwd_min: float = 0.25                     # минимум подачи вперёд (м/с)
    fwd_max: float = 1.00                     # максимум подачи вперёд (м/с)
    acc_max: float = 1.2                      # ограничение приращения скорости вперёд (м/с^2)
    yaw_max: float = 0.8                      # предел угловой скорости (рад/с)
    yaw_kp: float = 1.3                       # пропорциональный коэффициент по dx_norm
    yaw_slew: float = 2.5                     # ограничение изменения yaw (рад/с^2)
    no_det_hold_s: float = 0.6                # как долго «держим курс» при пропадании цели
    pass_width_ratio: float = 0.6             # доля ширины кадра, когда считаем «пролёт»
    pass_burst_vy: float = 1.0                # рывок при пролёте (м/с)
    pass_burst_time: float = 0.7              # длительность рывка (с)


TARGET_ALT_M = 1.5  # высота для информации/логов


# ========================= Вспомогательные классы ========================== #

def _limit(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class EMAFilter:
    def __init__(self, beta: float, init: float | None = None):
        self.beta = float(beta)
        self.val = init

    def push(self, v: float | None) -> float | None:
        if v is None:
            return self.val
        if self.val is None:
            self.val = float(v)
        else:
            self.val = (1.0 - self.beta) * self.val + self.beta * float(v)
        return self.val


class SlewRateLimiter:
    def __init__(self, rate: float):
        self.rate = float(rate)
        self.out = 0.0

    def step(self, target: float, dt: float) -> float:
        dv = _limit(target - self.out, -self.rate * dt, self.rate * dt)
        self.out += dv
        return self.out


class HysteresisSeen:
    """Счётчик с асимметричным порогом 'видим/не видим'."""
    def __init__(self, up: int, down: int):
        self.up = up
        self.down = down
        self.cnt = 0
        self.state = False

    def update(self, seen: bool) -> bool:
        if seen:
            self.cnt = min(self.up, self.cnt + 1)
        else:
            self.cnt = max(-self.down, self.cnt - 1)

        if self.cnt >= self.up:
            self.state = True
        elif self.cnt <= -self.down:
            self.state = False
        return self.state


class CommandBus:
    """Потокобезопасная шина команд скорости."""
    def __init__(self):
        self._lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw = 0.0

    def set(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        with self._lock:
            self._vx, self._vy, self._vz, self._yaw = vx, vy, vz, yaw_rate

    def get(self) -> tuple[float, float, float, float]:
        with self._lock:
            return self._vx, self._vy, self._vz, self._yaw


# ============================== Видеопоток/визия =========================== #

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
        self._mask = None

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

        if self._vcfg.show_windows and DEBUG:
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
                if self._vcfg.show_windows and DEBUG:
                    aruco.drawDetectedMarkers(frame, corners, ids)

            with self._data_lock:
                self._gate = gate_info
                self._aruco_centers = ids_centers
                self._frame_shape = frame.shape[:2]
                self._mask = mask

            if self._vcfg.show_windows and DEBUG:
                cv2.imshow(self._vcfg.win_frame_name, frame)
                cv2.imshow(self._vcfg.win_mask_name, mask)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    self.stop()

        # безопасно даже если окон не было
        cv2.destroyAllWindows()

    def stop(self):
        self._stop.set()


# ============================== Контроллер полёта ========================== #

class FlightController(threading.Thread):
    """Логика слежения за воротами и выдачи команд в CommandBus."""

    def __init__(self, pioneer: Pioneer, vision: VisionWorker, bus: CommandBus, ccfg: ControlCfg, vcfg: VisionCfg):
        super().__init__(daemon=True)
        self._pioneer = pioneer
        self._vision = vision
        self._bus = bus
        self._ccfg = ccfg
        self._min_gate_area = vcfg.min_area
        self._stop = threading.Event()

        # фильтры
        self._ema_cx = EMAFilter(ccfg.gate_ema_beta)
        self._ema_w = EMAFilter(ccfg.gate_ema_beta)
        self._seen = HysteresisSeen(ccfg.seen_up_frames, ccfg.seen_down_frames)
        self._vy_limiter = SlewRateLimiter(ccfg.acc_max)
        self._yaw_limiter = SlewRateLimiter(ccfg.yaw_slew)

        # служебные
        self._last_seen_t = 0.0
        self._last_dx_px = 0.0

    def arm_and_takeoff(self):
        self._pioneer.arm()
        time.sleep(1.0)
        self._pioneer.takeoff()
        dprint(f"[*] Взлёт выполнен, держим высоту около {TARGET_ALT_M} м")

    def run(self):
        self.arm_and_takeoff()

        period = 1.0 / self._ccfg.loop_hz
        last = time.time()

        try:
            while not self._stop.is_set():
                now = time.time()
                dt = max(1e-3, now - last)
                last = now

                gate, _, shape = self._vision.snapshot()
                if shape is None:
                    # ещё нет данных — замрём
                    self._bus.set(0.0, 0.0, 0.0, 0.0)
                    time.sleep(period)
                    continue

                img_h, img_w = shape
                cx_frame = 0.5 * img_w

                # статус ворот (учитываем минимальную площадь)
                has_gate = gate is not None and (gate[2] * gate[3] > self._min_gate_area)
                stable = self._seen.update(has_gate)

                # обновление фильтров
                if has_gate:
                    cx, cy, w, h = gate
                    self._last_seen_t = now
                    self._last_dx_px = cx - cx_frame
                    cx_f = self._ema_cx.push(float(cx))
                    w_f = self._ema_w.push(float(w))
                else:
                    cx_f = self._ema_cx.val
                    w_f = self._ema_w.val

                # ошибка по центру
                dx_px = (cx_f - cx_frame) if cx_f is not None else self._last_dx_px
                dx_norm = float(dx_px) / max(1.0, img_w)

                # yaw с мёртвой зоной
                if abs(dx_px) <= self._ccfg.dx_deadband_px:
                    yaw_des = 0.0
                else:
                    yaw_des = _limit(self._ccfg.yaw_kp * dx_norm, -self._ccfg.yaw_max, self._ccfg.yaw_max)
                yaw_cmd = self._yaw_limiter.step(yaw_des, dt)

                # скорость вперёд (ось Y body-fixed у Пионера — «вперёд»)
                alignment = 1.0 - min(1.0, abs(dx_norm) * 2.2)
                vy_des = self._ccfg.fwd_min + alignment * (self._ccfg.fwd_max - self._ccfg.fwd_min)

                # если цель долго не видим — плавный поиск поворотом
                if (not stable) and (now - self._last_seen_t > self._ccfg.no_det_hold_s):
                    yaw_cmd = self._yaw_limiter.step(0.35 * np.sign(self._last_dx_px) if self._last_dx_px != 0 else 0.3, dt)
                    vy_des = self._ccfg.fwd_min

                vy_cmd = self._vy_limiter.step(_limit(vy_des, -self._ccfg.fwd_max, self._ccfg.fwd_max), dt)

                # критерий пролёта
                passed = (w_f is not None) and (w_f > self._ccfg.pass_width_ratio * img_w)

                # отправка команд (vx: бок, vy: вперёд, vz: высота, yaw: рысканье)
                self._bus.set(0.0, vy_cmd, 0.0, yaw_cmd)

                if passed and DEBUG:
                    dprint("[✓] Ворота пройдены — выполняю короткий рывок вперёд")
                    t0 = time.time()
                    while time.time() - t0 < self._ccfg.pass_burst_time and not self._stop.is_set():
                        self._bus.set(0.0, self._ccfg.pass_burst_vy, 0.0, 0.0)
                        time.sleep(1.0 / self._ccfg.cmd_hz)

                # поддерживаем частоту цикла
                sleep_left = period - (time.time() - now)
                if sleep_left > 0:
                    time.sleep(sleep_left)

        except KeyboardInterrupt:
            pass
        finally:
            self._bus.set(0.0, 0.0, 0.0, 0.0)
            if DEBUG:
                dprint("[*] Посадка...")
            self._pioneer.land()
            time.sleep(2.0)
            self._pioneer.disarm()

    def stop(self):
        self._stop.set()


# ============================ Отправка команд в АП ========================= #

class CommandSender(threading.Thread):
    """Постоянная ретрансляция текущей команды в автопилот с частотой cmd_hz."""

    def __init__(self, pioneer: Pioneer, bus: CommandBus, cmd_hz: float):
        super().__init__(daemon=True)
        self._pioneer = pioneer
        self._bus = bus
        self._period = 1.0 / float(cmd_hz)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            vx, vy, vz, yaw = self._bus.get()
            # Команду скорости нужно слать постоянно, пока хотим лететь с этой скоростью!
            self._pioneer.set_manual_speed_body_fixed(vx, vy, vz, yaw)
            time.sleep(self._period)

    def stop(self):
        self._stop.set()


# =================================== main ================================== #

def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="Gate following with optional debug visualization.")
    parser.add_argument("ip", type=str, help="Pioneer IP")
    parser.add_argument("port", type=int, help="MAVLink port")
    parser.add_argument("camera_port", type=int, help="Camera port")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug prints and OpenCV windows")
    args = parser.parse_args(argv[1:])
    return args


def main():
    global DEBUG
    args = parse_args(sys.argv)
    DEBUG = bool(args.debug)

    # Конфиги: окна и принты завязаны на DEBUG
    vcfg = VisionCfg(show_windows=DEBUG)
    ccfg = ControlCfg()

    if DEBUG:
        dprint(f"[*] Подключение к Pioneer: {args.ip}:{args.port} | камера: {args.camera_port}")

    drone = Pioneer(ip=args.ip, mavlink_port=args.port)
    cam = Camera(ip=args.ip, port=args.camera_port)

    bus = CommandBus()

    vision = VisionWorker(drone, cam, vcfg)
    ctrl = FlightController(drone, vision, bus, ccfg, vcfg)
    sender = CommandSender(drone, bus, ccfg.cmd_hz)

    # Стартуем
    sender.start()
    vision.start()
    ctrl.start()

    try:
        ctrl.join()
    finally:
        # Акуратно останавливаем всё
        ctrl.stop()
        vision.stop()
        sender.stop()
        vision.join(timeout=1.0)
        sender.join(timeout=1.0)


if __name__ == "__main__":
    main()
