# -*- coding: utf-8 -*-
"""
Ядро: глобальный debug, конфиги, утилиты, контроллер полёта и отправщик команд.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt

# ------------------------------- Debug ------------------------------------ #

DEBUG: bool = False

def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# --------------------------- Конфигурация --------------------------------- #

TARGET_ALT_M = 1.5  # информативная целевая высота (для логов)

@dataclass
class VisionCfg:
    # Пороги по зелёному (работаем в RGB после конвертации из BGR)
    green_lo: npt.NDArray[np.uint8] = field(
        default_factory=lambda: np.array((130, 200, 130), dtype=np.uint8)
    )
    green_hi: npt.NDArray[np.uint8] = field(
        default_factory=lambda: np.array((255, 255, 255), dtype=np.uint8)
    )
    min_area: int = 200                 # минимальная площадь контура ворот (px^2)
    show_windows: bool = False          # разрешать ли cv2.imshow()
    win_frame_name: str = "View"
    win_mask_name: str = "GateMask"

@dataclass
class ControlCfg:
    cmd_hz: float = 30.0                # частота отправки set_manual_speed_* (Гц)
    loop_hz: float = 15.0               # частота логики управления (Гц)
    gate_ema_beta: float = 0.35         # EMA для центра/размера ворот
    dx_deadband_px: int = 18            # мёртвая зона по центру
    seen_up_frames: int = 2             # сколько кадров подряд "видим", чтобы засчитать
    seen_down_frames: int = 6           # сколько кадров подряд "не видим", чтобы потерять
    fwd_min: float = 0.25               # минимум подачи вперёд (м/с)
    fwd_max: float = 1.00               # максимум подачи вперёд (м/с)
    acc_max: float = 1.2                # максимум изменения скорости вперёд (м/с^2)
    yaw_max: float = 0.8                # предел угловой скорости (рад/с)
    yaw_kp: float = 1.3                 # пропорциональный коэффициент по dx_norm
    yaw_slew: float = 2.5               # ограничение изменения yaw (рад/с^2)
    no_det_hold_s: float = 0.6          # как долго держим курс при пропадании цели
    pass_width_ratio: float = 0.6       # доля ширины кадра — критерий "пролёт"
    pass_burst_vy: float = 1.0          # рывок при пролёте (м/с)
    pass_burst_time: float = 0.7        # длительность рывка (с)

# ------------------------------ Утилиты ----------------------------------- #

def limit(x: float, lo: float, hi: float) -> float:
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
        dv = limit(target - self.out, -self.rate * dt, self.rate * dt)
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

    def get(self) -> Tuple[float, float, float, float]:
        with self._lock:
            return self._vx, self._vy, self._vz, self._yaw

# --------------------------- Контроллер/Отправка -------------------------- #

from pioneer_sdk import Pioneer  # импорт здесь, чтобы core не требовал SDK при импорте типов

class FlightController(threading.Thread):
    """Логика слежения за воротами и выдачи команд в CommandBus."""

    def __init__(self, pioneer: Pioneer, vision, bus: CommandBus, ccfg: ControlCfg, vcfg: VisionCfg):
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
                    yaw_des = limit(self._ccfg.yaw_kp * dx_norm, -self._ccfg.yaw_max, self._ccfg.yaw_max)
                yaw_cmd = self._yaw_limiter.step(yaw_des, dt)

                # скорость вперёд (ось Y body-fixed у Пионера — «вперёд»)
                alignment = 1.0 - min(1.0, abs(dx_norm) * 2.2)
                vy_des = self._ccfg.fwd_min + alignment * (self._ccfg.fwd_max - self._ccfg.fwd_min)

                # если цель долго не видим — плавный поиск поворотом
                if (not stable) and (now - self._last_seen_t > self._ccfg.no_det_hold_s):
                    yaw_cmd = self._yaw_limiter.step(0.35 * np.sign(self._last_dx_px) if self._last_dx_px != 0 else 0.3, dt)
                    vy_des = self._ccfg.fwd_min

                vy_cmd = self._vy_limiter.step(limit(vy_des, -self._ccfg.fwd_max, self._ccfg.fwd_max), dt)

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