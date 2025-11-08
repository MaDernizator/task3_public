import argparse
import sys
from pioneer_sdk import Pioneer, Camera

import core
from core import VisionCfg, ControlCfg, CommandBus
from vision import VisionWorker
from core import FlightController, CommandSender


def parse_args(argv):
    p = argparse.ArgumentParser(description="Gate following (modular).")
    p.add_argument("ip", type=str, help="Pioneer IP")
    p.add_argument("port", type=int, help="MAVLink port")
    p.add_argument("camera_port", type=int, help="Camera port")
    return p.parse_args(argv[1:])


def main():
    args = parse_args(sys.argv)
    core.set_debug(False)

    # Конфиги (окна завязаны на DEBUG)
    vcfg = VisionCfg(show_windows=core.DEBUG)
    ccfg = ControlCfg()

    if core.DEBUG:
        core.dprint(f"[*] Connecting to Pioneer: {args.ip}:{args.port} | camera: {args.camera_port}")

    drone = Pioneer(ip=args.ip, mavlink_port=args.port)
    cam = Camera(ip=args.ip, port=args.camera_port)

    bus = CommandBus()

    vision = VisionWorker(drone, cam, vcfg)
    ctrl = FlightController(drone, vision, bus, ccfg, vcfg)
    sender = CommandSender(drone, bus, ccfg.cmd_hz)

    # Стартуем потоки
    sender.start()
    vision.start()
    ctrl.start()

    try:
        ctrl.join()
    finally:
        # Корректная остановка
        ctrl.stop()
        vision.stop()
        sender.stop()
        vision.join(timeout=1.0)
        sender.join(timeout=1.0)


if __name__ == "__main__":
    main()