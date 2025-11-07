"""
Шаблон для задания «Прохождение ворот».

Пример запуска:
    python main.py 127.0.0.1 8000 18000
"""

from pioneer_sdk import Pioneer, Camera
import sys

if __name__ == "__main__":
    ip = sys.argv[1]
    drone_port = int(sys.argv[2])
    camera_port = int(sys.argv[3])
      
    drone = Pioneer(ip=ip, mavlink_port=drone_port, simulator=True)
    camera = Camera(ip=ip, port=camera_port)
