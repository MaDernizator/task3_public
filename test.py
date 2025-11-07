from pioneer_sdk import Camera
from pioneer_sdk import Pioneer

import time
import cv2
import threading
import sys


def camera_stream(camera: Camera):
    """Получение изображения с камеры Пионера"""
    while True:
        frame = camera.get_cv_frame()
        if frame is not None:
            cv2.imshow("Socket Camera", frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def pioneer_control(pioneer: Pioneer):
    """Пример управления Пионером"""
    pioneer.arm()
    time.sleep(1)
    pioneer.takeoff()
    pioneer.go_to_local_point(3, 0, 2, 0)
    time.sleep(3)

    points = [[0, 0], [1, 0], [1, 1], [0, 1]]  #  Квадрат с длинной стороны 1
    height = 2

    while True:
        for point in points:
            pioneer.go_to_local_point(*point, height, 0)
            while not pioneer.point_reached():
                pass
            time.sleep(0.2)



def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "Не переданы необходимые аргументы: порт Пионера, порт камеры Пионера, порт Геобота. Пример: python main.py 8000 18000 8001"
        )
        exit(1)

    pioneer = Pioneer(ip=args[0], mavlink_port=int(args[1]))
    pioneer_camera = Camera(ip=args[0], port=int(args[2]))


    camera_thread = threading.Thread(target=camera_stream, args=(pioneer_camera,))
    pioneer_thread = threading.Thread(target=pioneer_control, args=(pioneer,))

    try:
        camera_thread.start()
        pioneer_thread.start()

        camera_thread.join()
        pioneer_thread.join()
    except KeyboardInterrupt:
        pioneer.land()
        pioneer.disarm()


if __name__ == "__main__":
    main()