import sys
import time
import cv2
import numpy as np
import threading
from pioneer_sdk import Pioneer
from pioneer_sdk import Camera
import cv2.aruco as aruco

# Настройка ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

GREEN_LOW = np.array((130, 200, 130))
GREEN_HIGH = np.array((255, 255, 255))

# Общие данные между потоками
shared_data = {
    "gate": None,       # (cx, cy, w, h)
    "aruco": {}         # {id: (cx, cy)}
}
data_lock = threading.Lock()
TARGET_HEIGHT = 1.5  # фиксированная высота

def find_green_gate(frame):
    """Находит зелёные ворота и возвращает их центр и размер"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, GREEN_LOW, GREEN_HIGH)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, mask, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:
        return frame, mask, None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    frame = cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame, mask, (cx, cy, w, h)

def camera_loop(pioneer, cam=None):
    """Поток камеры с ArUco и зелёными воротами"""
    while True:
        frame = None
        if cam:
            frame = cam.get_cv_frame()
        else:
            frame_bytes = pioneer.get_frame()
            if frame_bytes:
                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Найти ворота
        frame, mask, gate_info = find_green_gate(frame)
        with data_lock:
            shared_data["gate"] = gate_info

        # Найти ArUco маркеры
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        aruco_dict_local = {}
        if ids is not None:
            for i, c in zip(ids.flatten(), corners):
                cx = int(c[0][:,0].mean())
                cy = int(c[0][:,1].mean())
                aruco_dict_local[i] = (cx, cy)
            aruco.drawDetectedMarkers(frame, corners, ids)
        with data_lock:
            shared_data["gate"] = gate_info
            shared_data["mask"] = mask
            shared_data["frame_shape"] = frame.shape[:2]  # (height, width)
            shared_data["aruco"] = aruco_dict_local

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cv2.destroyAllWindows()

# --- Глобальные коэффициенты и настройки управления ---
LATERAL_SPEED_COEFF = 0.50    # скорость влево/вправо (м/с)
FORWARD_SPEED_COEFF = 1    # скорость вперёд (м/с)
YAW_SPEED_COEFF = 0.25        # коэффициент поворота
CENTER_TOLERANCE_PX = 20      # допуск центра ворот
ASPECT_RATIO_THRESHOLD = 30  # порог для узких ворот
VERT_BAR_DIFF_THRESH = 15     # порог разницы балок (пиксели)
CONTROL_LOOP_SLEEP = 0.2     # задержка цикла (сек)
MAX_YAW_SIMPLE = 0.6          # лимит yaw скорости
MAX_LATERAL_SIMPLE = 0.6      # лимит боковой скорости

def control_loop(pioneer):
    """
    Управление дроном по данным камеры:
    - удерживает высоту 1.5 м
    - центрируется по воротам
    - проходит через ворота
    - при отсутствии ворот ищет ArUco
    """
    pioneer.arm()
    time.sleep(1)
    pioneer.takeoff()
    print(f"[*] Drone airborne at height {TARGET_HEIGHT} м (удерживается)")

    visited_aruco = set()

    try:
        while True:
            with data_lock:
                gate_info = shared_data.get("gate")
                aruco_dict_local = dict(shared_data.get("aruco", {}))
                mask = shared_data.get("mask", None)
                frame_shape = shared_data.get("frame_shape", None)

            if frame_shape is None:
                pioneer.set_manual_speed_body_fixed(0, 0, 0, 0)
                time.sleep(CONTROL_LOOP_SLEEP)
                continue

            frame_h, frame_w = frame_shape
            frame_cx = frame_w // 2
            frame_cy = frame_h // 2

            # --- Если ворота не видны ---
            if gate_info is None:
                print("[!] Ворота не найдены")

                # Попробуем найти ArUco
                if aruco_dict_local:
                    aid, (ax, ay) = sorted(aruco_dict_local.items())[0]
                    dx = ax - frame_cx

                    if abs(dx) > CENTER_TOLERANCE_PX:
                        yaw_dir = np.sign(dx)
                        yaw_speed = min(MAX_YAW_SIMPLE, abs(dx) / frame_w * YAW_SPEED_COEFF * 10)
                        print(f"[↻] Поворачиваюсь {'вправо' if yaw_dir > 0 else 'влево'} к ArUco {aid}")
                        pioneer.set_manual_speed_body_fixed(0, 0, 0, yaw_dir * yaw_speed)
                    else:
                        print(f"[→] Двигаюсь вперёд к ArUco {aid}")
                        pioneer.set_manual_speed_body_fixed(FORWARD_SPEED_COEFF, 0, 0, 0)
                else:
                    print("[·] Нет ворот и ArUco — зависаю")
                    pioneer.set_manual_speed_body_fixed(0, 0, 0, 0)

                time.sleep(CONTROL_LOOP_SLEEP)
                continue

            # --- Если ворота найдены ---
            cx, cy, w, h = gate_info
            dx = cx - frame_cx

            # === 1. Проверка геометрии ворот ===
            # Попробуем определить 4-угольник ворот
            quad = None
            # --- Вариант 1: есть зелёная маска ворот ---
            if mask is not None:
                # ищем контуры на маске
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    # аппроксимируем контур до полигона
                    peri = cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                    if len(approx) == 4:
                        quad = approx.reshape(-1, 2)
            # --- Вариант 2: если нет — попробуем 4 ArUco маркера ---
            if quad is None and len(aruco_dict_local) >= 4:
                # просто возьмем координаты всех найденных ArUco
                pts = np.array(list(aruco_dict_local.values()), dtype=np.float32)
                # сделаем выпуклую оболочку по этим точкам
                hull = cv2.convexHull(pts)
                if len(hull) == 4:
                    quad = hull.reshape(-1, 2)
            # --- Если нашли четырёхугольник ---
            if quad is not None:
                # Сортируем точки по x, чтобы разделить левую и правую сторону
                quad = sorted(quad, key=lambda p: p[0])
                left_points = sorted(quad[:2], key=lambda p: p[1])  # верх/низ слева
                right_points = sorted(quad[2:], key=lambda p: p[1])  # верх/низ справа

                # длины вертикальных сторон
                left_length = np.linalg.norm(np.array(left_points[1]) - np.array(left_points[0]))
                right_length = np.linalg.norm(np.array(right_points[1]) - np.array(right_points[0]))
                diff = left_length - right_length

                if abs(diff) > VERT_BAR_DIFF_THRESH:
                    if diff > 0:
                        # левая сторона длиннее → сместиться влево
                        print(f"[⇠] Левая сторона длиннее ({diff:.1f}px) — двигаюсь вправо")
                        pioneer.set_manual_speed_body_fixed(LATERAL_SPEED_COEFF, 0, 0, 0)
                    else:
                        # правая сторона длиннее → сместиться вправо
                        print(f"[⇢] Правая сторона длиннее ({diff:.1f}px) — двигаюсь влево")
                        pioneer.set_manual_speed_body_fixed(-LATERAL_SPEED_COEFF, 0, 0, 0)
                    time.sleep(CONTROL_LOOP_SLEEP)
            else:
                # нет корректного четырёхугольника — пропускаем шаг
                print("[·] Геометрия ворот не определена (нет 4 углов) — пропускаю корректировку")

            # === 2. Центровка ворот по горизонтали ===
            if abs(dx) > CENTER_TOLERANCE_PX:
                if dx > CENTER_TOLERANCE_PX:
                    print(f"[↻] Ворота справа — поворачиваюсь вправо ({dx}px)")
                    pioneer.set_manual_speed_body_fixed(0, 0, 0, YAW_SPEED_COEFF)
                    time.sleep(CONTROL_LOOP_SLEEP)
                    continue
                elif dx < -CENTER_TOLERANCE_PX:
                    print(f"[↺] Ворота слева — поворачиваюсь влево ({dx}px)")
                    pioneer.set_manual_speed_body_fixed(0, 0, 0, -YAW_SPEED_COEFF)
                    time.sleep(CONTROL_LOOP_SLEEP)
                    continue

            # === 3. Центр найден — двигаемся вперёд ===
            print(f"[→] Центр ворот найден (dx={dx}px). Двигаюсь вперёд.")
            pioneer.set_manual_speed_body_fixed(0, FORWARD_SPEED_COEFF, 0, 0)
            time.sleep(CONTROL_LOOP_SLEEP*2)

            # === 4. Пройдены ли ворота? ===
            if w > frame_w * 0.6:
                print("[✔] Ворота пройдены")
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("[*] Landing...")
        pioneer.land()
        time.sleep(2)
        pioneer.disarm()

def main():
    if len(sys.argv) < 4:
        print("Usage: python main.py <ip> <port> <camera_port>")
        sys.exit(1)

    ip = sys.argv[1]
    port = int(sys.argv[2])
    camera_port = int(sys.argv[3])

    print(f"[*] Connecting to Pioneer at {ip}:{port}, camera port {camera_port}")

    pioneer = Pioneer(ip=ip, mavlink_port=port)
    cam = Camera(ip=ip, port=camera_port)

    cam_thread = threading.Thread(target=camera_loop, args=(pioneer, cam))
    ctrl_thread = threading.Thread(target=control_loop, args=(pioneer,))

    cam_thread.start()
    ctrl_thread.start()

    cam_thread.join()
    ctrl_thread.join()


if __name__ == "__main__":
    main()
