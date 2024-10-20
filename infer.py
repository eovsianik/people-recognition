import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import argparse


class KalmanTracker:
    def __init__(self):
        """
        Инициализация фильтра Калмана для отслеживания объектов
        Используется для предсказания и обновления состояний объектов на основе наблюдений
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Матрица перехода состояния
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Матрица наблюдения
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # Шум процесса и наблюдений
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.R[2:, 2:] *= 10

    def predict(self) -> np.ndarray:
        """
        Предсказывает новое состояние объекта на следующем кадре.

        Возвращает:
            np.ndarray: Вектор предсказанных координат рамки (x1, y1, x2, y2).
        """
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox: np.ndarray) -> None:
        """
        Обновляет состояние фильтра Калмана на основе новых наблюдений.

        Параметры:
            bbox (np.ndarray): Координаты обнаруженной рамки объекта (x1, y1, x2, y2).
        """
        self.kf.update(np.array([bbox[0], bbox[1], bbox[2], bbox[3]]))


def draw_tracks(frame: np.ndarray, trackers: list):
    """
    Отрисовка треков на кадре с указанием ID трека.

    Параметры:
        frame: Кадр, на котором нужно отрисовать треки
        trackers: Список трекеров (объектов KalmanTracker) для каждого объекта на кадре
    """
    for tracker in trackers:
        predicted_bbox = tracker.predict()
        x1, y1, x2, y2 = map(int, predicted_bbox)

        # Отрисовка уменьшенной рамки
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Получение уверенности
        confidence = float(tracker.kf.x[4])

        # Нормализация уверенности
        confidence = np.clip(confidence, 0, 1)

        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def update_trackers(trackers: list, detections: list) -> list:
    """
    Обновляет трекеры на основе новых детекций

    Параметры:
        trackers: Список трекеров (объектов KalmanTracker)
        detections: Список обнаруженных детекций (bbox)

    Возвращает:
        list: Обновленный список трекеров
    """
    new_trackers = []

    for tracker in trackers:
        # Предсказание новой позиции
        predicted_bbox = tracker.predict()

        # Поиск ближайшей детекции к предсказанной позиции
        min_distance = float("inf")
        best_detection = None

        for detection in detections:
            distance = np.linalg.norm(
                np.array(predicted_bbox[:2]) - np.array(detection[:2])
            )
            if distance < min_distance:
                min_distance = distance
                best_detection = detection

        if best_detection is not None:
            tracker.update(best_detection)  
            detections.remove(best_detection)

        # Сохраняем обновлённый трекер
        new_trackers.append(tracker)

    return new_trackers


def get_detections(results) -> list:
    """
    Извлекает детекции объектов из результатов модели

    Параметры:
        results: Результаты детекции объектов
    Возвращает:
        list: Список детекций, содержащий координаты рамок для обнаруженных людей
    """
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]  
            cls = int(box.cls[0]) 

            if cls == 0 and conf > 0.3:
                detections.append([x1, y1, x2, y2])

    return detections


def process_video(input_path: str, output_path: str, model) -> None:
    """
    Обрабатывает видео, выполняя детекцию объектов и трекинг

    Параметры:
        input_path: Путь к входному видео.
        output_path: Путь к выходному видео с отрисованными детекциями
        model: Модель для детекции объектов
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    trackers = []
    track_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция объектов с помощью YOLOv8
        results = model(frame)

        detections = get_detections(results)

        # Обновление трекеров
        new_trackers = update_trackers(trackers, detections)

        # Создание новых трекеров для оставшихся детекций
        for detection in detections:
            new_tracker = KalmanTracker()
            new_tracker.update(detection)
            new_trackers.append(new_tracker)
            track_id += 1

        trackers = new_trackers

        # Отрисовка треков
        draw_tracks(frame, trackers)

        # Запись кадра в выходной файл
        out.write(frame)

    cap.release()
    out.release()


def main():
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(
        description="Обработка видео с трекингом объектов."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Путь к входному видеофайлу."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Путь для сохранения обработанного видеофайла.",
    )
    args = parser.parse_args()

    # Загрузка модели YOLOv8
    model = YOLO("yolov8n.pt")

    # Запуск обработки видео
    process_video(args.input, args.output, model)


if __name__ == "__main__":
    main()
