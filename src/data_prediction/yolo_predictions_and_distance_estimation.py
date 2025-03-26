import torch
import cv2
from time import time
from ultralytics import YOLO
import depth_pro

class ObjectDetection_and_Distance_Estimation:
    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = 'mps' if torch.mps.is_available() else 'cpu'
        self.yolo_model = self.load_yolo_model()
        self.depth_model, self.transform = depth_pro.create_model_and_transforms()
        self.depth_model = self.depth_model.to(self.device)
        self.depth_model.eval()

    @staticmethod
    def load_yolo_model():
        yolo_model = YOLO("object_detection.pt")
        yolo_model.fuse()
        return yolo_model

    def predict(self, frame):
        results = self.yolo_model(frame)
        return results

    def plot_bboxes_and_depth_estimation(self, results, frame):
        object_boxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = result.names[int(cls)]
                object_boxes.append((x1, y1, x2, y2, class_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name}"
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 1.0
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

                text_x = x1
                text_y = y1 - 10
                rect_x1 = text_x - 5
                rect_y1 = text_y - text_size[1] - 10
                rect_x2 = text_x + text_size[0] + 5
                rect_y2 = text_y + 5
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_input = self.transform(image).to(self.device)
        f_px = None
        prediction = self.depth_model.infer(depth_input, f_px=f_px)
        depth = prediction["depth"].squeeze().cpu().numpy()

        for x1, y1, x2, y2, class_name in object_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth_value = depth[center_y, center_x]
            text = f'{class_name} Depth: {depth_value:.2f}m'

            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1.2
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_x = x1
            text_y = y1 - 10
            rect_x1 = text_x - 5
            rect_y1 = text_y - text_size[1] - 10
            rect_x2 = text_x + text_size[0] + 5
            rect_y2 = text_y + 5

            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Could not open video capture."

        while True:
            ret, frame = cap.read()
            assert ret, "Error: Could not read frame."

            start_time = time()
            results = self.predict(frame)
            frame = self.plot_bboxes_and_depth_estimation(results, frame)
            
            end_time = time()

            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            cv2.imshow('YOLO and Depth Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()