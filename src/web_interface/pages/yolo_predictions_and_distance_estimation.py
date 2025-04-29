import torch
import cv2
from time import time
from ultralytics import YOLO
import depth_pro
import pyttsx3

class ObjectDetection_and_Distance_Estimation:
    def __init__(self, capture_index):
        """
            Class for performing real-time object detection and distance estimation 
            using a YOLOv12 model and a depth prediction model.
            It also provides voice feedback to guide the user.
        """
        
        self.capture_index = capture_index
        self.device = 'mps' if torch.mps.is_available() else 'cpu'
        self.yolo_model = self.load_yolo_model()
        self.depth_model, self.transform = depth_pro.create_model_and_transforms()
        self.depth_model = self.depth_model.to(self.device)
        self.depth_model.eval()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 100)

    @staticmethod
    def load_yolo_model():
        """
        Load and fuse the YOLOv12 model for object detection.

        Returns:
            YOLO: Loaded YOLO object detection model.
        """
        
        yolo_model = YOLO("object_detection_model-3.pt")
        yolo_model.fuse()
        return yolo_model

    def predict(self, frame):
        """
        Run object detection on a given video frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            list: YOLO detection results.
        """
        
        results = self.yolo_model(frame)
        return results

    def plot_bboxes_and_depth_estimation(self, results, frame):
        """
        Draw bounding boxes for detected objects and estimate depth for each.

        Args:
            results (list): YOLO detection results.
            frame (np.ndarray): The input video frame.

        Returns:
            tuple: Annotated frame, list of object boxes, and depth map.
        """
        
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

        return frame, object_boxes, depth

    def get_directions(self, frame, object_boxes, depth):
        """
        Analyze object positions and depth to determine navigation instructions.

        Args:
            frame (np.ndarray): Current video frame.
            object_boxes (list): List of tuples with bounding box coords and class names.
            depth (np.ndarray): Depth map for the frame.

        Voice Feedback:
            - Warns about obstacles in front.
            - Suggests going left or right.
            - Tells the user to stop if no direction is safe.
        """
        
        image_height, image_width = frame.shape[:2]
        third_width = image_width // 3
        left_side = third_width
        front_side = 2 * third_width
        right_side = image_width
        
        # cv2.rectangle(frame, (0, 0), (left_side, image_height), (0, 255, 0), 2)
        # cv2.rectangle(frame, (left_side, 0), (front_side, image_height), (0, 255, 255), 2)
        # cv2.rectangle(frame, (front_side, 0), (right_side, image_height), (255, 0, 0), 2)

        left_obstacles = []
        right_obstacles = []
        front_obstacles = []

        for x1, y1, x2, y2, class_name in object_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth_value = depth[center_y, center_x]

            
            if center_y > image_height * 0.80:
                continue

            if center_x < left_side:
                left_obstacles.append((class_name, depth_value))
            elif center_x < front_side:
                front_obstacles.append((class_name, depth_value))
            else:
                right_obstacles.append((class_name, depth_value))


        if any(d <= 1.5 for _, d in front_obstacles):
            left_blocked = any(d <= 1.5 for _, d in left_obstacles)
            right_blocked = any(d <= 1.5 for _, d in right_obstacles)

            if left_blocked and right_blocked:
                self.engine.say(f"{class_name} ahead. No space left or right. Please stop.")
            elif not left_blocked:
                self.engine.say(f"{class_name} ahead. Go around it to the left.")
            elif not right_blocked:
                self.engine.say(f"{class_name} ahead. Go around it to the right.")

        self.engine.runAndWait()      

    def __call__(self):
        """
        Run the main loop for real-time detection, depth estimation,
        and navigation instructions until the user quits.
        """
        
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Could not open video capture."
    
        while True:
            ret, frame = cap.read()
            assert ret, "Error: Could not read frame."
    
            start_time = time()
            results = self.predict(frame)
            frame, object_boxes, depth = self.plot_bboxes_and_depth_estimation(results, frame)
            self.get_directions(frame, object_boxes, depth)
    
            end_time = time()
    
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
            cv2.imshow('YOLO and Depth Estimation', frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()