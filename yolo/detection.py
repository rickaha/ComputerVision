import cv2
import numpy as np
import onnxruntime

class YOLOv11ONNX:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45):
        self.session = onnxruntime.InferenceSession(model_path)
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def non_max_suppression(self, boxes, scores):
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        return indices.flatten() if len(indices) > 0 else []

    def detect(self, image):
        # Resize image to 640x640 for YOLO input
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (self.input_width, self.input_height), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})

        predictions = np.squeeze(outputs[0]).T
        boxes = predictions[:, :4]
        scores = np.max(predictions[:, 4:], axis=1)
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Scale boxes to original image dimensions
        img_height, img_width = image.shape[:2]
        xyxy_boxes = self.xywh2xyxy(boxes)
        xyxy_boxes[:, 0] *= (img_width / self.input_width)
        xyxy_boxes[:, 1] *= (img_height / self.input_height)
        xyxy_boxes[:, 2] *= (img_width / self.input_width)
        xyxy_boxes[:, 3] *= (img_height / self.input_height)

        indices = self.non_max_suppression(xyxy_boxes, scores)

        detections = []
        for i in indices:
            box = xyxy_boxes[i]
            detections.append({
                'class_id': int(class_ids[i]),
                'confidence': float(scores[i]),
                'box': box.astype(np.int32)
            })

        return detections

def draw_detections(image, detections, class_names):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_id = det['class_id']
        confidence = det['confidence']
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    model_path = "/path_to_file/yolo11n.onnx"
    with open('coco-labels-2014_2017.txt') as file:  # loading the classes names from the file.
        content = file.read()
        class_names = content.split('\n')

    yolo = YOLOv11ONNX(model_path)
#    cap =cv2.VideoCapture("name.mp4")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

#        image_path = input("Enter the path to an image or type 'exit' to quit: ")
#        frame = cv2.imread(image_path)

        detections = yolo.detect(frame)
        frame = draw_detections(frame, detections, class_names)
        cv2.imshow("YOLOv11 ONNX Detection (High Res)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
