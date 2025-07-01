import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image

class ONNXResult:
    def __init__(self, raw_output, class_names, original_image):
        self.raw_output = raw_output
        self.names = class_names
        self.original_image = original_image
        self.boxes = self._process_boxes()

    def _process_boxes(self, conf_thres=0.25):
        boxes = []
        raw = self.raw_output[0][0]  # shape: [N, 85]
        cls_list = []
        conf_list = []

        for pred in raw:
            if pred[4] > conf_thres:
                class_id = np.argmax(pred[5:])
                conf = pred[4] * pred[5 + class_id]
                cx, cy, w, h = pred[:4]
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                boxes.append([x1, y1, x2, y2, conf, class_id])
                cls_list.append(class_id)
                conf_list.append(conf)

        return type("Boxes", (), {"cls": cls_list, "conf": conf_list, "raw": boxes})

    def plot(self):
        img = self.original_image.copy()
        for x1, y1, x2, y2, conf, class_id in self.boxes.raw:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.names[class_id]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img


class YOLO:
    def __init__(self, model_path, class_names):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.class_names = class_names

    def __call__(self, image_input):  # now accepts array or PIL.Image
        if isinstance(image_input, str):  # if path
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):  # PIL image
            img = np.array(image_input.convert("RGB"))[..., ::-1]  # Convert to BGR for OpenCV
        elif isinstance(image_input, np.ndarray):  # already NumPy
            img = image_input
        else:
            raise ValueError("Unsupported image input type")

        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :] / 255.0
        img_input = img_input.astype(np.float32)

        outputs = self.session.run(None, {self.input_name: img_input})
        return ONNXResult(outputs, self.class_names, img_resized)
