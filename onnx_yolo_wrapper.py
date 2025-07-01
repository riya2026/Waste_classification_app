import onnxruntime as ort
import cv2
import numpy as np

class ONNXResult:
    def __init__(self, raw_output):
        self.raw_output = raw_output

    def __call__(self):
        return self.raw_output

    def show(self):
        print("You can add visualization here")

    def plot(self):
        return None  # optional image with boxes

    def save(self, filename):
        pass  # optional

class YOLO:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def __call__(self, image_path):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :] / 255.0
        img_input = img_input.astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: img_input})
        return ONNXResult(outputs)
