import cv2
from doclayout_yolo import YOLOv10
import torchvision
import pathlib

current_dir = pathlib.Path(__file__).parent.absolute()
model_path = str(current_dir) + "/model/doclayout_yolo_dsb_1024.pt"
# print(model_path)
model = YOLOv10(model_path)

# class_names = {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure',
# 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}


def resize_image_cv2(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    THRESH = 1024
    if w > THRESH:
        return img
    # Define new width
    new_width = THRESH
    scale_factor = new_width / w
    new_height = int(h * scale_factor)
    # Resize with interpolation
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_img

def get_circular_layout(image_path, device = 'cpu'):
    det_res = model.predict(
    image_path,         # Image to predict
    imgsz = 1024,       # Prediction image size
    conf = 0.1,         # Confidence threshold
    iou = 0.0001,       # NMS Threshold ??
    device = device,    # Device to use (e.g., 'cuda:0' or 'cpu')
    save = False,       # No need to save annotated image
    verbose = False)
    dets = []
    for entry in det_res:
        bboxes = entry.boxes.xyxy
        classes = entry.boxes.cls
        conf = entry.boxes.conf
        keep = torchvision.ops.nms(bboxes, conf, iou_threshold = 0.1)
        bboxes = bboxes[keep].cpu().numpy()
        classes = classes[keep].cpu().numpy()
        conf = conf[keep].cpu().numpy()
        for i in range(len(bboxes)):
            box = bboxes[i]
            dets.append([classes[i], [int(box[0]), int(box[1]), int(box[2]), int(box[3])]])
    return dets
