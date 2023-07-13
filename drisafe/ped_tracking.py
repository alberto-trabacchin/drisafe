import torchvision
from torchvision.utils import draw_bounding_boxes
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2
from drisafe.constants import SENSORS
from PIL import Image

def eval():
    xres = 1080
    yres = 1920
    xs = 4.62e-3
    ys = 6.16e-3
    mx = xres / xs
    my = yres / ys
    d0 = np.arange(10)
    h = 1.8
    dh = np.arange(0, 0.5, 0.1)

    f = 5.4e-3
    dhp_h = f / d0 * dh * my
    print(f"{dhp_h:.0f} pixels (people)")


def detect():
    model = YOLO("yolov8x.pt")
    image_path = "media/rt_camera_sample_2.png"
    image = torchvision.io.read_image(image_path)
    transform = torchvision.transforms.Lambda(lambda x: x[:3])
    image = transform(image)
    results = model(image_path)
    boxes = results[0].boxes.data
    labels = results[0].boxes.cls
    scores = results[0].boxes.conf
    ped_boxes = []
    ped_scores = []
    for b, l, s in zip(boxes, labels, scores):
        if l == 0 and s > 0.5:
            ped_boxes.append(b[0:4])
            ped_scores.append(str(s.item())[0:4])
            print(f"Pedestrians score: {s:.3f}.")
    boxes_image = draw_bounding_boxes(image, torch.stack(ped_boxes),
                                      labels = ped_scores,
                                      font = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
                                      font_size = 20,
                                      colors = "red",
                                      width = 2)
    plt.imshow(boxes_image.permute(1, 2, 0))
    plt.show()

def track():
    sensor = SENSORS[5]
    vid_path = sensor["roof_cam"]["vid_path"]
    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(str(vid_path))
    ped_boxes_all_rec = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist = True, verbose = False, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        labels = results[0].boxes.cls.cpu()
        scores = results[0].boxes.conf.cpu().numpy()
        ped_boxes = []
        ped_scores = []
        ped_ids = []
        for b, l, s, id in zip(boxes, labels, scores, ids):
            if l == 0 and s > 0.5:
                ped_boxes.append(b[0:4])
                ped_scores.append(str(s)[0:4])
                ped_ids.append(id)
                print(f"Pedestrians score: {s:.3f}.")
        ped_boxes_all_rec.append(ped_boxes)
        for box, id in zip(ped_boxes, ped_ids):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"id: {id}",
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 255),
                2,
            )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    

def estim_depth():
    image_path = "media/rt_camera_sample_2.png"
    midas = torch.hub.load('intel-isl/MiDaS', "DPT_Large")
    midas.to('cuda')
    midas.eval()
    transformss = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transformss.small_transform
    cap = cv2.imread(image_path)

    image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

    image_batch = transform(image).to('cuda')

    with torch.no_grad():
            prediction = midas(image_batch)
            prediction=torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size = image.shape[:2],
                        mode = 'bicubic',
                        align_corners=False
                        ).squeeze()

            output = prediction.cpu().numpy()
    p, axarr = plt.subplots(2, 1)
    axarr[0].imshow(output)
    axarr[1].imshow(image)
    plt.show()

if __name__ == "__main__":
    #estim_depth()
    track()