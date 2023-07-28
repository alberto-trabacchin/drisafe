import torchvision
from torchvision.utils import draw_bounding_boxes
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2
from drisafe.config.paths import TRACKING_DATA_PATH
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
import json

class Person(object):

    def __init__(self, id):
        self.data = {
            "id": id,
            "appeared": [],
            "observed": [],
            "boxes": [],
            "score": [],
            "position": [],
            "rt_gaze": []
        }

    def update(self, gaze_crds, box, score, frame_n, rec_id):
        box = box.reshape(4).tolist()
        gaze_crds = gaze_crds.reshape(2).tolist()
        xb1, yb1, xb2, yb2 = box[0], box[1], box[2], box[3]
        xg, yg = gaze_crds[0], gaze_crds[1]
        self.data["appeared"].append(frame_n)
        self.data["boxes"].append(box)
        self.data["score"].append(float(score))
        xp = (xb2 + xb1) / 2
        yp = (yb2 + yb1) / 2
        self.data["position"].append([xp, yp])
        if (xg > xb1 and xg < xb2 and yg > yb1 and yg < yb2):
            self.data["observed"].append(True)
        else:
            self.data["observed"].append(False)
        self.data["rt_gaze"].append(gaze_crds)
        print(f"({rec_id}, {frame_n}) - Pedestrian score: {score:.3f} - {self.data['observed'][-1]}")

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


def update_track_data(people_list, ids, boxes, scores, gaze_crds, frame_n, rec_id):
    for id, box, score in zip(ids, boxes, scores):
        det_people_ids = [p.data["id"] for p in people_list]
        if not id in det_people_ids:
            pers = Person(id)
            pers.update(gaze_crds, box, score, frame_n, rec_id)
            people_list.append(pers)
        else:
            pos = det_people_ids.index(id)
            people_list[pos].update(gaze_crds, box, score, frame_n, rec_id)

def get_people_data(results, rec_id, frame_n):
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
            ped_scores.append(s)
            ped_ids.append(id)
            #print(f"({rec_id}, {frame_n}) - Pedestrian score: {s:.3f}.")
    return ped_boxes, ped_scores, ped_ids

def draw_bbox(frame, box, id):
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

def track_people(rec_id):
    sstream = SensorStreams(SENSORS[rec_id - 1], rec_id)
    model = YOLO("yolov8x.pt")
    people_list = []
    print(f"Computing recording {rec_id}...")
    while True:
        sstream.read()
        frame = sstream.rt_frame
        results = model.track(frame, persist = True, verbose = False, tracker = "bytetrack.yaml")
        if results[0].boxes.id != None:
            frame_n = sstream.t_step
            ped_boxes, ped_scores, ped_ids = get_people_data(results, rec_id, frame_n)
            update_track_data(people_list, ped_ids, ped_boxes, ped_scores, 
                              sstream.rt_crd, frame_n, rec_id)
            for box, id in zip(ped_boxes, ped_ids):
                draw_bbox(frame, box, id)
        cv2.imshow("frame", frame)
        if (cv2.waitKey(1) & 0xFF == ord("q")): break
        if not sstream.online: break
    print([p.data for p in people_list])
    print(f"Recoring {rec_id} ended.")
    return people_list

def write_track_data(people_data, id):
    list_data = [p.data for p in people_data]
    print(list_data)
    data_path = TRACKING_DATA_PATH[id - 1]
    json_data = json.dumps(list_data, indent = 4, default = int)
    with open(data_path, "w") as outfile:
        outfile.write(json_data)

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
    rec_ids = [4, 6, 7, 10, 11, 12, 13, 16, 18, 19, 26, 27, 35, 38, 39, 40, 47, 51, 53, 58, 60, 61, 64, 65, 70, 72]
    rec_ids = [i for i in range(1, 75)]
    for id in rec_ids:
        people_data_rec = track_people(rec_id = id)
        write_track_data(people_data_rec, id)