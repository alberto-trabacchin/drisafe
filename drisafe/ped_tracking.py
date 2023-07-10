import torchvision
from torchvision.utils import draw_bounding_boxes
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8x.pt")
    image_path = "media/rt_camera_sample_1.png"
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