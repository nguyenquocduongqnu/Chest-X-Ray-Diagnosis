import os
import cv2
import torch
import random
from numpy import random
from flask import Markup
from utils.plots import plot_one_box
from models.experimental import attempt_load
from flask import Flask, render_template, request
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

# Create Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Load model
weights = 'models/best.pt'
set_logging()
device = select_device('')
half = device.type != 'cpu'
imgsz = 512

# Load model
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)
if half:
    model.half()

desc_file = "description.csv"
f = open(desc_file,  encoding="utf8")
desc = f.readlines()
f.close()
dict = {}
for line in desc:
    dict[line.split('|')[0]] = line.split('|')[1]

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
         try:
            image = request.files['file']
            if image:
                # Save file
                print(image.filename)
                print(app.config['UPLOAD_FOLDER'])
                source = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", source)
                image.save(source)
                save_img = True
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                conf_thres = 0.15
                iou_thres = 0.5

                for path, img, im0s, vid_cap,s in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

                    extra = ""
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                        save_path = source
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_img:  # Add bbox to image
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                                    extra += "<br>- <b>" + str(names[int(cls)]) + "</b> with <b>{:.2f}% confidence</b>".format(conf)

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)

                if extra == "":
                    extra = "Normal Healthy Lungs"

                return render_template("index.html", user_image = image.filename, rand = random.random(), msg="File upload successful", extra=Markup(extra), names = names, idBoolean = True)
              
            else:
                return render_template('index.html', msg='Please select a file to upload')

         except Exception as ex:
            print(ex)
            return render_template('index.html', msg='Unable to recognize image')

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)