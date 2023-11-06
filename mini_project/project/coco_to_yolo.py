import requests
import json
import os


def download_img(image_url, image_path):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    data = requests.get(image_url).content
    with open(image_path, "wb") as img_file:
        img_file.write(data)


def get_img_anns(dataset, image_id):
    anns = []
    for anno in dataset["annotations"]:
        if anno["image_id"] == image_id:
            anns.append(anno)
    return anns


def coco_to_yolo(annotations_json_path, folder_root_path, folder_name):
    with open(annotations_json_path, "r") as f:
        dataset = json.loads(f.read())

    count = 0
    for img in dataset["images"]:
        image_path = f"{folder_root_path}/{folder_name}/images/img_{count}.jpg"

        img_id = img["id"]
        img_w = img["width"]
        img_h = img["height"]
        img_ann = get_img_anns(dataset, img_id)
        if len(img_ann) == 0:
            return

        download_img(img["flickr_url"], image_path)

        text_file_path = f"{folder_root_path}/{folder_name}/labels/img_{count}.txt"
        os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
        text_file = open(text_file_path, "w+")

        for ann in img_ann:
            current_category = ann["category_id"]

            current_bbox = ann["bbox"]
            x = current_bbox[0]
            y = current_bbox[1]
            w = current_bbox[2]
            h = current_bbox[3]

            # Finding midpoints
            x_centre = (x + (x + w)) / 2
            y_centre = (y + (y + h)) / 2

            # Normalization
            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h

            # Limiting upto fix number of decimal places
            x_centre = format(x_centre, ".6f")
            y_centre = format(y_centre, ".6f")
            w = format(w, ".6f")
            h = format(h, ".6f")

            # Writing current object
            text_file.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

        text_file.close()
        count += 1
