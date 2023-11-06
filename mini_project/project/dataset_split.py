import os.path
import json
import numpy as np
import random
import datetime as dt
import copy


def random_split_dataset(
    annotations_json_path: str,
    picked_categories_ids: list,
    test_percentage=10,
    val_percentage=10,
) -> None:
    train_percentage = 100 - test_percentage - val_percentage
    # Load annotations
    with open(annotations_json_path, "r") as f:
        dataset = json.loads(f.read())

    anns = [
        anno
        for anno in dataset["annotations"]
        if anno["category_id"] in picked_categories_ids
    ]
    scene_anns = dataset["scene_annotations"]
    imgs = dataset["images"]

    nr_anns = len(anns)
    nr_testing_anns = int(nr_anns * test_percentage * 0.01 + 0.5)
    nr_nontraining_anns = int(nr_anns * (test_percentage + val_percentage) * 0.01 + 0.5)
    # for i in range(nr_trials):
    random.shuffle(anns)
    # Add new datasets
    train_set = {
        "info": None,
        "images": [],
        "annotations": [],
        "scene_annotations": [],
        "licenses": [],
        "categories": [],
        "scene_categories": [],
    }
    train_set["info"] = dataset["info"]
    train_set["categories"] = dataset["categories"]
    train_set["scene_categories"] = dataset["scene_categories"]

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    test_set["annotations"] = anns[0:nr_testing_anns]
    val_set["annotations"] = anns[nr_testing_anns:nr_nontraining_anns]
    train_set["annotations"] = anns[nr_nontraining_anns:nr_anns]

    print("Random Dataset splitting configuration:")
    print(
        f"TEST SET: {test_percentage}% Annotations Number:{len(test_set['annotations'])}"
    )
    print(
        f"VALIDATION SET: {test_percentage}% Annotations Number:{len( val_set['annotations'])}"
    )
    print(
        f"TRAIN SET: {train_percentage}% Annotations Number:{len( train_set['annotations'])}"
    )

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [], [], []
    i=0
    for anno in test_set["annotations"]:
        if i<=20:
            print(anno)
            i+=1
        test_img_ids.append(anno["image_id"])
    print("...")
    for anno in val_set["annotations"]:
        val_img_ids.append(anno["image_id"])

    for anno in train_set["annotations"]:
        train_img_ids.append(anno["image_id"])

    # Split Images

    for img in imgs:
        if img["id"] in test_img_ids:
            test_set["images"].append(img)
        elif img["id"] in val_img_ids:
            val_set["images"].append(img)
        elif img["id"] in train_img_ids:
            train_set["images"].append(img)

    # Split scene tags
    for ann in scene_anns:
        if ann["image_id"] in test_img_ids:
            test_set["scene_annotations"].append(ann)
        elif ann["image_id"] in val_img_ids:
            val_set["scene_annotations"].append(ann)
        elif ann["image_id"] in train_img_ids:
            train_set["scene_annotations"].append(ann)

    dataset_dir = "/".join(annotations_json_path.split("/")[:-1])
    # Write dataset splits
    ann_train_out_path = dataset_dir + "/" + "annotations" + "_train.json"
    ann_val_out_path = dataset_dir + "/" + "annotations" + "_val.json"
    ann_test_out_path = dataset_dir + "/" + "annotations" + "_test.json"

    with open(ann_train_out_path, "w+") as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, "w+") as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, "w+") as f:
        f.write(json.dumps(test_set))
