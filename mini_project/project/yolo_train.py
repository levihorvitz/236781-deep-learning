import torch
import yaml

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    PoseModel,
    SegmentationModel,
)


class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
        }

from ultralytics.utils.plotting import plot_results
"""
parser = argparse.ArgumentParser(description="Yolov8 training and validation script")
parser.add_argument(
    "--model_name",
    default="yolov8s",
    help="Name you'd like to give your model"
)
parser.add_argument(
    "--epochs",
    default=100,
    help="number of epochs to train the model"
    )
"""


def train_model(model_name: str = "yolov8s", epochs: int = 100) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(f"{model_name}.pt").to(device=device)

    # Use the model
    model.train(
        data="project/TACO.yaml",
        epochs=epochs,
        patience=25,
        imgsz=640,
        name=f"{model_name}_{epochs}epochs",
        pretrained=True,
        optimizer="Adam",
        verbose=True
    )

    metrics = model.val()


def show_model(model_name: str = "yolov8s", epochs: int = 10) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(f"{model_name}.pt").to(device=device)
    print("Layer (type)\t\t\t\tOutput Shape\t\t\tParam #")
    print("-" * 80)
    total_params = 0
    i = 0
    for name, param in model.model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if i <= 20 or i>150:
            if param.dim() == 1:
                print(f"{name:<40}{param.size()}\t\t{num_params}")
            else:
                print(f"{name:<40}{list(param.size())}\t\t\t{num_params}")
        i += 1
    print("-" * 80)

    with open(
        f"runs/detect/train/{model_name}_{epochs}epochs/args.yaml", "r"
    ) as stream:
        args = yaml.safe_load(stream=stream)

    print(f"Total params: {total_params}")
    print(f"Epochs: {args['epochs']}")
    print(f"Batch size: {args['batch']}")
    print(f"Optimizer: {args['optimizer']}")
    print(f"Cosine learning rate: {args['cos_lr']}")
    print(f"Dropout: {args['dropout']}")
    print(f"IoU: {args['iou']}")
    print(f"Augment: {args['augment']}")
    print(f"Learning rate 0: {args['lr0']}")
    print(f"Learning rate f: {args['lrf']}")
