import argparse

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


"""
parser = argparse.ArgumentParser(description="Yolov8 inference script")
parser.add_argument(
    "--model",
    type=str,
    default="runs/detect/train/yolov8s_100epochs/weights/best.pt",
    help="path to yolo weights"
    )
parser.add_argument(
    "--save",
    action="store_true",
    help="save predictions"
)
"""


def predict(
    source: str,
    model_path="taco_train/yolov8s_100epochs/weights/best.pt",
    save: bool = True,
):
    model = YOLO(model_path)
    results = model.predict(source=source, save=save)
