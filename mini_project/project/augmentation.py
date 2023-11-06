import albumentations as A
import random
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from PIL import Image

def random_augment_images(images_path:str,images_num:int):
    images = os.listdir(images_path)
    random.shuffle(images)
    random_images = images[:images_num]
    return [f"{images_path}/{random_image}" for random_image in random_images]


def augment_sample(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = augmentation_generator(image)
    transformed_image = transformed["image"]
    #print(f"type(transformed_image):{type(transformed_image)}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(transformed_image)
    ax[1].set_title('Augmented Image')
    ax[1].axis('off')

    plt.show()


def create_augmented_sample(image_path:str):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed_image = augmentation_generator(image)
    transformed_image = transformed_image["image"]

    image_id = len(os.listdir(os.path.dirname(image_path)))
    image_path_dest = f"{os.path.dirname(image_path)}/img_{image_id}.jpg"

    label_path = image_path.replace("images","lables").replace("jpg","txt")
    label_path_dest = f"{os.path.dirname(label_path)}/img_{image_id}.txt"

    image_file= Image.fromarray(transformed_image)
    image_file.save(image_path_dest)

    #with open(image_path_dest,"w+") as image_file:
    #    image_file.write(transformed_image)

    shutil.copyfile(label_path, label_path_dest)



def augmentation_generator(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Cutout(num_holes=8, max_h_size=32,
                 max_w_size=32, fill_value=0, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.50, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
        ], p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.ToGray(p=0.1),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=1)
    return transform(image=image)
