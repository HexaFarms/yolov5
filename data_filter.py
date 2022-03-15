import numpy as np
import pandas as pd
from pathlib import Path
import os
from loguru import logger

""" Filter YOLO Dataset to use only relevant class """

INPUT_LABEL_DIR = "/media/hexaburbach/onetb/yolo_data/coco/labels/train2017"
OUTPUT_LABEL_DIR = "/media/hexaburbach/onetb/yolo_data/coco/labels/filtered_train"
INPUT_IMG_DIR = "/media/hexaburbach/onetb/yolo_data/coco/images/train2017"
OUTPUT_IMG_DIR = "/media/hexaburbach/onetb/yolo_data/coco/images/filtered_train"
INTEREST_CLASS = [0, 58, 63, 67]  # Interested class in the original dataset
MAPPING_CLASS = {0: 1, 58: 0, 63: 2, 67: 3}  # {index of original dataset : index of new dataset}

if __name__ == "__main__":

    logger.info("Start Data Filtering.")

    path_label = Path(INPUT_LABEL_DIR)
    labels = list(path_label.glob("*.txt"))
    save_loc_label = Path(OUTPUT_LABEL_DIR)

    path_img = Path(INPUT_IMG_DIR)
    save_loc_img = Path(OUTPUT_IMG_DIR)
    used_files = labels.copy()

    classes = INTEREST_CLASS  # person, potted plant, laptop, cell phone
    new_class_idx = MAPPING_CLASS  # map to the new labels

    logger.info(f"{len(labels)} images will be processed.")
    """ Leave only relevant classes"""
    for file in labels:

        text = np.genfromtxt(file, delimiter=" ")

        if len(text.shape) == 1:  # one object only, then keep the format.
            text = np.expand_dims(text, axis=0)

        if not bool(set(text[:, 0]) & set(classes)):
            used_files.remove(file)
            logger.info(f"{file} has no relevant classes. So, this image will be deleted.")
            continue

        text = np.delete(text, np.argwhere(np.isin(text[:, 0], classes, invert=True)).ravel(), axis=0)
        text[:, 0] = np.vectorize(new_class_idx.get)(text[:, 0])  # change to the new lebeling number.

        """ Convert to pandas to convert the first column to integer"""
        df = pd.DataFrame(data=text).astype('float16')
        df[0] = df[0].astype('int8')
        df.to_csv(os.path.join(save_loc_label, file.name), index=False, header=False, sep=' ')

    logger.info(f"Now relevant images are moved.")
    for used_file in used_files:
        try:
            # if the input format is different, should be changed.
            img_name = path_img.joinpath(used_file.stem).with_suffix('.jpg')
            img_name.rename(os.path.join(save_loc_img, img_name.name))
        except:
            logger.error(f"{used_file} is not found in the image folder.")

    logger.info(f"{len(used_files)} images are successfully processed.")
