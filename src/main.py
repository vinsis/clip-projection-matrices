import torch
from glob import glob
import os

from utils import convert_image_file_to_vector, convert_text_to_vector, create_projection_matrix

POSITIVE_PROMPTS = [
    "A high-quality portrait photo",
    "Good composition",
    "Good lighting",
    "Happy",
    "Cute",
    "Smiling",
    "Beautiful",
    "A person smiling",
    "Face clearly visible",
    "People celebrating",
]

NEGATIVE_PROMPTS = [
    "Bad-quality photo",
    "Blurred photo",
    "Random photo",
    "Sad or angry",
    "Disturbing, scary",
    "Face partially visible",
    "Face covered",
    "Out of focus",
    "Too bright",
    "Too dark",
]

IMAGEFILES = glob("../images/dataset/*.jpg")
PWD = os.path.dirname(__file__)
IMAGE_VECTORS_DICT_FILE = os.path.join(PWD, '..', 'assets', 'image_vectors.pt')

if not os.path.exists(IMAGE_VECTORS_DICT_FILE):
    print("Creating image vectors. This may take a while...")
    image_vectors = torch.stack([convert_image_file_to_vector(imagefile) for imagefile in IMAGEFILES], dim=0)
    dict_to_save = {
        "vectors": image_vectors,
        "filenames": IMAGEFILES
    }
    torch.save(dict_to_save, IMAGE_VECTORS_DICT_FILE)