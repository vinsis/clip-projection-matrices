import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def convert_image_to_vector(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).squeeze()
    return image_features

def convert_image_file_to_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    return convert_image_to_vector(image)

def convert_text_to_vector(text):
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features

def create_projection_matrix(vectors):
    '''
    `vectors.shape` is assumed to be (vector_dim, num_vectors)
    '''
    # Note that P = A(A^TA)^-1A^T where a set of vectors `A`
    return vectors @ ( torch.linalg.inv(vectors.t() @ vectors) ) @ vectors.t()

def resize_image(image, new_w):
    w,h = image.size
    new_h = new_w * h // w
    return image.resize((new_w, new_h))

def stitch_images(pil_images, width=300):
    return Image.fromarray(np.concatenate([np.array(resize_image(image, width)) for image in pil_images], axis=1))

