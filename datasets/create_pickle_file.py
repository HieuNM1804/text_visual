import os
import pickle
import numpy as np
from glob import glob
from vectorization import svg_to_vector_sequence

def build_coordinate_pickle(dataset_root, dataset_name):
    root = os.path.join(dataset_root, dataset_name)
    sketch_root = os.path.join(root, "sketch")
    svg_root = os.path.join(root, "sketch_svg")

    coordinate = {}

    for label in os.listdir(sketch_root):
        sketch_label_dir = os.path.join(sketch_root, label)
        svg_label_dir = os.path.join(svg_root, label)

        if not os.path.isdir(sketch_label_dir):
            continue

        for img_path in glob(os.path.join(sketch_label_dir, "*")):
            if not os.path.isfile(img_path):
                print("Dose not exist: ", img_path)
                continue

            name = os.path.splitext(os.path.basename(img_path))[0]
            svg_path = os.path.join(svg_label_dir, name + ".svg")

            if not os.path.exists(svg_path):
                print("Does not exist: ", svg_path)
                continue

            try:
                vector = svg_to_vector_sequence(svg_path)
                vector = vector.astype(np.float16)
            except Exception as e:
                print(f"Skip corrupted SVG: {svg_path}")
                continue

            # relative path của ảnh sketch
            rel_path = os.path.relpath(img_path, root).replace("\\", "/")
            rel_path = str(rel_path)
            coordinate[rel_path] = vector

    save_path = os.path.join(root, "sketchy_vectorization")
    with open(save_path, "wb") as f:
        pickle.dump(coordinate, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", save_path)
    print("Total samples:", len(coordinate))

dataset_root = "D:/Research/VLM_project/dataset"
dataset_name = "Sketchy"
build_coordinate_pickle(dataset_root, dataset_name)