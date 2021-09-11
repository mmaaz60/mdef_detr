import os
import json
import argparse

OUTPUT_DIR = ""


def correct_category_ids(file_path):
    print(f"One file {os.path.basename(file_path)}.")
    with open(file_path) as f:
        file_contents = json.load(f)
    annotations = file_contents["annotations"]

    counter = 0
    for ann in annotations:
        category_id = ann["category_id"]
        if category_id != 0:
            ann["category_id"] = 0
            counter += 1
    print(f"Total {counter} annotations with wrong category ids are corrected.")
    with open(f"{OUTPUT_DIR}/{os.path.basename(file_path)}", "w") as f:
        json.dump(file_contents, f)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir_path", required=True,
                    help="Path to the input directory containing json files.")
    ap.add_argument("-o", "--output_dir_path", required=True,
                    help="Path to the output directory for storing the filtered annotations.")
    args = vars(ap.parse_args())

    return args


if __name__ == "__main__":
    args = parse_arguments()
    input_dir_path = args["input_dir_path"]
    output_dir_path = args["output_dir_path"]
    OUTPUT_DIR = output_dir_path
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for file in os.listdir(input_dir_path):
        file_path = f"{input_dir_path}/{file}"
        if file_path.endswith('.json'):
            try:
                correct_category_ids(file_path)
            except:
                pass
