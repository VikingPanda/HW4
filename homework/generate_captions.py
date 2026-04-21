from pathlib import Path
import json
import fire
from matplotlib import pyplot as plt

#from .generate_qa import draw_detections, extract_frame_info
from generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    output = []
    kart_objects = extract_kart_objects(info_path, view_index)
    image_file = list(Path(info_path).parent.glob(f"{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"))[0]
    image_file = str(image_file)[8:]
    num_karts = len(kart_objects)
    track_name = extract_track_info(info_path)
    ego_cart = next((kart for kart in kart_objects if kart["is_center_kart"]), None)

    # Generate captions based on the extracted information
    # 1. Ego car
    c1 = kart_objects[0]["kart_name"] if num_karts > 0 else "No karts detected"
    if (num_karts > 0):
        c1 = f"{c1} is the ego car."
    else:
        c1 = "No ego car detected."

    output.append({"caption": c1, "image_file": image_file})

    # 2. Number of karts
    c2 = str(num_karts)
    c2 = f"There are {c2} karts in the scenario."

    output.append({"caption": c2, "image_file": image_file})

    # 3. Track name
    c3 = track_name
    c3 = f"The track is {c3}."

    output.append({"caption": c3, "image_file": image_file})

    # Additional captions based on relative positions of karts
    kart_left = [kart for kart in kart_objects if kart["center"][0] < ego_cart["center"][0]]
    kart_right = [kart for kart in kart_objects if kart["center"][0] > ego_cart["center"][0]]
    kart_front = [kart for kart in kart_objects if kart["center"][1] < ego_cart["center"][1]]
    kart_back = [kart for kart in kart_objects if kart["center"][1] > ego_cart["center"][1]]

    # 4. Relative position
    c4 = str(len(kart_front))
    c4 = f"{c4} karts are in front of the ego car."
    output.append({"caption": c4, "image_file": image_file})

    c5 = str(len(kart_back))
    c5 = f"{c5} karts are behind the ego car."
    output.append({"caption": c5, "image_file": image_file})
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    return output


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
def generate_captions():
    # Loop through all info files and views, generate captions, and save to a new JSON file for training
    print("Generating captions...")
    data_dir = Path("../data/tux_train")
    output_dir = Path("../data/train")
    output_dir.mkdir(exist_ok=True)

    info_files = list(data_dir.glob("*_info.json"))
    for info_file in info_files:
        for view_index in range(10): # Assuming there are 10 views (0-9) for each info file
            captions = generate_caption(str(info_file), view_index)
            output_file = output_dir / f"{info_file.stem}_view{view_index}_captions.json"
            with open(output_file, "w") as f:
                json.dump(captions, f, indent=4)
            print(f"Generated captions saved to {output_file}")
    print("Caption generation complete.")

def main():
    fire.Fire({"check": check_caption, "generate": generate_captions})


if __name__ == "__main__":
    main()
