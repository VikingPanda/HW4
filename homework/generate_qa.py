import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    kart_objects = []
    data = json.load(open(info_path))["detections"][view_index]
    kart_names = json.load(open(info_path))["karts"]
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    for detection in data:
        class_id, track_id, x1, y1, x2, y2 = detection
        if int(class_id) == 1:  # Only consider karts
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            kart_name = kart_names[track_id]

            if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
                continue

            if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
                continue

            center_x = (x1_scaled + x2_scaled) / 2
            center_y = (y1_scaled + y2_scaled) / 2
            kart_objects.append(
                {
                    "instance_id": track_id,
                    "kart_name": kart_name,
                    "center": (center_x, center_y),
                    "is_center_kart": False,  # Update this later after finding the center kart
                }
            )
            for kart in kart_objects:
                kart["is_center_kart"] = (
                    kart["center"] == min(kart_objects, key=lambda k: (k["center"][0] - img_width / 2) ** 2 + (k["center"][1] - img_height / 2) ** 2)["center"]
                )
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    data = json.load(open(info_path))
    track_name = data.get("track", "Unknown Track")
    return track_name
#TODO train demo is the example that is exactly that (balanced_qa_pairs.json)
#evo kart is center of the image, find kart closest to center
#compare cordinates for front and back, and left and right
#filter out small karts less than 20 percent visible, and karts that are out of sight (outside the image boundaries)
#also remove intersection of bounding boxes with the ego kart less than 20 percent visible, and karts that are out of sight (outside the image boundaries)
#dont include 0 answer questions, only include questions with answers that are not 0 (e.g. 0 karts to the left of the ego car should not be included, but 1 kart to the left should be included)
# one json for one view (like the demo_train), with all the questions and answers for that view, and then we can use that to train the model



def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    output = []
    image_file = list(Path(info_path).parent.glob(f"{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"))[0]
    image_file = str(image_file)[8:]
    #print(f"Generating QA pairs for {image_file}...")
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    num_karts = len(kart_objects)
    track_name = extract_track_info(info_path)
    ego_cart = next((kart for kart in kart_objects if kart["is_center_kart"]), None)
    kart_left = [kart for kart in kart_objects if kart["center"][0] < ego_cart["center"][0]]
    kart_right = [kart for kart in kart_objects if kart["center"][0] > ego_cart["center"][0]]
    kart_front = [kart for kart in kart_objects if kart["center"][1] < ego_cart["center"][1]]
    kart_back = [kart for kart in kart_objects if kart["center"][1] > ego_cart["center"][1]]
    q1 = "What kart is the ego car?"
    a1 = ego_cart["kart_name"] if ego_cart else "Unknown"
    output.append({"question": q1, "answer": a1, "image_file": image_file})
    q2 = "How many karts are there in the scenario?"
    a2 = str(num_karts)
    output.append({"question": q2, "answer": a2, "image_file": image_file})
    q3 = "What track is this?"
    a3 = track_name
    output.append({"question": q3, "answer": a3, "image_file": image_file})
    q4 = "Is {kart_name} to the left or right of the ego car?"
    for kart in kart_objects:
        if kart["is_center_kart"]:
            continue
        position = "left" if kart in kart_left else "right"
        q4_kart = q4.replace("{kart_name}", kart["kart_name"])
        output.append({"question": q4_kart, "answer": position, "image_file": image_file})
    q5 = "Is {kart_name} in front of or behind the ego car?"
    for kart in kart_objects:
        if kart["is_center_kart"]:
            continue
        position = "front" if kart in kart_front else "behind"
        q5_kart = q5.replace("{kart_name}", kart["kart_name"])
        output.append({"question": q5_kart, "answer": position, "image_file": image_file})
    q6 = "Where is {kart_name} relative to the ego car?"
    for kart in kart_objects:
        if kart["is_center_kart"]:
            continue
        position = []
        position.append("left" if kart in kart_left else "right")
        position.append("front" if kart in kart_front else "behind")
        q6_kart = q6.replace("{kart_name}", kart["kart_name"])
        output.append({"question": q6_kart, "answer": " and ".join(position), "image_file": image_file})
    q7 = "How many karts are to the left of the ego car?"
    a7 = str(len(kart_left))
    output.append({"question": q7, "answer": a7, "image_file": image_file})
    q8 = "How many karts are to the right of the ego car?"
    a8 = str(len(kart_right))
    output.append({"question": q8, "answer": a8, "image_file": image_file})
    q9 = "How many karts are in front of the ego car?"
    a9 = str(len(kart_front))
    output.append({"question": q9, "answer": a9, "image_file": image_file})
    q10 = "How many karts are behind the ego car?"
    a10 = str(len(kart_back))
    output.append({"question": q10, "answer": a10, "image_file": image_file})

    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    return output


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
def generate_qa_dataset():
    # Loop through all info files and views, generate QA pairs, and save to a new JSON file for training
    print("Generating QA dataset...")
    count = 0
    data_dir = Path("../data/tux_train")
    output_dir = Path("../data/train")
    output_dir.mkdir(exist_ok=True)

    for info_file in data_dir.glob("*_info.json"):
        print(f"Processing {info_file}...")
        for view_index in range(10):  # Assuming 10 views per scenario
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            output_file = output_dir / f"{info_file.stem}_view{view_index}_qa_pairs.json"
            with open(output_file, "w") as f:
                json.dump(qa_pairs, f, indent=4)
            print(count)
            count += 1
    print("QA dataset generation complete.")


def main():
    fire.Fire({"check": check_qa_pairs, "generate_dataset": generate_qa_dataset})



if __name__ == "__main__":
    main()
