import os
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort

# Assuming this function is part of a script to test DeepSort

def test_deep_sort_output():
    """Function to test the output of DeepSort.update()"""
    # Example input (replace with actual bounding box and confidence inputs)
    bbox_xywh = [[100, 200, 50, 100], [300, 400, 60, 80]]  # Format: x_center, y_center, width, height
    confidences = [0.9, 0.85]
    classes = [0, 1]  # Class IDs
    ori_img = None  # Placeholder for original image, use a dummy image for testing if needed

    # Initialize DeepSort with a valid model checkpoint path (replace with actual path)
    model_path = '/home/ganesh/A1/src/deep_sort_tracker/models/ckpt.t7'  # Adjust path as necessary
    deepsort = DeepSort(
        model_path=model_path,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100
    )

    # Call update and capture the output
    try:
        tracked_objects = deepsort.update(
            bbox_xywh=bbox_xywh,
            confidences=confidences,
            classes=classes,
            ori_img=ori_img
        )
        # Log the output structure
        print("Output from DeepSort.update():")
        for obj in tracked_objects:
            print(obj)  # Print each tracked object (list or array)

    except Exception as e:
        print(f"Error while testing DeepSort: {e}")

# Run the test function
test_deep_sort_output()
