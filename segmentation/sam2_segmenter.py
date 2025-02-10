import os
import shutil
import cv2
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamSegmenter:
    """
    A class to handle image loading, segmentation with SAM (Segment Anything), and saving mask results.
    """

    def __init__(
        self,
        script_dir: str,
        input_image_name: str = "input_image.jpg",        
        sam_checkpoint_name: str = "sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        pred_iou_thresh: float = 0.99, # TODO: adjust parameters
        stability_score_thresh: float = 0.97, # TODO: adjust parameters
        box_nms_thresh: float = 0.3 # TODO: adjust parameters
    ):
        # Initialize paths
        self.script_dir = script_dir
        self.input_image_path = os.path.join(script_dir, "input_images", input_image_name)
        self.output_dir = os.path.join(script_dir, "output_segments")
        self.sam_checkpoint = os.path.join(script_dir, sam_checkpoint_name)
        self.model_type = model_type

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint).to(self.device)

        # Automatic mask generator
        # model: Sam,  # The SAM model instance
        # points_per_side: int | None = 32,  # Number of points per side for the grid
        # points_per_batch: int = 64,  # Number of points processed per batch
        # pred_iou_thresh: float = 0.88,  # IoU threshold to filter masks
        # stability_score_thresh: float = 0.95,  # Stability score threshold for mask quality
        # stability_score_offset: float = 1,  # Offset for stability score computation
        # box_nms_thresh: float = 0.7,  # NMS threshold for filtering overlapping boxes
        # crop_n_layers: int = 0,  # Number of crops per image
        # crop_nms_thresh: float = 0.7,  # NMS threshold for cropping
        # crop_overlap_ratio: float = 512 / 1500,  # Overlapping ratio for cropped regions
        # crop_n_points_downscale_factor: int = 1,  # Downscale factor for crop points
        # point_grids: List[ndarray] | None = None,  # Custom point grids for mask generation
        # min_mask_region_area: int = 0,  # Minimum mask area to keep
        # output_mode: str = "binary_mask"  # Output mode (binary mask or RLE encoding)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh
        )
        
        self.image = None
        self.masks = []

    def load_image(self) -> None:
        """
        Loads the image from the specified path and converts it to RGB.
        """
        image_bgr = cv2.imread(self.input_image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image at {self.input_image_path}")
        self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def generate_masks(self) -> None:
        """
        Generates masks using the SamAutomaticMaskGenerator.
        """
        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() before generate_masks().")
        self.masks = self.mask_generator.generate(self.image)

    def save_segmentation_results(self) -> None:
        """
        Saves the segmented images based on the generated masks, printing mask info.
        """
        if not self.masks:
            raise ValueError("No masks have been generated. Call generate_masks() first.")

        # Clean the output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Print info and save each mask
        # Masks is a list of dictionaries, each containing:
        # segmentation (dict(str, any) or np.ndarray): The mask. If output_mode='binary_mask', is an array of shape HW.
        # bbox (list(float)): The box around the mask, in XYWH format.
        # area (int): The area in pixels of the mask.
        # predicted_iou (float): The model's own prediction of the mask's quality.
        # point_coords (list(list(float))): The point coordinates used to generate this mask.
        # stability_score (float): A measure of the mask's quality.
        # crop_box (list(float)): The crop of the image used to generate the mask, in XYWH format.

        for i, mask_data in enumerate(self.masks):
            print(f"--- Segment {i} ---")
            print(f"Area: {mask_data.get('area')}")
            print(f"BBox: {mask_data.get('bbox')}")
            print(f"Predicted IoU: {mask_data.get('predicted_iou')}")
            print(f"Stability Score: {mask_data.get('stability_score')}")
            print(f"Point Coordinates: {mask_data.get('point_coords')}")
            print("-------------------")

            segmentation = mask_data["segmentation"].astype(np.uint8)

            # Convert image back to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            segment = cv2.bitwise_and(image_bgr, image_bgr, mask=segmentation * 255)

            x, y, w, h = [int(coord) for coord in mask_data["bbox"]]

            cropped_segment = segment[y:y+h, x:x+w]

            output_path = os.path.join(self.output_dir, f"automatic_segment_{i}.png")
            cv2.imwrite(output_path, cropped_segment)
            print(f"Saved: {output_path}\n")

        print("Automatic segmentation completed.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the segmenter instance
    segmenter = SamSegmenter(
        script_dir=script_dir,
        input_image_name="input_image2.jpg",
        pred_iou_thresh=0.99, # TODO: adjust parameters
        stability_score_thresh=0.97, # TODO: adjust parameters
        box_nms_thresh=0.01 # TODO: adjust parameters
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()
    segmenter.save_segmentation_results()


if __name__ == "__main__":
    main()
