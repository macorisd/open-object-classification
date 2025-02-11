import os
import shutil
import cv2
import torch
import numpy as np
import time
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamSegmenter:
    """
    A class to handle image loading, segmentation with SAM (Segment Anything), and saving mask results.

    Attributes:        
        input_image_name (str): Name of the input image file.
        sam_checkpoint_name (str): Name of the SAM model checkpoint file.
        model_type (str): Type of the SAM model.
        pred_iou_thresh (float): IoU threshold for filtering masks.
        stability_score_thresh (float): Stability score threshold for mask quality.
        box_nms_thresh (float): NMS threshold for filtering overlapping boxes.
        reduce_masks (bool): Whether to reduce the number of generated masks.
        MAX_MASKS (int): Maximum number of masks to keep.
        segment_min_area (int): Minimum mask area to keep.
        segment_max_area_ratio (float): Maximum mask area ratio to keep.
        timeout (int): Timeout in seconds for mask generation.
        device (torch.device): Device to use for the SAM model.
        sam (torch.nn.Module): SAM model instance.
        mask_generator (SamAutomaticMaskGenerator): SAM automatic mask generator instance.
        image (np.ndarray): Input image as a NumPy array.
        masks (list[dict]): List of dictionaries containing segmentation mask data.
        prev_params (dict): Previous SAM mask generator parameters for backtracking.
    """

    def __init__(
        self,        
        input_image_name: str = "input_image.jpg",        
        sam_checkpoint_name: str = "sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.8,
        box_nms_thresh: float = 0.3,
        reduce_masks: bool = True,
        MAX_MASKS: int = 10,
        segment_min_area: int = 20000,
        segment_max_area_ratio: float = 0.4,
        timeout: int = 120
    ):
        """
        Initializes the SamSegmenter instance.        
        """
        # Initialize attributes
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_image_path = os.path.join(self.script_dir, "input_images", input_image_name)
        self.output_dir = os.path.join(self.script_dir, "output_segments")
        self.sam_checkpoint = os.path.join(self.script_dir, sam_checkpoint_name)
        self.model_type = model_type
        self.reduce_masks = reduce_masks
        self.MAX_MASKS = MAX_MASKS
        self.segment_min_area = segment_min_area
        self.segment_max_area_ratio = segment_max_area_ratio
        self.timeout = timeout

        

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint).to(self.device)

        # SAM automatic mask generator parameters
            # model: Sam,                               # The SAM model instance
            # points_per_side: int | None = 32,         # Number of points per side for the grid
            # points_per_batch: int = 64,               # Number of points processed per batch
            # pred_iou_thresh: float = 0.88,            # IoU threshold to filter masks
            # stability_score_thresh: float = 0.95,     # Stability score threshold for mask quality
            # stability_score_offset: float = 1,        # Offset for stability score computation
            # box_nms_thresh: float = 0.7,              # NMS threshold for filtering overlapping boxes
            # crop_n_layers: int = 0,                   # Number of crops per image
            # crop_nms_thresh: float = 0.7,             # NMS threshold for cropping
            # crop_overlap_ratio: float = 512 / 1500,   # Overlapping ratio for cropped regions
            # crop_n_points_downscale_factor: int = 1,  # Downscale factor for crop points
            # point_grids: List[ndarray] | None = None, # Custom point grids for mask generation
            # min_mask_region_area: int = 0,            # Minimum mask area to keep
            # output_mode: str = "binary_mask"          # Output mode (binary mask or RLE encoding)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh
        )
        
        self.image = None
        self.masks = []
        self.prev_params = None

    def load_image(self) -> None:
        """
        Loads the image from the specified path and converts it to RGB.
        """
        image_bgr = cv2.imread(self.input_image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image at {self.input_image_path}")
        self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def segment_filtering(self) -> None:
        """
        Filters out masks based on certain properties.
        """
        self.min_area_filtering()
        self.max_area_filtering()

    def min_area_filtering(self) -> None:
        """
        Filters out masks with an area less than the minimum area threshold.
        """
        self.masks = [mask for mask in self.masks if mask["area"] >= self.segment_min_area]

    def max_area_filtering(self) -> None:
        """
        Filters out masks whose area is greater than the input image area multiplied by area_ratio.
        """
        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() before max_area_filtering().")

        image_area = self.image.shape[0] * self.image.shape[1]
        threshold = self.segment_max_area_ratio * image_area

        self.masks = [mask for mask in self.masks if mask["area"] <= threshold]

    def adjust_mask_generator_params(self, backtrack: bool = False) -> None:
        """
        Adjusts the SAM mask generator parameters to reduce the number of generated masks.

        Args:
            backtrack (bool): Whether to backtrack the adjustments.
        """
        # Attempt to reduce the number of masks by adjusting the parameters
        if not backtrack:
            # Save the current parameters for backtracking
            self.prev_params = {
                "pred_iou_thresh": self.mask_generator.pred_iou_thresh,
                "stability_score_thresh": self.mask_generator.stability_score_thresh,
                "box_nms_thresh": self.mask_generator.box_nms_thresh
            }
            
            # Adjust the parameters
            self.mask_generator.pred_iou_thresh = min(0.99, self.mask_generator.pred_iou_thresh + 0.02)
            self.mask_generator.stability_score_thresh = min(0.99, self.mask_generator.stability_score_thresh + 0.02)
            self.mask_generator.box_nms_thresh = max(0.001, self.mask_generator.box_nms_thresh - 0.06)
        else:
            # Backtrack the adjustments            
            if self.prev_params is not None:
                self.mask_generator.pred_iou_thresh = self.prev_params["pred_iou_thresh"]
                self.mask_generator.stability_score_thresh = self.prev_params["stability_score_thresh"]
                self.mask_generator.box_nms_thresh = self.prev_params["box_nms_thresh"]

                self.prev_params = None
            else:
                print("No previous parameters for backtracking.\n")

    def generate_masks(self) -> None:
        """
        Generates masks using the SamAutomaticMaskGenerator.
        """
        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() before generate_masks().")
        
        # Generate masks and apply filtering
        self.masks = self.mask_generator.generate(self.image)
        self.segment_filtering()
        print(f"Number of masks: {len(self.masks)}")
        
        # Adjust the mask generator parameters to reduce the number of masks if necessary
        if (self.reduce_masks and len(self.masks) > self.MAX_MASKS):
            print(f"More than {self.MAX_MASKS} masks generated. Reducing masks...")

            start_time = time.time()
            while time.time() - start_time < self.timeout:
                # Adjust the mask generator parameters
                self.adjust_mask_generator_params()

                # Generate masks and apply filtering
                self.masks = self.mask_generator.generate(self.image)
                self.segment_filtering()
                
                # If the number of masks is reduced to the MAX_MASKS or less, break the loop
                if (len(self.masks) <= self.MAX_MASKS):
                    # If no masks are generated, backtrack the adjustments
                    if not self.masks:
                        self.adjust_mask_generator_params(backtrack=True)
                        self.masks = self.mask_generator.generate(self.image)
                        self.segment_filtering()
                        print(f"Number of masks: {len(self.masks)}. Tried to reduce the number of masks to {self.MAX_MASKS} or less, but no masks were generated.\n")
                    
                    else:
                        print(f"Number of masks reduced to {len(self.masks)}.\n")
                    
                    break

                else:
                    print(f"Number of masks reduced to {len(self.masks)}. Reducing to {self.MAX_MASKS} (or less) masks...")
            else:
                print(f"Number of masks ({len(self.masks)}) could not be reduced to {self.MAX_MASKS} or less within the timeout period.\n")

    def save_segmentation_results(self) -> list:
        """
        Saves the segmented images based on the generated masks to disk and returns a list of the cropped segments.

        Returns:
            list: A list containing the cropped segment images as NumPy arrays.
        """
        if not self.masks:
            print("No masks were generated. Call generate_masks() before save_segmentation_results(), and if you've already done so, check the mask generator parameters.")
            return []

        # Clean the output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        segments_list = []  # List to store the cropped segments

        
        # Masks is a list of dictionaries, each containing:
            # segmentation (dict(str, any) or np.ndarray):  The mask. If output_mode='binary_mask', is an array of shape HW.
            # bbox (list(float)):                           The box around the mask, in XYWH format.
            # area (int):                                   The area in pixels of the mask.
            # predicted_iou (float):                        The model's own prediction of the mask's quality.
            # point_coords (list(list(float))):             The point coordinates used to generate this mask.
            # stability_score (float):                      A measure of the mask's quality.
            # crop_box (list(float)):                       The crop of the image used to generate the mask, in XYWH format.


        # Print info and save each mask
        for i, mask_data in enumerate(self.masks):
            print(f"--- Segment {i+1} ---")
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

            # Crop the segment based on the bounding box
            x, y, w, h = [int(coord) for coord in mask_data["bbox"]]
            cropped_segment = segment[y:y+h, x:x+w]

            # Append the cropped segment to the list
            segments_list.append(cropped_segment)

            # Save the cropped segment
            output_path = os.path.join(self.output_dir, f"automatic_segment_{i+1}.png")
            cv2.imwrite(output_path, cropped_segment)
            print(f"Saved: {output_path}\n")

        print(f"Automatic segmentation completed. {len(self.masks)} segments saved to {self.output_dir}")
        return segments_list


def main():
    # Create the segmenter instance
    segmenter = SamSegmenter(        
        input_image_name="input_image2.jpg",
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        box_nms_thresh=0.3,
        # reduce_masks=False,
        MAX_MASKS=10
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()    
    segmenter.save_segmentation_results()


if __name__ == "__main__":
    main()
