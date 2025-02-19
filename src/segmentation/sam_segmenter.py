import os
import shutil
import cv2
import torch
import numpy as np
import time

from base_sam_segmenter import BaseSamSegmenter

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamSegmenter(BaseSamSegmenter):
    """
    A class to handle image loading, segmentation with SAM (Segment Anything), and saving mask results. Inherits from BaseSamSegmenter.

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
        segment_crop_type (int): Crop type for segmentation.
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
        segment_max_area_ratio: float = 0.3,
        segment_crop_type: int = 1,
        timeout: int = 120
    ):        
        """
        Initializes the SamSegmenter instance.        
        """

        super().__init__()

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
        self.segment_crop_type = segment_crop_type
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


def main():
    # Create the segmenter instance
    segmenter = SamSegmenter(        
        input_image_name="2.jpg",
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        box_nms_thresh=0.5,
        # reduce_masks=False,
        MAX_MASKS=10,
        segment_crop_type=SamSegmenter.CROP_NO_BACKGROUND
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()    
    segmenter.save_segmentation_results()


if __name__ == "__main__":
    main()
