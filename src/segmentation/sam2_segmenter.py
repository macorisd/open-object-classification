import os
import shutil
import cv2
import torch
import numpy as np
import time

from base_sam_segmenter import BaseSamSegmenter

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class Sam2Segmenter(BaseSamSegmenter):
    """
    A class to handle image loading, segmentation with SAM2 (Segment Anything 2), and saving mask results. Inherits from BaseSamSegmenter.

    Attributes:
        input_image_name (str): Name of the input image file.
        model_cfg (str): Configuration file for the SAM2 model.
        model_checkpoint (str): Checkpoint file for the SAM2 model.
        pred_iou_thresh (float): IoU threshold for filtering masks.
        stability_score_thresh (float): Stability score threshold for mask quality.
        box_nms_thresh (float): NMS threshold for filtering overlapping boxes.
        reduce_masks (bool): Whether to reduce the number of generated masks.
        MAX_MASKS (int): Maximum number of masks to keep.
        segment_min_area (int): Minimum mask area to keep.
        segment_max_area_ratio (float): Maximum mask area ratio to keep.
        segment_crop_type (int): Crop type for segmentation.
        timeout (int): Timeout in seconds for mask generation.
        device (torch.device): Device to use for the SAM2 model.
        sam2 (torch.nn.Module): SAM2 model instance.
        mask_generator (SAM2AutomaticMaskGenerator): SAM2 automatic mask generator instance.
        image (np.ndarray): Input image as a NumPy array.
        masks (list[dict]): List of dictionaries containing segmentation mask data.
        prev_params (dict): Previous SAM2 mask generator parameters for backtracking.
    """

    def __init__(
        self,
        input_image_name: str = "input_image.jpg",
        model_cfg: str = "sam2.1_hiera_l.yaml",
        model_checkpoint: str = "sam2.1_hiera_large.pt",
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
        super().__init__()

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_image_path = os.path.join(self.script_dir, "input_images", input_image_name)
        self.output_dir = os.path.join(self.script_dir, "output_segments")
        self.model_checkpoint = os.path.join(self.script_dir, model_checkpoint)
        self.model_cfg = os.path.join("configs/sam2.1/", model_cfg)
        self.reduce_masks = reduce_masks
        self.MAX_MASKS = MAX_MASKS
        self.segment_min_area = segment_min_area
        self.segment_max_area_ratio = segment_max_area_ratio
        self.segment_crop_type = segment_crop_type
        self.timeout = timeout

        # Select the device for computation
        self.select_device()

        # Load SAM2 model
        self.sam2 = build_sam2(self.model_cfg, self.model_checkpoint, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh
        )    


def main():
    # Create the segmenter instance
    segmenter = Sam2Segmenter(        
        input_image_name="2.jpg",
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        box_nms_thresh=0.6,
        # reduce_masks=False,
        MAX_MASKS=20,
        segment_crop_type=Sam2Segmenter.CROP_BBOX_PADDING
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()    
    segmenter.save_segmentation_results()


if __name__ == "__main__":
    main()
