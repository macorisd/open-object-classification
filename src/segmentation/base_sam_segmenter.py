import os
import shutil
import cv2
import torch
import numpy as np
import time

class BaseSamSegmenter:
    """
    Base class containing common functionalities for SAM-based segmenters.
    """

    CROP_NO_BACKGROUND: int = 1         # Crop the mask without the background
    CROP_BBOX: int = 2                  # Crop the entire bounding box
    CROP_BBOX_PADDING: int = 3          # Crop the entire bounding box with padding
    HIGHLIGHTED_MASK: int = 4           # Highlight the mask in the original image
    HIGHLIGHTED_BBOX: int = 5           # Highlight the mask's bounding box in the original image   

    def __init__(self):
        self.image = None
        self.masks = []
        self.prev_params = None

    def select_device(self) -> None:
        """
        Selects the device for computation.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda") # CUDA NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") # MPS Apple GPU
        else:
            self.device = torch.device("cpu")
        print(f"\nUsing device: {self.device}\n")

        if self.device.type == "cuda":
            # Use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

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

    def crop_segment(self, image: np.ndarray, segment: np.ndarray, mask_data_bbox: list, segmentation: np.ndarray) -> np.ndarray:
        """
        Crops the segment based on the bounding box.

        Args:
            image_bgr (np.ndarray): The original image in BGR format.
            segment (np.ndarray): The segment image to crop.
            mask_data_bbox (list): The bounding box coordinates of the mask.
            segmentation (np.ndarray): The mask segmentation.

        Returns:
            np.ndarray: The cropped segment image.
        """
        x, y, w, h = [int(coord) for coord in mask_data_bbox]
        
        if self.segment_crop_type == self.CROP_NO_BACKGROUND: # Crop the mask without the background
            cropped_segment = segment[y:y+h, x:x+w]
        elif self.segment_crop_type == self.CROP_BBOX: # Crop the entire bounding box
            cropped_segment = self.image_bgr[y:y+h, x:x+w]
        elif self.segment_crop_type == self.CROP_BBOX_PADDING: # Crop the entire bounding box with padding
            padding = 25
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
            cropped_segment = image[y1:y2, x1:x2]
        elif self.segment_crop_type == self.HIGHLIGHTED_MASK: # Highlight the mask in the original image
            highlighted_image = image.copy()
            mask_255 = segmentation * 255
            contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), thickness=2)
            cropped_segment = highlighted_image
        elif self.segment_crop_type == self.HIGHLIGHTED_BBOX: # Highlight the mask's bounding box in the original image
            highlighted_image = image.copy()
            cv2.rectangle(highlighted_image, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
            cropped_segment = highlighted_image

        return cropped_segment

    def save_segmentation_results(self) -> list:
        """
        Saves the segmented images based on the generated masks to disk and returns a list of the cropped segments.

        Returns:
            list: A list containing the cropped segment images as NumPy arrays.
        """
        if not self.masks:
            print("No masks were generated. Call generate_masks() before save_segmentation_results(), and if you've already done so, check the mask generator parameters.")
            return None

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
            cropped_segment = self.crop_segment(image=image_bgr, segment=segment, mask_data_bbox=mask_data["bbox"], segmentation=segmentation)

            # Append the cropped segment to the list
            segments_list.append(cropped_segment)

            # Save the cropped segment
            output_path = os.path.join(self.output_dir, f"automatic_segment_{i+1}.png")
            cv2.imwrite(output_path, cropped_segment)
            print(f"Saved: {output_path}\n")

        print(f"Automatic segmentation completed. {len(self.masks)} segments saved to {self.output_dir}")
        return segments_list
