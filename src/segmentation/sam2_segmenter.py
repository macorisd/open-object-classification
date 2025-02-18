import os
import shutil
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2Segmenter:
    """
    A class to handle image loading, segmentation with SAM2, and saving mask results.
    """

    CROP_NO_BACKGROUND: int = 1
    CROP_BBOX: int = 2

    def __init__(
        self,
        input_image_name: str = "input_image.jpg",
        model_cfg: str = "sam2.1_hiera_l.yaml",
        model_checkpoint: str = "sam2.1_hiera_large.pt",
        max_masks: int = 10,
        segment_min_area: int = 20000,
        segment_max_area_ratio: float = 0.3,
        segment_crop_type: int = 1,
        timeout: int = 120
    ):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_image_path = os.path.join(self.script_dir, "input_images", input_image_name)
        self.output_dir = os.path.join(self.script_dir, "output_segments")        
        self.model_cfg = os.path.join("./configs/sam2.1/", model_cfg)
        self.model_checkpoint = os.path.join(self.script_dir, model_checkpoint)
        self.max_masks = max_masks
        self.segment_min_area = segment_min_area
        self.segment_max_area_ratio = segment_max_area_ratio
        self.segment_crop_type = segment_crop_type
        self.timeout = timeout
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SAM2 model
        self.model = build_sam2(self.model_cfg, self.model_checkpoint).to(self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        
        self.image = None
        self.masks = []

    def load_image(self) -> None:
        image_bgr = cv2.imread(self.input_image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image at {self.input_image_path}")
        self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)

    def generate_masks(self) -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.masks, _, _ = self.predictor.predict()

        print(f"Generated {len(self.masks)} masks.")

    def save_segmentation_results(self) -> list:
        # Remove the previous output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        segments_list = []
        for i, mask in enumerate(self.masks):
            # Ensure the mask is of type uint8
            mask = mask.astype(np.uint8)

            # Convert the original image from RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            # Create a segmented image by applying the mask (scaling mask by 255 for visibility)
            segment = cv2.bitwise_and(image_bgr, image_bgr, mask=mask * 255)

            # Compute bounding box coordinates from the mask
            ys, xs = np.where(mask > 0)
            if ys.size == 0 or xs.size == 0:
                print(f"Mask {i+1} is empty. Skipping.")
                continue  # Skip empty masks

            # Bounding box coordinates: (x_min, y_min, x_max, y_max)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Crop the segment using the computed bounding box
            cropped_segment = segment[y_min:y_max + 1, x_min:x_max + 1]
            segments_list.append(cropped_segment)

            # Save the cropped segment to a file
            output_path = os.path.join(self.output_dir, f"sam2_segment_{i+1}.png")
            cv2.imwrite(output_path, cropped_segment)
            print(f"Saved: {output_path}")

        return segments_list



def main():
    segmenter = Sam2Segmenter(
        input_image_name="2.jpg",
        model_cfg="sam2.1_hiera_l.yaml",
        model_checkpoint="sam2.1_hiera_large.pt",
        max_masks=10,
        segment_crop_type=Sam2Segmenter.CROP_NO_BACKGROUND
    )
    
    segmenter.load_image()
    segmenter.generate_masks()
    segmenter.save_segmentation_results()


if __name__ == "__main__":
    main()
