import time

from segmentation.sam_segmenter import SamSegmenter
from segmentation.sam2_segmenter import Sam2Segmenter
from lvlm_description.llava_descriptor import LlavaDescriptor
from llm_classification.deepseek_classifier import DeepseekClassifier

def segment_sam() -> bool:
    print("\n[PIPELINE] SEGMENTATION WITH SAM -------------------------------------\n")
    
    # Create the segmenter instance
    segmenter = SamSegmenter(
        input_image_name="input_image2.jpg",
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        box_nms_thresh=0.3
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()
    segment_list = segmenter.save_segmentation_results()

    return segment_list is not None

def segment_sam2() -> bool:
    print("\n[PIPELINE] SEGMENTATION WITH SAM2 -------------------------------------\n")
    
    # Create the segmenter instance
    segmenter = Sam2Segmenter(
        input_image_name="2.jpg",
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        box_nms_thresh=0.3
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()
    segment_list = segmenter.save_segmentation_results()

    return segment_list is not None

def describe_llava() -> str:
    print("\n[PIPELINE] SEGMENT DESCRIPTION WITH LLaVA -------------------------------------\n")

    # Create the descriptor instance
    descriptor = LlavaDescriptor()

    # Workflow
    descriptor.load_segments()
    descriptions = descriptor.describe_images()

    # Return the generated descriptions
    return descriptions

def classify_deepseek(descriptions: str) -> str:
    print("\n[PIPELINE] CLASSIFICATION WITH DEEPSEEK -------------------------------------\n")

    # Create the classifier instance
    classifier = DeepseekClassifier(        
        pipeline_descriptions=descriptions
    )

    # Workflow
    classification_results = classifier.classify()

    # Return the classification results
    return classification_results

def main():
    try:
        start_time = time.time()

        # Image segmentation
        masks_were_generated = segment_sam2()
        
        if masks_were_generated:
            # Segment description
            descriptions = describe_llava()

            # Classification
            classify_deepseek(descriptions)

            print(f"\n[PIPELINE] Pipeline execution completed in {time.time() - start_time} seconds.")

        else:
            print("\n[PIPELINE] No masks were generated. Exiting...")
    except Exception as e:
        print(f"\n[PIPELINE] An error occurred: {e}")

if __name__ == "__main__":
    main()