from segmentation.sam_segmenter import SamSegmenter
from lvlm_description.llava_descriptor import LlavaDescriptor
from llm_classification.deepseek_classifier import DeepseekClassifier

def segment_sam2() -> bool:
    print("\n[PIPELINE] SEGMENTATION WITH SAM2 -------------------------------------\n")
    
    # Create the segmenter instance
    segmenter = SamSegmenter(
        input_image_name="2.jpg",
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        box_nms_thresh=0.5
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
        masks_were_generated = segment_sam2()

        if masks_were_generated:
            descriptions = describe_llava()
            classify_deepseek(descriptions)

        else:
            print("\n[PIPELINE] No masks were generated. Exiting...")
    except Exception as e:
        print(f"\n[PIPELINE] An error occurred: {e}")

if __name__ == "__main__":
    main()