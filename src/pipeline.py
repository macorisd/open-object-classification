from segmentation.sam2_segmenter import SamSegmenter
from lvlm_description.llava_descriptor import LlavaDescriptor

def segment_sam2() -> list:
    print("\nSEGMENTATION WITH SAM2 -------------------------------------\n")
    
    # Create the segmenter instance
    segmenter = SamSegmenter(        
        input_image_name="input_image2.jpg"
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()
    segments = segmenter.save_segmentation_results()

    # Return the generated segments
    return segments

def describe_llava(segments: list) -> str:
    print("\nSEGMENT DESCRIPTION WITH LLaVA -------------------------------------\n")

    # Create the descriptor instance
    descriptor = LlavaDescriptor(
        segment_images=segments
    )

    # Workflow
    descriptor.load_segments()
    descriptions = descriptor.describe_images()

    # Return the generated descriptions
    return descriptions

def classify_deepseek(descriptions: str) -> str:
    pass

def main():
    segments = segment_sam2()
    descriptions = describe_llava(segments)
    print(descriptions)
    # classification = classify_deepseek(descriptions)

    # print(classification)

if __name__ == "__main__":
    main()