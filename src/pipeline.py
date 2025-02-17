from segmentation.sam2_segmenter import SamSegmenter
from lvlm_description.llava_descriptor import LlavaDescriptor
from llm_classification.deepseek_classifier import DeepseekClassifier

def segment_sam2() -> list:
    print("\nSEGMENTATION WITH SAM2 -------------------------------------\n")
    
    # Create the segmenter instance
    segmenter = SamSegmenter(        
        input_image_name="input_image2.jpg"
    )

    # Workflow
    segmenter.load_image()
    segmenter.generate_masks()
    segmenter.save_segmentation_results()

def describe_llava() -> str:
    print("\nSEGMENT DESCRIPTION WITH LLaVA -------------------------------------\n")

    # Create the descriptor instance
    descriptor = LlavaDescriptor()

    # Workflow
    descriptor.load_segments()
    descriptions = descriptor.describe_images()

    # Return the generated descriptions
    return descriptions

def classify_deepseek(descriptions: str) -> str:
    print("\nCLASSIFICATION WITH DEEPSEEK -------------------------------------\n")

    # Create the classifier instance
    classifier = DeepseekClassifier(        
        pipeline_descriptions=descriptions
    )

    # Workflow
    classification_results = classifier.classify()

    # Return the classification results
    return classification_results

def main():
    segment_sam2()
    descriptions = describe_llava()    
    classify_deepseek(descriptions)

if __name__ == "__main__":
    main()