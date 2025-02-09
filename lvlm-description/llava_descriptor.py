import os
import ollama
import time

class LlavaDescriptor:
    """
    A class to scan a folder for segmented images, generate descriptions using Ollama's llava model,
    and save those descriptions to a text file.
    """

    def __init__(
        self, 
        script_dir: str, 
        llava_model_name: str = "llava:34b",
        prompt: str = "Describe the image."
    ):
        """
        Initialize the paths and create necessary directories.
        """
        self.script_dir = script_dir
        self.llava_model_name = llava_model_name
        self.prompt = prompt
        
        # Build path to the segmentation output directory
        self.segment_dir = os.path.join(
            self.script_dir, 
            "..", 
            "segmentation", 
            "output_segments"
        )

        # Directory to store the text descriptions
        self.descriptions_dir = os.path.join(self.script_dir, "output_descriptions")
        os.makedirs(self.descriptions_dir, exist_ok=True)

        # Prepare timestamped output file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_filename = f"descriptions_{timestamp}.txt"
        self.output_file = os.path.join(self.descriptions_dir, self.output_filename)

    def describe_images(self) -> None:
        """
        Finds PNG images in the segmentation directory, generates a description for each using Ollama's llava model,
        and writes the descriptions to a text file.
        """
        # Gather segment images
        segment_images = sorted([f for f in os.listdir(self.segment_dir) if f.endswith(".png")])
        if not segment_images:
            raise FileNotFoundError(f"No PNG images found in {self.segment_dir}")

        # Describe each image and save to the text file
        with open(self.output_file, "w", encoding="utf-8") as f:
            for i, seg_file in enumerate(segment_images):
                segment_path = os.path.join(self.segment_dir, seg_file)
                response = ollama.chat(
                    model=self.llava_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.prompt,
                            "images": [segment_path]
                        }
                    ]
                )

                description = response["message"]["content"]
                output_line = f"Segment {i} ({seg_file}): {description}"
                print(output_line + "\n")
                f.write(output_line + "\n")

        print(f"Descriptions saved in {self.output_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    descriptor = LlavaDescriptor(script_dir)
    descriptor.describe_images()


if __name__ == "__main__":
    main()
