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
        llava_model_name: str = "llava:34b",
        prompt: str = "Describe the image.",
        save_file: bool = True,  # Whether to save the description results to a file
        timeout: int = 120  # Timeout in seconds
    ):
        """
        Initialize the paths and create necessary directories.
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))        
        self.llava_model_name = llava_model_name
        self.prompt = prompt
        self.save_file = save_file
        self.timeout = timeout

        # Directory to store the text descriptions
        self.descriptions_dir = os.path.join(self.script_dir, "output_descriptions")
        os.makedirs(self.descriptions_dir, exist_ok=True)

        # Prepare timestamped output file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_filename = f"descriptions_{timestamp}.txt"
        self.output_file = os.path.join(self.descriptions_dir, self.output_filename)

    def load_segments(self) -> list:
        """
        Load the segment images.
        """
        
        # Build path to the segmentation output directory
        self.segment_dir = os.path.join(
            self.script_dir, 
            "..", 
            "segmentation", 
            "output_segments"
        )

        # Gather segment images
        self.segment_images = sorted([f for f in os.listdir(self.segment_dir) if f.endswith(".png")])
        if not self.segment_images:
            raise FileNotFoundError(f"No PNG images found in {self.segment_dir}")

    def describe_images(self) -> str:
        """
        Finds PNG images in the segmentation directory, generates a description for each using Ollama's llava model,
        and writes the descriptions to a text file.
        """
        descriptions = ""
        start_time = time.time()  # Start timer for timeout
        
        # Describe each image        
        for i, seg_file in enumerate(self.segment_images):
            segment = os.path.join(self.segment_dir, seg_file)                
            description = ""

            while time.time() - start_time < self.timeout:                
                response = ollama.chat(
                    model=self.llava_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.prompt,
                            "images": [segment]
                        }
                    ]
                )
                description = response["message"]["content"]
                if description.strip():
                    break
                else:
                    print("\nThe description is empty. Trying again...\n")
            else:
                raise TimeoutError(f"Timeout of {self.timeout} seconds reached for segment {i} ({seg_file}) without receiving a valid description.")

            output_line = f"Segment {i+1} ({seg_file}): {description}"            
            print(output_line + "\n")
            descriptions += output_line + "\n"            

        # Save the descriptions to a text file if saving is enabled
        if self.save_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(descriptions)
            print(f"Descriptions saved in {self.output_file}")
        else:
            print("Saving file is disabled. Descriptions were not saved.")

        return descriptions


def main():
    descriptor = LlavaDescriptor()
    descriptor.load_segments()
    descriptor.describe_images()


if __name__ == "__main__":
    main()
