import os
import ollama
import time

class DeepseekClassifier:
    """
    A class to classify image segments based on textual descriptions generated by LLaVA.
    It uses Ollama's Deepseek model to determine which segments contain recognizable objects.
    """

    def __init__(
        self, 
        script_dir: str,
        deepseek_model_name: str = "deepseek-r1:14b",
        
    ):
        """
        Initializes the paths and creates the classification directory.
        """
        self.script_dir = script_dir
        self.deepseek_model_name = deepseek_model_name

        # Load prompt from prompt.txt
        prompt_path = os.path.join(self.script_dir, "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

        # Directory containing the .txt descriptions (output from the LVLM descriptor)
        self.descriptions_dir = os.path.join(
            self.script_dir,
            "..",
            "lvlm-description",
            "output_descriptions"
        )

        # Directory to store the classification results
        self.classification_dir = os.path.join(self.script_dir, "output_classification")
        os.makedirs(self.classification_dir, exist_ok=True)

    def classify(self) -> None:
        """
        Main workflow to locate the most recent .txt description, classify segments with DeepSeek,
        and save the result.
        """
        # Gather all .txt files in descriptions_dir
        txt_files = [
            os.path.join(self.descriptions_dir, f)
            for f in os.listdir(self.descriptions_dir)
            if f.endswith(".txt")
        ]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.descriptions_dir}")

        # Select and read the most recently modified .txt file
        latest_txt_path = max(txt_files, key=os.path.getmtime)

        with open(latest_txt_path, "r", encoding="utf-8") as f:
            descriptions_content = f.read()

        response = ollama.chat(
            model=self.deepseek_model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt + "\n" + descriptions_content
                }
            ]
        )

        deepseek_answer = response["message"]["content"]

        # Save the Deepseek answer with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"classification_{timestamp}.txt"
        output_path = os.path.join(self.classification_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(deepseek_answer)

        print(f"Deepseek answer saved to {output_path}")
        print("Content:\n", deepseek_answer)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier = DeepseekClassifier(script_dir)
    classifier.classify()


if __name__ == "__main__":
    main()
