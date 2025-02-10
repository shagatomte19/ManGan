import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import io
import os
import gdown

model_path = "models/generator_epoch_39.pth"
drive_file_id = "12Yq2j-8SH61bKzwFurmNyv-_P8s-ejnc"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/file/d/12Yq2j-8SH61bKzwFurmNyv-_P8s-ejnc/view?usp=drive_link", model_path, quiet=False)

# Load the trained GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextToImageGAN(torch.nn.Module):
    def __init__(self, generator_path):
        super().__init__()
        self.generator = torch.load(generator_path, map_location=device)
        self.generator.eval()  # Set to evaluation mode

    def generate_image(self, text_embedding):
        with torch.no_grad():
            generated_image = self.generator(text_embedding.to(device))
            generated_image = (generated_image + 1) / 2  # Normalize to [0,1]
        return generated_image

# Function to generate image from text embedding
def generate_image_from_text(prompt):
    # Convert text to an embedding (implement your embedding method here)
    text_embedding = torch.randn(1, 100).to(device)  # Placeholder for real text embedding

    model = TextToImageGAN("models/generator_epoch_39.pth")
    generated_image = model.generate_image(text_embedding)

    buffer = io.BytesIO()
    save_image(generated_image, buffer, format="PNG")
    buffer.seek(0)

    return Image.open(buffer)
