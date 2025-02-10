from flask import Flask, request, jsonify, send_file
import torch
from torchvision.utils import save_image
from model import AdvancedTextEncoder, AdvancedGenerator  # Import your models
from utils import process_prompt  # Utility for tokenizing prompts
import os

app = Flask(__name__)

# Load the models
text_encoder = AdvancedTextEncoder(vocab_size=10000, embed_dim=256, hidden_dim=256)
text_encoder.load_state_dict(torch.load('C:\Users\HP\Desktop\ext2image\model\text' ))
text_encoder.eval()

generator = AdvancedGenerator(text_dim=512, image_channels=3, image_size=128)
generator.load_state_dict(torch.load('C:\\Users\\HP\\Desktop\\text2image\\Backend\\model\\model\\generator_epoch_39.pth'))
generator.eval()

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Process the prompt and generate image
    text_features = process_prompt(prompt, text_encoder)
    with torch.no_grad():
        generated_image = generator(text_features.unsqueeze(0))
        generated_image = (generated_image + 1) / 2.0  # Normalize to [0, 1]

    # Save the image
    output_path = "output/generated_image.png"
    save_image(generated_image, output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
