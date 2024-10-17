from PIL import Image

def preprocess_image(image_path):
    """Load and preprocess the image for the neural network."""
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Example size for a model like ResNet
    image = image.convert('RGB')  # Ensure it's in RGB format
    return image
