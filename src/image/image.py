from PIL import Image
from io import BytesIO
import requests

def load_image_from_url(url: str) -> Image.Image:
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))
  return image
