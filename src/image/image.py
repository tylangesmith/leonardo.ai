from PIL import Image
from io import BytesIO
import requests


def load_image_from_url(url: str) -> Image:
    """
    Loads an image from a given URL and returns it as a PIL Image object.

    This function sends a GET request to the specified URL to fetch the image data,
    then it reads this data into a BytesIO stream, and finally, it opens the stream
    as a PIL Image object.

    Args:
        url (str): The URL of the image to be loaded.

    Returns:
        Image: A PIL Image object of the loaded image.
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))
