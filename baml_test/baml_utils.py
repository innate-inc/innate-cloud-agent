from baml_py import Image
from baml_client import b
from dotenv import load_dotenv

# Load API keys and other environment variables from .env
load_dotenv()


async def extract_receipt_from_url(url: str):
    """
    Extracts a receipt from an image stored at a URL.

    Args:
        url (str): The URL of the receipt image.

    Returns:
        dict: The receipt data based on the Receipt model.
    """
    img = Image.from_url(url)
    output = b.ExtractReceiptFromImage(img)
    return output


async def extract_receipt_from_base64(base64_str: str):
    """
    Extracts a receipt from a base64 encoded receipt image.

    Args:
        base64_str (str): Base64 string encoded image

    Returns:
        dict: The receipt data based on the Receipt model.
    """
    img_64 = Image.from_base64("image/png", base64_str)
    output = await b.ExtractReceiptFromImage(img_64)
    return output
