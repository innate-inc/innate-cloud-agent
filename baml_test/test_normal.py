import os
import time
import base64
from groq import Groq
from dotenv import load_dotenv


def send_image_to_groq_with_timing(image_b64: str) -> (str, float):
    """
    Sends a base64-encoded image (assumed to be JPEG) to the Groq API using the
    official message structure and times the request.

    Args:
        image_b64 (str): Base64-encoded string for the image.

    Returns:
        Tuple[str, float]: A tuple containing the response from Groq and the elapsed time.
    """
    load_dotenv()  # Load environment variables including GROQ_API_KEY

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # Create the Groq client.
    client = Groq(api_key=api_key)

    # Prepare the image as a Data URL.
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    # Build the message payload as a list: one text prompt and one image.
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }

    print("Sending image to Groq API...")
    start_time = time.time()

    # Send the request to the Groq chat completions endpoint.
    chat_completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[message],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    elapsed_time = time.time() - start_time

    # Extract the response text from the Groq output.
    response_content = chat_completion.choices[0].message.content

    print(f"Received response in {elapsed_time:.2f} seconds.")
    return response_content, elapsed_time


# Example usage:
if __name__ == "__main__":
    # Make sure you have a test image (e.g., "test.jpg") in the same directory.
    image_path = "test.jpg"

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        exit(1)

    image_b64 = base64.b64encode(image_data).decode("utf-8")

    response, duration = send_image_to_groq_with_timing(image_b64)
    print("Groq API response:")
    print(response)
