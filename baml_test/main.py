import asyncio
import base64
from baml_utils import extract_receipt_from_base64


async def main():
    # Open test.jpg
    with open("test.jpg", "rb") as file:
        base64_str = base64.b64encode(file.read()).decode("utf-8")

    # Extract receipt from base64 (async call with await)
    print("Extracting receipt from base64")
    receipt = await extract_receipt_from_base64(base64_str)
    print(receipt)


if __name__ == "__main__":
    asyncio.run(main())
