from baml_utils import vision_agent
import base64
import dotenv

dotenv.load_dotenv()

with open("test.jpg", "rb") as file:
    base64_str = base64.b64encode(file.read()).decode("utf-8")


async def main():
    res = await vision_agent(base64_str, "grab a bottle")
    print(res)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
