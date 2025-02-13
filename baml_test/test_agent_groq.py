from va_utils import vision_agent
import base64
import dotenv
import time

dotenv.load_dotenv()

tasks = [
    {
        "name": "SendEmail",
        "description": "Send an email notification to the user",
        "inputs": {"email_address": "string"},
    },
    {
        "name": "GenerateReport",
        "description": "Generate a summary report of recent activities",
        "inputs": {},
    },
    {
        "name": "BackupData",
        "description": "Backup the database to secure storage",
        "inputs": {"db_name": "string", "storage_location": "string"},
    },
    {
        "name": "SaveReceipt",
        "description": "Save the receipt to the database",
        "inputs": {"receipt_id": "string", "amount_paid": "float"},
    },
]


with open("test.jpg", "rb") as file:
    base64_str = base64.b64encode(file.read()).decode("utf-8")


async def main():
    for i in range(10):
        start_time = time.perf_counter()
        try:
            result = await vision_agent(base64_str, "grab a bottle", tasks)
        except Exception as e:
            continue
        elapsed_time = time.perf_counter() - start_time
        print(f"Iteration {i+1}: Time elapsed: {elapsed_time:.4f} seconds")
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
