import asyncio
import base64
import time
import json
from baml_utils import extract_receipt_from_base64, extract_receipt_from_base64_11b


# A helper function that will be used to encode Pydantic models.
def pydantic_default(o):
    # If the object has a "dict" method (like most Pydantic models), return its dictionary.
    if hasattr(o, "dict"):
        return o.dict()
    # Otherwise, try model_dump (for Pydantic v2)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    # Fall back to string conversion.
    return str(o)


async def main():
    iterations = 10

    # This header is defined at the beginning so it can be easily changed.
    FILE_HEADER = (
        "=== Test Run Results ===\n" "You can change this header content easily.\n\n"
    )

    # Open test.jpg and encode it as base64
    with open("test.jpg", "rb") as file:
        base64_str = base64.b64encode(file.read()).decode("utf-8")

    # --- Test for model llama-3.2-90b-vision-preview ---
    print(f"Testing model llama-3.2-90b-vision-preview over {iterations} iterations")
    times_90b = []
    results_90b = []
    json_results_90b = []
    error_count_90b = 0

    for i in range(1, iterations + 1):
        start = time.perf_counter()
        try:
            result = await extract_receipt_from_base64(base64_str)
            duration = time.perf_counter() - start
            results_90b.append(f"{duration:.4f} seconds")
            json_results_90b.append(
                {"iteration": i, "duration": round(duration, 4), "result": result}
            )
        except Exception as e:
            duration = time.perf_counter() - start
            print(f"Iteration {i} (90b) error: {e}")
            results_90b.append("error")
            json_results_90b.append(
                {"iteration": i, "duration": round(duration, 4), "result": "error"}
            )
            error_count_90b += 1
        times_90b.append(duration)

    # Compute statistics for 90b
    avg_90b = sum(times_90b) / iterations
    max_90b = max(times_90b)
    min_90b = min(times_90b)

    # --- Test for model llama-3.2-11b-vision-preview ---
    print(f"\nTesting model llama-3.2-11b-vision-preview over {iterations} iterations")
    times_11b = []
    results_11b = []
    json_results_11b = []
    error_count_11b = 0

    for i in range(1, iterations + 1):
        start = time.perf_counter()
        try:
            result = await extract_receipt_from_base64_11b(base64_str)
            duration = time.perf_counter() - start
            results_11b.append(f"{duration:.4f} seconds")
            json_results_11b.append(
                {"iteration": i, "duration": round(duration, 4), "result": result}
            )
        except Exception as e:
            duration = time.perf_counter() - start
            print(f"Iteration {i} (11b) error: {e}")
            results_11b.append("error")
            json_results_11b.append(
                {"iteration": i, "duration": round(duration, 4), "result": "error"}
            )
            error_count_11b += 1
        times_11b.append(duration)

    # Compute statistics for 11b
    avg_11b = sum(times_11b) / iterations
    max_11b = max(times_11b)
    min_11b = min(times_11b)

    # --- Write results for llama-3.2-90b-vision-preview (text file) ---
    content_90b = FILE_HEADER
    content_90b += "Model: llama-3.2-90b-vision-preview\n"
    content_90b += "Statistics:\n"
    content_90b += f"Average Time: {avg_90b:.4f} seconds\n"
    content_90b += f"Max Time: {max_90b:.4f} seconds\n"
    content_90b += f"Min Time: {min_90b:.4f} seconds\n"
    content_90b += f"Error Count: {error_count_90b}\n\n"
    content_90b += "Iteration Results:\n"
    for i, result in enumerate(results_90b, start=1):
        content_90b += f"Iteration {i}: {result}\n"

    with open("llama-3.2-90b-vision-preview.txt", "w") as f:
        f.write(content_90b)

    # --- Write results for llama-3.2-11b-vision-preview (text file) ---
    content_11b = FILE_HEADER
    content_11b += "Model: llama-3.2-11b-vision-preview\n"
    content_11b += "Statistics:\n"
    content_11b += f"Average Time: {avg_11b:.4f} seconds\n"
    content_11b += f"Max Time: {max_11b:.4f} seconds\n"
    content_11b += f"Min Time: {min_11b:.4f} seconds\n"
    content_11b += f"Error Count: {error_count_11b}\n\n"
    content_11b += "Iteration Results:\n"
    for i, result in enumerate(results_11b, start=1):
        content_11b += f"Iteration {i}: {result}\n"

    with open("llama-3.2-11b-vision-preview.txt", "w") as f:
        f.write(content_11b)

    # --- Write JSON results for llama-3.2-90b-vision-preview ---
    json_content_90b = {
        "header": FILE_HEADER,
        "model": "llama-3.2-90b-vision-preview",
        "statistics": {
            "average_time": round(avg_90b, 4),
            "max_time": round(max_90b, 4),
            "min_time": round(min_90b, 4),
            "error_count": error_count_90b,
        },
        "iterations": json_results_90b,
    }

    with open("llama-3.2-90b-vision-preview.json", "w") as f:
        json.dump(json_content_90b, f, indent=4, default=pydantic_default)

    # --- Write JSON results for llama-3.2-11b-vision-preview ---
    json_content_11b = {
        "header": FILE_HEADER,
        "model": "llama-3.2-11b-vision-preview",
        "statistics": {
            "average_time": round(avg_11b, 4),
            "max_time": round(max_11b, 4),
            "min_time": round(min_11b, 4),
            "error_count": error_count_11b,
        },
        "iterations": json_results_11b,
    }

    with open("llama-3.2-11b-vision-preview.json", "w") as f:
        json.dump(json_content_11b, f, indent=4, default=pydantic_default)


if __name__ == "__main__":
    asyncio.run(main())
