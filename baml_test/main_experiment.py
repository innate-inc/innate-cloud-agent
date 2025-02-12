import asyncio
import base64
import time
import json
from baml_utils import (
    extract_receipt_from_base64_11b_two,
    extract_receipt_from_base64_11b_three,
    extract_receipt_from_base64_11b_five,
)


# A helper function that will be used to encode Pydantic models.
def pydantic_default(o):
    if hasattr(o, "dict"):
        return o.dict()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


async def run_experiment(func, experiment_name, base64_str, iterations=10):
    """
    Runs the specified extraction function for 'iterations' times,
    collects timing, and records errors.

    Differentiates between extraction errors and rate limit errors.
    In case of a rate limit error, waits one minute and retries the iteration.

    Returns a dict with the experiment results.
    """
    print(f"Starting experiment '{experiment_name}' over {iterations} iterations.")
    times = []
    results = []
    json_results = []
    extraction_error_count = 0
    rate_limit_error_count = 0

    for i in range(1, iterations + 1):
        while True:
            start = time.perf_counter()
            try:
                result = await func(base64_str)
            except Exception as e:
                duration = time.perf_counter() - start
                error_message = str(e)
                if "rate limit" in error_message.lower():
                    rate_limit_error_count += 1
                    print(
                        f"Iteration {i} ({experiment_name}) rate limit encountered. Waiting 30 seconds and retrying..."
                    )
                    await asyncio.sleep(30)
                    continue  # Retry the current iteration after waiting.
                else:
                    extraction_error_count += 1
                    print(
                        f"Iteration {i} ({experiment_name}) extraction error: {error_message}"
                    )
                    results.append("extraction error")
                    json_results.append(
                        {
                            "iteration": i,
                            "duration": round(duration, 4),
                            "result": "extraction error",
                        }
                    )
                    times.append(duration)
                    break  # Exit the while loop for this iteration.
            else:
                duration = time.perf_counter() - start
                results.append(f"{duration:.4f} seconds")
                json_results.append(
                    {"iteration": i, "duration": round(duration, 4), "result": result}
                )
                times.append(duration)
                break  # Exit the while loop for a successful iteration.

    # Calculate statistics over the iterations.
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0
    total_error_count = extraction_error_count + rate_limit_error_count

    experiment_data = {
        "header": FILE_HEADER,
        "model": experiment_name,
        "statistics": {
            "average_time": round(avg_time, 4),
            "max_time": round(max_time, 4),
            "min_time": round(min_time, 4),
            "error_count": total_error_count,
            "extraction_error_count": extraction_error_count,
            "rate_limit_error_count": rate_limit_error_count,
        },
        "iterations": json_results,
        "text_results": results,
    }
    return experiment_data


async def main():
    iterations = 10

    global FILE_HEADER
    FILE_HEADER = (
        "=== Multi-Text Experiment Results ===\n"
        "This header can be easily changed.\n\n"
    )

    # Open test.jpg and encode it as base64 (re-use the same image for the experiment)
    with open("test.jpg", "rb") as file:
        base64_str = base64.b64encode(file.read()).decode("utf-8")

    # Define experiments: Tuple with (experiment_name, extraction_function)
    experiments = [
        # ("llama-3.2-11b-vision-plus-one-lorem", extract_receipt_from_base64_11b_two),
        # ("llama-3.2-11b-vision-plus-two-lorem", extract_receipt_from_base64_11b_three),
        ("llama-3.2-11b-vision-plus-four-lorem", extract_receipt_from_base64_11b_five),
    ]

    # Dictionary to store experiment results.
    all_experiment_results = {}

    # Run each experiment.
    for exp_name, func in experiments:
        result = await run_experiment(func, exp_name, base64_str, iterations)
        all_experiment_results[exp_name] = result

        # Write a text file with the experiment details.
        text_content = FILE_HEADER
        text_content += f"Model: {exp_name}\n"
        text_content += "Statistics:\n"
        stats = result["statistics"]
        text_content += f"Average Time: {stats['average_time']:.4f} seconds\n"
        text_content += f"Max Time: {stats['max_time']:.4f} seconds\n"
        text_content += f"Min Time: {stats['min_time']:.4f} seconds\n"
        text_content += f"Error Count: {stats['error_count']}\n"
        text_content += f"Extraction Error Count: {stats['extraction_error_count']}\n"
        text_content += f"Rate Limit Error Count: {stats['rate_limit_error_count']}\n\n"
        text_content += "Iteration Results:\n"
        for i, res in enumerate(result["text_results"], start=1):
            text_content += f"Iteration {i}: {res}\n"

        txt_filename = f"{exp_name}.txt"
        with open(txt_filename, "w") as f:
            f.write(text_content)
        print(f"Written text results to {txt_filename}")

    # Write all experiments' JSON results to separate JSON files.
    for exp_name, data in all_experiment_results.items():
        json_filename = f"{exp_name}.json"
        with open(json_filename, "w") as f:
            json.dump(data, f, indent=4, default=pydantic_default)
        print(f"Written JSON results to {json_filename}")


if __name__ == "__main__":
    asyncio.run(main())
