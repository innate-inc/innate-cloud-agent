import json
from datetime import datetime
from src.history import History, HistoryEntry, HistoryEntryType


def load_history_from_file(file_path):
    history = History()

    with open(file_path, "r") as f:
        data = json.load(f)

    for entry_data in data:
        # Convert the string timestamp back to datetime
        timestamp = datetime.fromisoformat(entry_data["timestamp"])
        # Convert the string type back to enum, handling both old and new type names
        type_str = entry_data["type"]
        entry_type = HistoryEntryType(type_str)

        entry = HistoryEntry(
            timestamp=timestamp,
            type=entry_type,
            description=entry_data["description"],
        )
        history.entries.append(entry)

    return history


def main():
    # Get the most recent history file from the histories folder
    import os
    import glob

    history_files = glob.glob("histories/history_20250329_000024.json")
    if not history_files:
        print("No history files found in the histories folder!")
        return

    # Get the most recent file
    latest_file = max(history_files, key=os.path.getctime)
    print(f"Loading history from: {latest_file}")

    # Load and display the history
    history = load_history_from_file(latest_file)
    print("\nHistory contents:")
    print("=" * 80)
    print(history.get_as_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
