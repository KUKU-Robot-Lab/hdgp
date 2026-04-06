#!/usr/bin/env python3
"""Extract TensorBoard data from event files for test3 vs test4 comparison."""

import os
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(event_file_path, max_walltime=None):
    """Extract all scalar tags from event file."""
    try:
        ea = EventAccumulator(str(event_file_path))
        ea.Reload()

        scalars = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            scalars[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_times': [e.wall_time for e in events]
            }

        return scalars
    except Exception as e:
        print(f"Error loading {event_file_path}: {e}", file=sys.stderr)
        return {}

def main():
    test3_dir = Path("/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v9/test3/summaries")
    test4_dir = Path("/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v9/test4/summaries")

    # Find event files
    test3_files = list(test3_dir.glob("events.out.tfevents.*"))
    test4_files = list(test4_dir.glob("events.out.tfevents.*"))

    print(f"Found test3 files: {len(test3_files)}")
    print(f"Found test4 files: {len(test4_files)}")

    if not test3_files or not test4_files:
        print("Error: Could not find event files!")
        return

    # Extract data
    print("\n=== Extracting test3 data ===")
    test3_scalars = extract_scalars(test3_files[0])
    print(f"test3 tags: {len(test3_scalars)}")
    for tag in sorted(test3_scalars.keys())[:10]:
        print(f"  {tag}: {len(test3_scalars[tag]['steps'])} data points")

    print("\n=== Extracting test4 data ===")
    test4_scalars = extract_scalars(test4_files[0])
    print(f"test4 tags: {len(test4_scalars)}")
    for tag in sorted(test4_scalars.keys())[:10]:
        print(f"  {tag}: {len(test4_scalars[tag]['steps'])} data points")

    # Save to files for analysis
    import json

    # Convert to JSON-serializable format
    def convert_to_json(scalars):
        result = {}
        for tag, data in scalars.items():
            result[tag] = {
                'steps': data['steps'],
                'values': data['values']
            }
        return result

    with open('/tmp/test3_scalars.json', 'w') as f:
        json.dump(convert_to_json(test3_scalars), f)

    with open('/tmp/test4_scalars.json', 'w') as f:
        json.dump(convert_to_json(test4_scalars), f)

    print("\n✓ Data saved to /tmp/test3_scalars.json and /tmp/test4_scalars.json")

    # Print all available tags for inspection
    all_tags = sorted(set(test3_scalars.keys()) | set(test4_scalars.keys()))
    print(f"\n=== All available tags ({len(all_tags)}) ===")
    for tag in all_tags:
        test3_len = len(test3_scalars.get(tag, {}).get('steps', []))
        test4_len = len(test4_scalars.get(tag, {}).get('steps', []))
        print(f"{tag:60s} | test3:{test3_len:4d} | test4:{test4_len:4d}")

if __name__ == '__main__':
    main()
