#!/usr/bin/env python3
"""
Navigation Point Testing Tool

This script allows testing the navigation point sampling and visualization
without running the entire robot system. It provides test maps and positions
and displays the navigation points on the map.

Usage:
    python test_navigation_points.py

Advanced Usage:
    python test_navigation_points.py --angle-test     # Tests different angle ranges
    python test_navigation_points.py --position-grid  # Tests positions in a grid pattern
    python test_navigation_points.py --rotation-test  # Tests different orientations at one position

Requirements:
    - OpenCV (cv2)
    - NumPy
    - The src module from the project

This will generate map visualizations in the navigation_points directory.
"""

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime
import math
import argparse
import base64

# Add the current directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.primitives.navigate_in_sight import NavigateInSight
from src.utils import decode_map_payload


def load_test_map(map_path):
    """Load a test map from a file."""
    print(f"Loading map from {map_path}")

    with open(map_path, "r") as f:
        map_payload = json.load(f)

    return map_payload


def create_simple_test_map():
    """Create a simple test map with some obstacles."""
    # Create a 10x10 meter map with 0.05m resolution (200x200 grid)
    width = 200
    height = 200
    resolution = 0.05

    # Initialize an empty map (0 = free space)
    map_array = np.zeros((height, width), dtype=np.int8)

    # Add some obstacles (100 = obstacle)
    # Add a wall at the top
    map_array[10:20, 20:180] = 100

    # Add a wall on the right
    map_array[20:180, 170:180] = 100

    # Add a wall at the bottom
    map_array[170:180, 20:170] = 100

    # Add a wall on the left
    map_array[20:170, 10:20] = 100

    # Add some furniture-like obstacles
    # Table in the center
    map_array[80:120, 80:120] = 100

    # Sofa on the left
    map_array[50:90, 30:50] = 100

    # Chair on the right
    map_array[50:70, 140:160] = 100

    # Create the map payload
    # Note: The decode_map_payload function expects base64 encoded data
    map_data = base64.b64encode(map_array.tobytes()).decode("ascii")

    map_payload = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": -5.0,  # Center the map around (0,0)
        "origin_y": -5.0,
        "origin_z": 0.0,
        "origin_yaw": 0.0,
        "frame_id": "map",
        "data": map_data,
    }

    return map_payload


def create_complex_test_map():
    """Create a more complex test map with rooms, corridors, and more obstacles."""
    # Create a 20x20 meter map with 0.05m resolution (400x400 grid)
    width = 400
    height = 400
    resolution = 0.05

    # Initialize an empty map (0 = free space)
    map_array = np.zeros((height, width), dtype=np.int8)

    # Add outer walls
    map_array[10:20, 20:380] = 100  # Top wall
    map_array[20:380, 370:380] = 100  # Right wall
    map_array[370:380, 20:370] = 100  # Bottom wall
    map_array[20:370, 10:20] = 100  # Left wall

    # Room divider walls
    map_array[20:180, 150:160] = 100  # Vertical room divider (top left)
    map_array[180:190, 20:280] = 100  # Horizontal room divider
    map_array[190:370, 150:160] = 100  # Vertical room divider (bottom left)
    map_array[150:300, 280:290] = 100  # Vertical room divider (right)

    # Add doorways (gaps in walls)
    map_array[100:120, 150:160] = 0  # Door in top left divider
    map_array[180:190, 80:100] = 0  # Door in horizontal divider (left)
    map_array[180:190, 220:240] = 0  # Door in horizontal divider (right)
    map_array[260:280, 150:160] = 0  # Door in bottom left divider
    map_array[210:230, 280:290] = 0  # Door in right divider

    # Add some furniture-like obstacles
    # Tables
    map_array[50:70, 50:90] = 100  # Table in top left room
    map_array[50:70, 220:260] = 100  # Table in top right room
    map_array[260:300, 50:90] = 100  # Table in bottom left room
    map_array[290:330, 300:340] = 100  # Table in bottom right room

    # Chairs and sofas
    map_array[90:110, 40:60] = 100  # Chair in top left
    map_array[90:130, 220:240] = 100  # Sofa in top right
    map_array[230:250, 60:100] = 100  # Sofa in bottom left
    map_array[300:320, 230:250] = 100  # Chair in bottom right

    # Narrow corridor for testing
    map_array[300:305, 160:280] = 100  # Narrow corridor walls
    map_array[340:345, 160:280] = 100  # Narrow corridor walls

    # Create the map payload
    map_data = base64.b64encode(map_array.tobytes()).decode("ascii")

    map_payload = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": -10.0,  # Center the map around (0,0)
        "origin_y": -10.0,
        "origin_z": 0.0,
        "origin_yaw": 0.0,
        "frame_id": "map",
        "data": map_data,
    }

    return map_payload


def save_test_map(map_payload, output_path):
    """Save a test map to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(map_payload, f)

    print(f"Saved test map to {output_path}")


def test_navigation_points(
    map_payload,
    robot_positions,
    min_obstacle_distance=0.25,
    num_samples=8,
    test_name="standard",
):
    """Test navigation point sampling and visualization for different robot positions."""

    # Create the NavigateInSight object
    nav = NavigateInSight()

    # Decode the map payload
    map_array, map_info = decode_map_payload(map_payload)

    # Add min_obstacle_distance to map_info for visualization
    map_info["min_obstacle_distance"] = min_obstacle_distance

    # Create directory for navigation images if it doesn't exist
    test_dir = f"navigation_points/{test_name}"
    os.makedirs(test_dir, exist_ok=True)

    # Track the angular distribution of points
    angular_stats = {}

    # Process each robot position
    for position_name, position in robot_positions.items():
        print(f"\nTesting position: {position_name}")
        current_x, current_y, current_yaw = position

        # Sample valid navigation points
        navigation_points = nav.sample_valid_navigation_points(
            current_x,
            current_y,
            current_yaw,
            map_array,
            map_info,
            min_distance=0.5,
            max_distance=2.5,
            min_obstacle_distance=min_obstacle_distance,
            num_samples=num_samples,
        )

        print(f"Found {len(navigation_points)} valid navigation points")

        if not navigation_points:
            print("No valid navigation points found!")
            continue

        # Calculate angular distribution statistics
        angles = []
        for x, y, theta in navigation_points:
            # Calculate angle relative to robot's orientation
            dx = x - current_x
            dy = y - current_y
            point_angle = math.atan2(dy, dx)
            rel_angle = point_angle - current_yaw
            # Normalize to [-pi, pi]
            while rel_angle > math.pi:
                rel_angle -= 2 * math.pi
            while rel_angle < -math.pi:
                rel_angle += 2 * math.pi
            angles.append(math.degrees(rel_angle))

        # Store statistics
        if angles:
            angular_stats[position_name] = {
                "min_angle": min(angles),
                "max_angle": max(angles),
                "range": max(angles) - min(angles),
                "count": len(angles),
                "std_dev": np.std(angles) if len(angles) > 1 else 0,
                "all_angles": angles,
            }

        # Visualize the map with navigation points
        map_vis = nav.visualize_map_with_navigation_points(
            map_array,
            map_info,
            current_x,
            current_y,
            current_yaw,
            navigation_points,
        )

        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_vis_path = f"{test_dir}/map_{position_name}_{timestamp}.jpg"
        cv2.imwrite(map_vis_path, map_vis)
        print(f"Saved map visualization to {map_vis_path}")

        # Print the navigation points
        print("Navigation points (x, y, theta):")
        for i, (x, y, theta) in enumerate(navigation_points):
            degrees = math.degrees(theta)
            print(
                f"  Point {i+1}: ({x:.2f}, {y:.2f}, {theta:.2f} rad / {degrees:.1f}°)"
            )

    # Print angular distribution statistics summary
    print("\n--- Angular Distribution Statistics ---")
    for pos_name, stats in angular_stats.items():
        print(f"{pos_name}:")
        print(
            f"  Range: {stats['min_angle']:.1f}° to {stats['max_angle']:.1f}° (span: {stats['range']:.1f}°)"
        )
        print(f"  Count: {stats['count']} points")
        print(f"  Std Dev: {stats['std_dev']:.2f}°")
        print(f"  All angles: {[f'{a:.1f}°' for a in sorted(stats['all_angles'])]}")
        print()

    return angular_stats


def test_angle_ranges(map_payload, min_obstacle_distance=0.25):
    """Test how different angle ranges affect point distribution."""
    # Create a custom version of NavigateInSight to override angle_range
    nav = NavigateInSight()

    # Reference to the original method
    original_sample_method = nav.sample_valid_navigation_points

    # Position to test from (center of map, facing east)
    test_position = (0.0, 0.0, 0.0)

    # Different angle ranges to test (in degrees)
    angle_ranges = [60, 90, 120, 180, 240, 360]

    # Create directory for results
    test_dir = "navigation_points/angle_test"
    os.makedirs(test_dir, exist_ok=True)

    # Decode the map payload
    map_array, map_info = decode_map_payload(map_payload)
    map_info["min_obstacle_distance"] = min_obstacle_distance

    results = {}

    for angle_deg in angle_ranges:
        print(f"\nTesting angle range: {angle_deg}°")

        # Override the method with a custom angle range
        def custom_sample_method(*args, **kwargs):
            # Extract the positional arguments
            current_x, current_y, current_yaw = args[0], args[1], args[2]
            map_array, map_info = args[3], args[4]

            # Get other parameters from kwargs or use defaults
            min_distance = kwargs.get("min_distance", 0.5)
            max_distance = kwargs.get("max_distance", 2.5)
            min_obstacle_distance = kwargs.get("min_obstacle_distance", 0.25)
            num_samples = kwargs.get("num_samples", 8)

            # Modified method with custom angle range
            import numpy as np
            import math

            # Get map metadata
            resolution = map_info["resolution"]
            origin_x = map_info["origin_x"]
            origin_y = map_info["origin_y"]

            # Sample points in a sector in front of the robot
            valid_points = []

            # Calculate obstacle radius in grid cells
            obstacle_radius = int(min_obstacle_distance / resolution)

            # Custom angle range
            angle_range = angle_deg * (np.pi / 180)  # Convert degrees to radians

            # Sample angles with uniform distribution
            angles = np.linspace(
                current_yaw - angle_range / 2,
                current_yaw + angle_range / 2,
                num_samples,
            )

            # Sample distances
            distances = np.linspace(
                min_distance, max_distance, 3
            )  # 3 distances per angle

            # For each combination of angle and distance
            for angle in angles:
                for distance in distances:
                    # Calculate world coordinates
                    point_x = current_x + distance * np.cos(angle)
                    point_y = current_y + distance * np.sin(angle)

                    # Convert to grid coordinates
                    grid_x = int((point_x - origin_x) / resolution)
                    grid_y = int((point_y - origin_y) / resolution)

                    # Skip if outside map bounds
                    if (
                        grid_x < 0
                        or grid_x >= map_info["width"]
                        or grid_y < 0
                        or grid_y >= map_info["height"]
                    ):
                        continue

                    # Check if point is a valid navigable location (away from obstacles)
                    is_valid = True

                    # Define a search window around the point
                    min_x = max(0, grid_x - obstacle_radius)
                    max_x = min(map_info["width"] - 1, grid_x + obstacle_radius)
                    min_y = max(0, grid_y - obstacle_radius)
                    max_y = min(map_info["height"] - 1, grid_y + obstacle_radius)

                    # Check if any cell within the radius is an obstacle (value = 100)
                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            # Calculate distance from this cell to the target cell
                            cell_distance = math.sqrt(
                                (x - grid_x) ** 2 + (y - grid_y) ** 2
                            )

                            # If this cell is within our search radius and is an obstacle
                            if (
                                cell_distance <= obstacle_radius
                                and map_array[y, x] == 100
                            ):
                                is_valid = False
                                break

                        if not is_valid:
                            break

                    # Add the point if it's valid
                    if is_valid:
                        # The navigation target will face in the direction from robot to point
                        target_theta = angle
                        valid_points.append((point_x, point_y, target_theta))

                        # Limit the total number of points - but collect more for analysis
                        if len(valid_points) >= num_samples * 2:
                            return valid_points

            return valid_points

        # Replace the method temporarily
        nav.sample_valid_navigation_points = custom_sample_method

        # Sample points with the custom method
        navigation_points = nav.sample_valid_navigation_points(
            test_position[0],
            test_position[1],
            test_position[2],
            map_array,
            map_info,
            min_obstacle_distance=min_obstacle_distance,
            num_samples=12,  # Request more samples to see distribution
        )

        print(f"Found {len(navigation_points)} valid navigation points")

        # Calculate angular distribution
        angles = []
        for x, y, theta in navigation_points:
            # Calculate angle relative to robot's orientation
            dx = x - test_position[0]
            dy = y - test_position[1]
            point_angle = math.atan2(dy, dx)
            rel_angle = point_angle - test_position[2]
            # Normalize to [-pi, pi]
            while rel_angle > math.pi:
                rel_angle -= 2 * math.pi
            while rel_angle < -math.pi:
                rel_angle += 2 * math.pi
            angles.append(math.degrees(rel_angle))

        # Store statistics
        if angles:
            results[f"{angle_deg}deg"] = {
                "min_angle": min(angles),
                "max_angle": max(angles),
                "range": max(angles) - min(angles),
                "count": len(angles),
                "std_dev": np.std(angles) if len(angles) > 1 else 0,
                "all_angles": angles,
            }

        # Visualize the map with navigation points
        map_vis = nav.visualize_map_with_navigation_points(
            map_array,
            map_info,
            test_position[0],
            test_position[1],
            test_position[2],
            navigation_points,
        )

        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_vis_path = f"{test_dir}/angle_range_{angle_deg}deg_{timestamp}.jpg"
        cv2.imwrite(map_vis_path, map_vis)
        print(f"Saved map visualization to {map_vis_path}")

    # Restore the original method
    nav.sample_valid_navigation_points = original_sample_method

    # Print results
    print("\n--- Angle Range Test Results ---")
    for range_name, stats in results.items():
        print(f"{range_name}:")
        print(
            f"  Range: {stats['min_angle']:.1f}° to {stats['max_angle']:.1f}° (span: {stats['range']:.1f}°)"
        )
        print(f"  Count: {stats['count']} points")
        print(f"  Std Dev: {stats['std_dev']:.2f}°")
        print(f"  All angles: {[f'{a:.1f}°' for a in sorted(stats['all_angles'])]}")
        print()

    return results


def test_position_grid(map_payload, min_obstacle_distance=0.25, grid_size=3):
    """Test navigation points across a grid of positions in the map."""
    # Decode the map payload
    map_array, map_info = decode_map_payload(map_payload)

    # Get map dimensions in world coordinates
    map_width_meters = map_info["width"] * map_info["resolution"]
    map_height_meters = map_info["height"] * map_info["resolution"]

    # Define bounds for the grid
    min_x = map_info["origin_x"] + 1.0  # 1m from map edge
    max_x = map_info["origin_x"] + map_width_meters - 1.0
    min_y = map_info["origin_y"] + 1.0
    max_y = map_info["origin_y"] + map_height_meters - 1.0

    # Create a grid of positions
    x_positions = np.linspace(min_x, max_x, grid_size)
    y_positions = np.linspace(min_y, max_y, grid_size)

    # Define orientations to test (in degrees)
    orientations = [0, 90, 180, 270]  # East, North, West, South

    grid_positions = {}
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            for angle in orientations:
                yaw = math.radians(angle)
                grid_positions[f"grid_{i}_{j}_facing_{angle}deg"] = (x, y, yaw)

    # Run the test with these positions
    return test_navigation_points(
        map_payload,
        grid_positions,
        min_obstacle_distance=min_obstacle_distance,
        test_name="position_grid",
    )


def test_rotations(map_payload, min_obstacle_distance=0.25):
    """Test how rotation affects point distribution at a fixed position."""
    # Position to test (center of map)
    center_x = 0.0
    center_y = 0.0

    # Create positions rotating 360 degrees
    rotation_positions = {}
    for angle in range(0, 360, 15):  # Every 15 degrees
        yaw = math.radians(angle)
        rotation_positions[f"rot_{angle}deg"] = (center_x, center_y, yaw)

    # Run the test with these positions
    return test_navigation_points(
        map_payload,
        rotation_positions,
        min_obstacle_distance=min_obstacle_distance,
        test_name="rotation_test",
    )


def test_near_obstacles(map_payload, min_obstacle_distance=0.25):
    """Test navigation points when close to obstacles."""
    # Decode the map payload to find obstacle positions
    map_array, map_info = decode_map_payload(map_payload)

    # Define test positions near different types of obstacles
    near_obstacle_positions = {
        "near_wall_front": (0.5, 3.8, math.radians(0)),  # Near wall, facing wall
        "near_wall_parallel": (
            0.5,
            3.8,
            math.radians(90),
        ),  # Near wall, parallel to wall
        "near_corner": (-4.0, -4.0, math.radians(45)),  # In corner, facing corner
        "near_corner_away": (-4.0, -4.0, math.radians(225)),  # In corner, facing away
        "narrow_passage": (0.0, 2.0, math.radians(90)),  # In narrow passage
        "near_object_left": (1.0, 0.0, math.radians(270)),  # Object on left
        "near_object_right": (-1.0, 0.0, math.radians(90)),  # Object on right
        "near_object_front": (0.0, -2.0, math.radians(0)),  # Object in front
    }

    # Run the test with these positions
    return test_navigation_points(
        map_payload,
        near_obstacle_positions,
        min_obstacle_distance=min_obstacle_distance,
        test_name="near_obstacles",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test navigation point sampling and visualization"
    )
    parser.add_argument("--map", type=str, help="Path to map file (JSON format)")
    parser.add_argument(
        "--min-obstacle-distance",
        type=float,
        default=0.25,
        help="Minimum distance from obstacles (meters)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of navigation points to sample",
    )
    parser.add_argument(
        "--complex-map",
        action="store_true",
        help="Use a more complex test map with rooms and corridors",
    )
    parser.add_argument(
        "--angle-test",
        action="store_true",
        help="Test different angle ranges",
    )
    parser.add_argument(
        "--position-grid",
        action="store_true",
        help="Test positions in a grid pattern",
    )
    parser.add_argument(
        "--rotation-test",
        action="store_true",
        help="Test different rotations at one position",
    )
    parser.add_argument(
        "--near-obstacles",
        action="store_true",
        help="Test positions near obstacles",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run all test scenarios",
    )
    args = parser.parse_args()

    # Load or create a test map
    if args.map and os.path.exists(args.map):
        map_payload = load_test_map(args.map)
    elif args.complex_map:
        print("Creating a complex test map")
        map_payload = create_complex_test_map()
        # Save the complex test map
        os.makedirs("maps", exist_ok=True)
        save_test_map(map_payload, "maps/complex_test_map.json")
    else:
        print("Creating a simple test map")
        map_payload = create_simple_test_map()
        # Save the test map for future use
        os.makedirs("maps", exist_ok=True)
        save_test_map(map_payload, "maps/simple_test_map.json")

    # Decide which tests to run
    if args.all_tests or (
        not any(
            [
                args.angle_test,
                args.position_grid,
                args.rotation_test,
                args.near_obstacles,
            ]
        )
    ):
        # Define standard test robot positions (x, y, yaw in radians)
        robot_positions = {
            "center": (0.0, 0.0, 0.0),  # Center of map, facing east
            "center_north": (0.0, 0.0, math.radians(90)),  # Center, facing north
            "center_west": (0.0, 0.0, math.radians(180)),  # Center, facing west
            "center_south": (0.0, 0.0, math.radians(270)),  # Center, facing south
            "near_wall": (0.5, 3.0, math.radians(270)),  # Near a wall, facing south
            "near_table": (
                -1.0,
                -1.0,
                math.radians(45),
            ),  # Near table, facing northeast
            "corner": (-4.0, -4.0, math.radians(45)),  # In a corner, facing northeast
            "corner_away": (
                -4.0,
                -4.0,
                math.radians(225),
            ),  # In corner, facing southwest
            "hallway": (3.0, 0.0, math.radians(180)),  # In a hallway, facing west
            "open_space": (2.0, 2.0, math.radians(135)),  # Open space, facing northwest
        }

        # Run the standard test
        test_navigation_points(
            map_payload, robot_positions, args.min_obstacle_distance, args.num_samples
        )

    # Run specific tests if requested
    if args.all_tests or args.angle_test:
        print("\n=== Running Angle Range Test ===")
        test_angle_ranges(map_payload, args.min_obstacle_distance)

    if args.all_tests or args.position_grid:
        print("\n=== Running Position Grid Test ===")
        test_position_grid(map_payload, args.min_obstacle_distance)

    if args.all_tests or args.rotation_test:
        print("\n=== Running Rotation Test ===")
        test_rotations(map_payload, args.min_obstacle_distance)

    if args.all_tests or args.near_obstacles:
        print("\n=== Running Near Obstacles Test ===")
        test_near_obstacles(map_payload, args.min_obstacle_distance)

    print(
        "\nTesting complete. Check the navigation_points directory for visualizations."
    )


if __name__ == "__main__":
    main()
