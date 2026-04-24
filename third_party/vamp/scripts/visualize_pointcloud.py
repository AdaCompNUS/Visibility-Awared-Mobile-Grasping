import argparse
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import pybullet as pb
import open3d as o3d

from vamp.pointcloud import problem_to_pointcloud


def load_problem(problems_path: Path, scenario: str | None, index: int) -> dict:
    with open(problems_path, "r") as f:
        data = json.load(f)

    problems = data["problems"]
    if scenario is None:
        scenario = sorted(problems.keys())[0]

    if scenario not in problems:
        raise ValueError(
            f"Scenario '{scenario}' not found. Available: {sorted(problems.keys())}"
        )

    if index < 0 or index >= len(problems[scenario]):
        raise IndexError(
            f"Index {index} out of range for scenario '{scenario}' with {len(problems[scenario])} problems"
        )

    return problems[scenario][index]


def draw_points_batched(pc: np.ndarray, colors: np.ndarray, point_size: float, chunk_size: int) -> None:
    total = pc.shape[0]
    if total == 0:
        return
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        pb.addUserDebugPoints(pc[start:end], colors[start:end], pointSize=point_size, lifeTime=0)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a pure environment pointcloud in PyBullet (white background)"
    )
    parser.add_argument(
        "--problems",
        type=Path,
        default=Path("resources/panda/problems.json"),
        help="Path to problems.json",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Path to a PLY pointcloud (overrides --problems)",
    )
    parser.add_argument(
        "--ply-dir",
        type=Path,
        default=None,
        help="Directory containing .ply files to load (overrides --ply and --problems)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario key inside problems.json (default: first available)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Problem index within the scenario",
    )
    parser.add_argument(
        "--samples-per-object",
        type=int,
        default=300,
        help="Surface samples per environment object",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size for visualization",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Subsample to at most this many points when reading PLY",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Add points to PyBullet in chunks of this size to avoid limits",
    )
    args = parser.parse_args()

    # Decide source of pointcloud
    if args.ply_dir is not None:
        pc = read_ply_dir_vertices(args.ply_dir, args.max_points)
    elif args.ply is not None:
        if args.ply.is_dir():
            pc = read_ply_dir_vertices(args.ply, args.max_points)
        else:
            pc = read_ply_vertices(args.ply, args.max_points)
    else:
        problem = load_problem(args.problems, args.scenario, args.index)
        pc = problem_to_pointcloud(problem, args.samples_per_object)

    # Color theme: same as PyBulletSimulator.draw_pointcloud
    maxes = np.max(pc, axis=0)
    maxes[maxes == 0] = 1.0
    colors = 0.8 * (pc / maxes)

    # Connect to PyBullet GUI with white background
    pb.connect(
        pb.GUI,
        options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0",
    )
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

    # Draw points in chunks to avoid PyBullet per-call limits
    draw_points_batched(pc, colors, args.point_size, args.chunk_size)

    print("Close the window or press Ctrl+C to exit.")
    try:
        while pb.isConnected():
            time.sleep(0.016)
    except KeyboardInterrupt:
        pass
    finally:
        if pb.isConnected():
            pb.disconnect()


def read_ply_vertices(path: Path, max_points: int | None = None) -> np.ndarray:
    """Read vertex positions (x,y,z) from a PLY file (ASCII or binary_little_endian).

    Returns an (N, 3) float32 numpy array. If max_points is set and N > max_points,
    uniformly subsample without replacement.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header_text = b"".join(header_lines).decode("ascii", errors="ignore")
        fmt = None
        vertex_count = None
        vertex_props = []
        in_vertex = False
        for raw in header_text.splitlines():
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                # e.g., property float x
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))

        if fmt is None or vertex_count is None or len(vertex_props) == 0:
            raise ValueError("Invalid or unsupported PLY header")

        # Determine indices of x, y, z within vertex properties
        prop_names = [p[1] for p in vertex_props]
        try:
            ix = prop_names.index("x")
            iy = prop_names.index("y")
            iz = prop_names.index("z")
        except ValueError as e:
            raise ValueError("PLY vertex properties must include x, y, z") from e

        if fmt == "ascii":
            # Read ASCII vertex lines
            ascii_stream = io.TextIOWrapper(f, encoding="ascii", errors="ignore")
            rows = []
            for _ in range(vertex_count):
                line = ascii_stream.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < len(vertex_props):
                    continue
                rows.append((float(parts[ix]), float(parts[iy]), float(parts[iz])))
            pc = np.asarray(rows, dtype=np.float32)
        elif fmt == "binary_little_endian":
            # Build numpy dtype for vertex
            np_dtype = []
            for t, name in [(p[0], p[1]) for p in vertex_props]:
                np_dtype.append((name, _ply_scalar_to_dtype(t)))
            dtype = np.dtype(np_dtype, align=False)
            data = f.read(vertex_count * dtype.itemsize)
            if len(data) < vertex_count * dtype.itemsize:
                raise ValueError("Unexpected EOF in PLY vertex data")
            arr = np.frombuffer(data, dtype=dtype, count=vertex_count)
            pc = np.stack([arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32)], axis=1)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

        if max_points is not None and pc.shape[0] > max_points:
            # Use Open3D farthest point sampling
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64, copy=False))
            fps = pcd.farthest_point_down_sample(max_points)
            pc = np.asarray(fps.points, dtype=np.float32)
        return pc


def _ply_scalar_to_dtype(token: str) -> np.dtype:
    """Map PLY scalar type to a little-endian NumPy dtype string."""
    mapping = {
        "char": "<i1",
        "int8": "<i1",
        "uchar": "<u1",
        "uint8": "<u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    if token not in mapping:
        # Default to float32 if unknown, to avoid crashing on uncommon types
        return np.dtype("<f4")
    return np.dtype(mapping[token])


def read_ply_dir_vertices(dir_path: Path, max_points: int | None = None) -> np.ndarray:
    """Read and concatenate vertices from all .ply files in a directory.

    If max_points is set and the total points exceed it, apply a global subsample.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    ply_files = sorted(dir_path.glob("*.ply"))
    if not ply_files:
        raise ValueError(f"No .ply files found in {dir_path}")

    pcs = []
    for pf in ply_files:
        try:
            pcs.append(read_ply_vertices(pf, None))
        except Exception as e:
            print(f"Warning: failed to read {pf}: {e}")

    if not pcs:
        raise ValueError(f"Failed to read any PLY files in {dir_path}")

    pc = np.vstack(pcs)
    if max_points is not None and pc.shape[0] > max_points:
        # Use Open3D farthest point sampling over the concatenated set
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64, copy=False))
        fps = pcd.farthest_point_down_sample(max_points)
        pc = np.asarray(fps.points, dtype=np.float32)
    return pc



if __name__ == "__main__":
    main()

