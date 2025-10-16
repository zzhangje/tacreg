# tacreg

Simple, lightweight point cloud registration with Open3D. This package provides a TAC-REG implementation that estimates a rigid transform (R, t) between a source and a target point cloud.

## Installation

```bash
pip install git+https://github.com/zzhangje/tacreg
```

## Example

```python
import numpy as np
import open3d as o3d
from tacreg import tacreg, TacRegParam

# Load point clouds (replace with your own paths)
src = o3d.io.read_point_cloud("source.ply")
tar = o3d.io.read_point_cloud("target.ply")

# Configure (adjust voxel_size/radii for your data scale)
params = TacRegParam(voxel_size=0.005, verbose=True)

# Run registration
R, t, score = tacreg(src, tar, params)
print("score:", score)

# Apply transform to visualize alignment
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t
src_aligned = src.transform(T.copy())
o3d.visualization.draw_geometries([src_aligned, tar])
```

## API overview

- `tacreg(src_pcd, tar_pcd, params) -> (R, t, score)`

  - `src_pcd`, `tar_pcd`: `open3d.geometry.PointCloud` objects
  - Returns a 3x3 rotation matrix `R`, 3-vector translation `t`, and a scalar `score` (higher is better)

- `TacRegParam` (all have sensible defaults):
  - `voxel_size`: downsample voxel size
  - `normal_radius`, `normal_max_nn`: normal estimation
  - `iss_*`: ISS keypoint detection params
  - `fpfh_*`: FPFH feature params
  - `correspondence_size`, graph thresholds (`dist_threshold`, `angle_threshold`)
  - `candidate_size`, `parallel_jobs`, `verbose`

## Tips

- Ensure source/target point clouds are in the same scale and units.
- For very dense clouds, increase `voxel_size` to speed up processing.
- If normals are already present and reliable, you can set `estimate_src_normals=False` and/or `estimate_tar_normals=False` in `TacRegParam`.

## License

MIT. See `LICENSE` for details.
