# Fixed robot geometry

These are ARX X5 robot geometry and interface definitions, independent of task
objects, layouts, scoring and historical manipulation trials.

The grasp reference is flange-local `[0.145,0,0]` metres. Axes are
`(tool, jaw, tool cross jaw)`. Offsets from the grasp reference:

| Robot landmark | Offset in metres |
| --- | --- |
| Front-tip midpoint | `[0.01257,-0.000002,-0.00014012]` |
| Distal-pad bounding-box midpoint | `[0.00507,-0.000002,-0.00144044]` |

Source: bundled robot model geometry (`ARX.usd` metrology), not object poses or
task trials. The controller and `geom.grasp_reference_for` use the same geometry.
`commanded_gap_mm` is a nominal conversion from commanded finger opening, not
a sensor reading. Robot calibration does not provide object locations.

World points/displacements use metres, rotations use radians, pixels use `(u,v)`
with origin at the upper left. NumPy image indexing is `[v,u]`.
Camera projection and triangulation use supplied per-observation camera matrices.
Plane intersections require a caller-selected plane; no fixed table height,
object dimensions, scoring conditions or manipulation examples are provided.
