<!--
SPDX-FileCopyrightText: 2022 Thomas Kramer

SPDX-License-Identifier: CC-BY-SA-4.0
-->

# 1D & 2D Interpolation

`interp` provides functions for interpolation of one dimensional and two dimensional array.

# Example
```rust
use interp::interp2d::Interp2D;

// Create 2D array of values.
let grid = ndarray::array![
    [0.0f64, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0]
];

// Create grid coordinates.
let xs = vec![0.0, 1.0, 2.0];
let ys = vec![3.0, 4.0, 5.0];

// Create interpolator struct.
let interp = Interp2D::new(xs, ys, grid);

// Evaluate the interpolated data at some point.
assert!((interp.eval_no_extrapolation((1.0, 4.0)).unwrap() - 1.0).abs() < 1e-6);
```