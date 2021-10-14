/*
 * Copyright (c) 2021-2021 Thomas Kramer.
 *
 * This file is part of interp 
 * (see https://codeberg.org/libreda/interp).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

//! Two-dimensional interpolation.
//!
//! # Example
//! ```
//! use interp::interp2d::Interp2D;
//!
//! let grid = ndarray::array![
//!     [0.0f64, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 0.0]
//! ];
//! let xs = vec![0.0, 1.0, 2.0];
//! let ys = vec![3.0, 4.0, 5.0];
//!
//! // Create interpolator struct.
//! let interp = Interp2D::new(xs, ys, grid);
//!
//! // Evaluate the interpolated data at some point.
//! assert!((interp.eval_no_extrapolation((1.0, 4.0)).unwrap() - 1.0).abs() < 1e-6);
//! ```

use num_traits::{Num};
use ndarray::{Axis, ArrayBase, Data, Ix2, OwnedRepr};
use std::ops::Mul;
use crate::find_closest_neighbours_indices;

// /// Define the extrapolation behaviour when a point outside of the sample grid should
// /// be evaluated.
// #[derive(Copy, Clone, Debug)]
// pub enum ExtrapolationMode {
//     /// Don't extrapolate and panic.
//     Panic,
//     /// Stay at the closest sample value.
//     Constant,
//     /// Extrapolate with closest sample derivative.
//     Linear,
// }

/// Two dimensional bilinear interpolator for data on an irregular grid.
///
/// * `C`: Coordinate type.
/// * `Z`: Value type.
/// * `AZ`: Array representation of `Z` values.
pub struct Interp2D<C, Z, S>
    where Z: Num, S: Data<Elem=Z> {
    x: Vec<C>,
    y: Vec<C>,
    z: ArrayBase<S, Ix2>,
}

impl<C, Z, S> Clone for Interp2D<C, Z, S>
    where Z: Num,
          C: Clone,
          S: Data<Elem=Z>,
          ArrayBase<S, Ix2>: Clone {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

/// Interpolate over a two dimensional *normalized* grid cell.
/// The sample values (`xij`) are located on the corners of the normalized quadratic grid cell:
///
/// ```txt
/// [[x10, x11]
///  [x00, x01]]
/// ```
///
/// The normalized grid cell starts at `(0, 0)` and ends at `(1, 1)`
///
/// `alpha` and `beta` are the normalized coordinates.
/// The parameters `alpha` and `beta` give the position of the interpolation point.
///
/// `alpha` and `beta` should range from `0.0` to `1.0` for interpolation, otherwise
/// the value is *extrapolated*.
fn interpolate2d_bilinear<C, Z>(x00: Z, x10: Z, x01: Z, x11: Z, alpha: C, beta: C) -> Z
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z>, {
    // debug_assert!(alpha >= C::zero() && alpha <= C::one(), "Cannot extrapolate.");
    // debug_assert!(beta >= C::zero() && beta <= C::one(), "Cannot extrapolate.");
    x00 + alpha * (x10 - x00) + beta * (x01 - x00) + alpha * beta * (x00 + x11 - x10 - x01)
}

#[test]
fn test_interpolate2d() {
    // Tolerance for test.
    let tol = 1e-6f64;

    assert!((interpolate2d_bilinear(1.0f64, 2., 3., 4., 0., 0.) - 1.).abs() < tol);
    assert!((interpolate2d_bilinear(1.0f64, 2., 3., 4., 1., 0.) - 2.).abs() < tol);
    assert!((interpolate2d_bilinear(1.0f64, 2., 3., 4., 0., 1.) - 3.).abs() < tol);
    assert!((interpolate2d_bilinear(1.0f64, 2., 3., 4., 1., 1.) - 4.).abs() < tol);
    assert!((interpolate2d_bilinear(0.0f64, 1., 1., 0., 0.5, 0.5) - 0.5).abs() < tol);
}

/// Find the value of `f` at `(x, y)`
/// given four corner values `vij = f(xi, yj)` for all `(i, j) in [0, 1] x [0, 1]`.
fn interp2d<C, Z>((x, y): (C, C),
                  (x0, x1): (C, C), (y0, y1): (C, C),
                  (v00, v10, v01, v11): (Z, Z, Z, Z)) -> Z
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z> {
    let dx = x1 - x0;
    let dy = y1 - y0;

    let alpha = (x - x0) / dx;
    let beta = (y - y0) / dy;

    interpolate2d_bilinear(v00, v10, v01, v11, alpha, beta)
}

impl<C, Z, S> Interp2D<C, Z, S>
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z>,
          S: Data<Elem=Z> + Clone {
    /// Create a new interpolation engine.
    ///
    /// Interpolates values which are sampled on a rectangular grid.
    ///
    /// # Parameters
    /// * `x`: The x-coordinates. Must be monotonic.
    /// * `y`: The y-coordinates. Must be monotonic.
    /// * `z`: The values `z(x, y)` for each grid point defined by the `x` and `y` coordinates.
    ///
    /// # Panics
    /// Panics when
    /// * dimensions of x/y axis and z-values don't match.
    /// * one axis is empty.
    /// * `x` and `y` values are not monotonic.
    pub fn new(x: Vec<C>, y: Vec<C>, z: ArrayBase<S, Ix2>) -> Self {
        assert_eq!(z.len_of(Axis(0)), x.len(), "x-axis length mismatch.");
        assert_eq!(z.len_of(Axis(1)), y.len(), "y-axis length mismatch.");
        assert!(!x.is_empty());
        assert!(!y.is_empty());

        fn is_monotonic<T: PartialOrd, I: IntoIterator<Item=T>>(i: I) -> bool {
            let mut it = i.into_iter();
            if let Some(prev) = it.next() {
                let mut prev = prev;
                for n in it {
                    if prev <= n {} else {
                        return false;
                    }
                    prev = n;
                }
            }
            true
        }

        assert!(is_monotonic(&x), "x values must be monotonic.");
        assert!(is_monotonic(&y), "x values must be monotonic.");

        Self {
            x,
            y,
            z,
        }
    }

    /// Evaluate the sampled function by interpolation at `(x, y)`.
    ///
    /// If `(x, y)` lies out of the sampled range then the function is silently *extrapolated*.
    /// The boundaries of the sample range can be queried with `bounds()`.
    pub fn eval(&self, (x, y): (C, C)) -> Z {

        // Find closest grid points.
        let (x0, x1) = find_closest_neighbours_indices(&self.x, x);
        let (y0, y1) = find_closest_neighbours_indices(&self.y, y);

        interp2d((x, y),
                 (self.x[x0], self.x[x1]),
                 (self.y[y0], self.y[y1]),
                 (self.z[[x0, y0]], self.z[[x1, y0]], self.z[[x0, y1]], self.z[[x1, y1]]),
        )
    }

    /// Returns the same value as `eval()` as long as `(x, y)` is within the
    /// range of the samples. Otherwise `None` is returned instead of an extrapolation.
    pub fn eval_no_extrapolation(&self, xy: (C, C)) -> Option<Z> {
        if self.is_within_bounds(xy) {
            Some(self.eval(xy))
        } else {
            None
        }
    }

    /// Get the boundaries of the sample range
    /// as `((x0, x1), (y0, y1))` tuple.
    pub fn bounds(&self) -> ((C, C), (C, C)) {
        ((self.x[0], self.x[self.x.len() - 1]),
         (self.y[0], self.y[self.y.len() - 1]))
    }

    /// Check if the `(x, y)` coordinate lies within the defined sample range.
    pub fn is_within_bounds(&self, (x, y): (C, C)) -> bool {
        let ((x0, x1), (y0, y1)) = self.bounds();
        x0 <= x && x <= x1 && y0 <= y && y <= y1
    }

    /// Get the x-coordinate values.
    pub fn xs(&self) -> &Vec<C> {
        &self.x
    }

    /// Get the y-coordinate values.
    pub fn ys(&self) -> &Vec<C> {
        &self.y
    }

    /// Get the raw z values.
    pub fn z(&self) -> &ArrayBase<S, Ix2> {
        &self.z
    }
}

impl<C, Z, S> Interp2D<C, Z, S>
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z>,
          S: Data<Elem=Z> + Clone,
          OwnedRepr<Z>: Data<Elem=Z> {
    /// Swap the input variables.
    pub fn swap_variables(self) -> Interp2D<C, Z, OwnedRepr<Z>> {
        let Self { x, y, z } = self;
        Interp2D::new(y, x, z.t().to_owned())
    }
}


#[test]
fn test_interp2d_on_view() {
    let grid = ndarray::array![
        [0.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ];
    let xs = vec![0.0, 1.0, 2.0];
    let ys = vec![3.0, 4.0, 5.0];

    let interp = Interp2D::new(xs, ys, grid.view());

    // Tolerance for test.
    let tol = 1e-6f64;

    assert!((interp.eval_no_extrapolation((1.0, 4.0)).unwrap() - 1.0).abs() < tol);
}

#[test]
fn test_interp2d() {
    let grid = ndarray::array![
        [0.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ];
    let xs = vec![0.0, 1.0, 2.0];
    let ys = vec![3.0, 4.0, 5.0];

    let interp = Interp2D::new(xs, ys, grid);

    // Tolerance for test.
    let tol = 1e-6f64;

    assert!((interp.eval_no_extrapolation((1.0, 4.0)).unwrap() - 1.0).abs() < tol);
    assert!((interp.eval_no_extrapolation((0.0, 3.0)).unwrap() - 0.0).abs() < tol);
    assert!((interp.eval_no_extrapolation((0.5, 3.5)).unwrap() - 0.25).abs() < tol);

    // // Swap input variables.
    // let interp = interp.swap_variables();
    //
    // assert!((interp.eval_no_extrapolation((4.0, 1.0)).unwrap() - 1.0).abs() < tol);
    // assert!((interp.eval_no_extrapolation((3.0, 0.0)).unwrap() - 0.0).abs() < tol);
    // assert!((interp.eval_no_extrapolation((3.5, 0.5)).unwrap() - 0.25).abs() < tol);
}

