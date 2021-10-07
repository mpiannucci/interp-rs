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

//! One-dimensional linear interpolation.
//!
//! # Examples
//! ```
//! use interp::interp1d::Interp1D;
//!
//! let xs = vec![0.0, 1.0, 2.0];
//! let zs = vec![0.0f64, 1.0, 0.0];
//!
//! let interp = Interp1D::new(xs, zs);
//!
//! assert!((interp.eval_no_extrapolation(1.0).unwrap() - 1.0).abs() < 1e-6);
//! ```

use num_traits::Num;
use std::ops::Mul;
use std::cmp::Ordering;
use crate::find_closest_neighbours_indices;

///
/// * `C`: Coordinate type.
/// * `Z`: Value type.
#[derive(Clone)]
pub struct Interp1D<C, Z> {
    /// Index.
    x: Vec<C>,
    /// Samples.
    z: Vec<Z>,
}

/// Interpolate between two values `x0` and `x1`.
/// `alpha` should range from `0.0` to `1.0` for interpolation, otherwise
/// the value is *extrapolated*.
fn interpolate1d<C, Z>(x0: Z, x1: Z, alpha: C) -> Z
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z>, {
    x0 * (C::one() - alpha) + alpha * x1
}

#[test]
fn test_interpolate1d() {
    assert!((interpolate1d(1.0f64, 2., 0.) - 1.).abs() < 1e-6);
    assert!((interpolate1d(1.0f64, 2., 1.) - 2.).abs() < 1e-6);
    assert!((interpolate1d(1.0f64, 2., 0.5) - 1.5).abs() < 1e-6);
}

/// Find the value of `f(x)`
/// given two sample values `vi = f(xi)` for all `i in [0, 1]`.
fn interp1d<C, Z>(x: C,
                  (x0, x1): (C, C),
                  (v0, v1): (Z, Z)) -> Z
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z> {
    let dx = x1 - x0;

    let alpha = (x - x0) / dx;

    interpolate1d(v0, v1, alpha)
}

impl<C, Z> Interp1D<C, Z>
    where C: Num + Copy + Mul<Z, Output=Z> + PartialOrd,
          Z: Num + Copy + Mul<C, Output=Z>, {
    /// Create a new interpolation engine.
    ///
    /// Interpolates values which are sampled on a rectangular grid.
    ///
    /// # Parameters
    /// * `x`: The x-coordinates. Must be monotonic.
    /// * `z`: The values `z(x)` for each grid point defined by the `x` coordinates.
    ///
    /// # Panics
    /// Panics when
    /// * dimensions of the indices and z-values don't match.
    /// * one axis is empty.
    /// * x values are not monotonic.
    pub fn new(x: Vec<C>, z: Vec<Z>) -> Self {
        assert_eq!(z.len(), x.len(), "x-axis length mismatch.");
        assert!(!x.is_empty());

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

        Self {
            x,
            z,
        }
    }

    /// Evaluate the sampled function by interpolation at `x`.
    ///
    /// If `x` lies out of the sampled range then the function is silently *extrapolated*.
    /// The boundaries of the sample range can be queried with `bounds()`.
    pub fn eval(&self, x: C) -> Z {

        // Find closest grid points.

        let (x0, x1) = find_closest_neighbours_indices(&self.x, x);

        interp1d(x,
                 (self.x[x0], self.x[x1]),
                 (self.z[x0], self.z[x1])
        )
    }

    /// Returns the same value as `eval()` as long as `x` is within the
    /// range of the samples. Otherwise `None` is returned instead of an extrapolation.
    pub fn eval_no_extrapolation(&self, x: C) -> Option<Z> {
        if self.is_within_bounds(x) {
            Some(self.eval(x))
        } else {
            None
        }
    }

    /// Get the boundaries of the sample range
    /// as `(x0, x1)` tuple.
    pub fn bounds(&self) -> (C, C){
        (self.x[0], self.x[self.x.len() - 1])
    }

    /// Check if the `x` coordinate lies within the defined sample range.
    pub fn is_within_bounds(&self, x: C) -> bool {
        let (x0, x1) = self.bounds();
        x0 <= x && x <= x1
    }

    /// Get the x-coordinate values.
    pub fn xs(&self) -> &Vec<C> {
        &self.x
    }

    /// Get the raw z values.
    pub fn z(&self) -> &Vec<Z> {
        &self.z
    }
}

#[test]
fn test_interp1d() {

    let xs = vec![0.0f64, 1.0, 2.0];
    let zs = vec![0.0, 1.0, 0.0];

    let interp = Interp1D::new(xs, zs);

    assert!((interp.eval_no_extrapolation(1.0).unwrap() - 1.0).abs() < 1e-6);
    assert!((interp.eval_no_extrapolation(2.0).unwrap() - 0.0).abs() < 1e-6);
    assert!((interp.eval_no_extrapolation(1.5).unwrap() - 0.5).abs() < 1e-6);
}