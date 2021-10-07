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

//! Interpolation of one and two dimensional data.

#![deny(missing_docs)]


use std::cmp::Ordering;

pub mod interp1d;
pub mod interp2d;

/// Find closest grid points.
fn find_closest_neighbours_indices<C>(v: &Vec<C>, x: C) -> (usize, usize)
    where C: PartialOrd {
    let idx = match v.binary_search_by(|a| {
        if *a < x {
            Ordering::Less
        } else if *a == x {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }) {
        Ok(i) => i,
        Err(i) => i.max(1)-1,
    };

   if idx == v.len() - 1 {
        (idx - 1, idx)
    } else {
        (idx, idx + 1)
    }
}

#[test]
fn test_find_closest_neighbours_indices() {
    let v = vec![0., 1., 2.];
    assert_eq!(find_closest_neighbours_indices(&v, -0.1), (0, 1));
    assert_eq!(find_closest_neighbours_indices(&v, 0.), (0, 1));
    assert_eq!(find_closest_neighbours_indices(&v, 0.01), (0, 1));
    assert_eq!(find_closest_neighbours_indices(&v, 0.99), (0, 1));
    assert_eq!(find_closest_neighbours_indices(&v, 1.0), (1, 2));
    assert_eq!(find_closest_neighbours_indices(&v, 1.99), (1, 2));
    assert_eq!(find_closest_neighbours_indices(&v, 2.99), (1, 2));
}