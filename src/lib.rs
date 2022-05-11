// Copyright (c) 2021-2021 Thomas Kramer.
// SPDX-FileCopyrightText: 2022 Thomas Kramer <code@tkramer.ch>
//
// SPDX-License-Identifier: GPL-3.0-or-later

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