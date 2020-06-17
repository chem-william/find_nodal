extern crate ndarray;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
pub fn get_info(
    cube_data: Vec<f64>,
    xyz_vec: Vec<f64>,
    shape: Vec<i32>,
) -> PyResult<Vec<Vec<f64>>> {
    let mut total: usize = 0;

    let mut all_info: Vec<Vec<f64>> = Vec::with_capacity(cube_data.len());

    let x: f64 = xyz_vec[0];
    let y: f64 = xyz_vec[1];
    let z: f64 = xyz_vec[2];

    for ix in 0..shape[0] {
        let tmp_x: f64 = x * ix as f64;

        for iy in 0..shape[1] {
            let tmp_y: f64 = y * iy as f64;

            for iz in 0..shape[2] {
                let tmp_z: f64 = z * iz as f64;

                all_info.push(
                    vec![tmp_x, tmp_y, tmp_z, cube_data[total]]
                );

                total += 1;
            }
        }
    }

    Ok(all_info)
}

#[pymodule]
fn libread_cube(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_info)).unwrap();

    Ok(())
}
