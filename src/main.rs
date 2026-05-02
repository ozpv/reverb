#![feature(iter_array_chunks)]

use itertools::Itertools;
use rustfft::{FftPlanner, num_complex::Complex};
use std::{fs, path::Path, thread, time::Instant};

const NORMALIZE_TARGET_DB: f64 = -1.0;

struct Data {
    header: Vec<u8>,
    left: Vec<i32>,
    right: Vec<i32>,
}

fn read_32_bit_stereo_pcm_wav(file: impl AsRef<Path>) -> std::io::Result<(Data, usize)> {
    let bytes = fs::read(file)?;
    let data_offset = bytes.windows(4).position(|s| s == b"data").unwrap();

    let mut bytes_iter = bytes.into_iter();

    let header = bytes_iter
        .by_ref()
        .take(data_offset + 8)
        .collect::<Vec<u8>>();

    let (left, right) = bytes_iter
        .array_chunks::<4>()
        .map(i32::from_le_bytes)
        // (l, r), (l, r), (l, r), ...
        .tuples::<(i32, i32)>()
        // Vec<(l, r)> into (Vec<l>, Vec<r>)
        .unzip::<i32, i32, Vec<i32>, Vec<i32>>();

    Ok((
        Data {
            header,
            left,
            right,
        },
        data_offset,
    ))
}

fn write_32_bit_stereo_samples_as_pcm_wav(
    output_file: impl AsRef<Path>,
    mut header: Vec<u8>,
    left: Vec<i32>,
    right: Vec<i32>,
    data_offset: usize,
) -> std::io::Result<()> {
    let mut data = left
        .into_iter()
        .interleave(right)
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<u8>>();

    // update data length
    header[data_offset + 4..data_offset + 8].copy_from_slice(&u32::to_le_bytes(data.len() as u32));

    println!(
        "Bytes len: {} Byte rate: {}",
        data.len(),
        (44_100 * 2 * 32) / 8
    );

    header.append(&mut data);

    fs::write(output_file, header)
}

fn forward_real_fft(
    signal: Vec<i32>,
    fft_planner: &mut FftPlanner<f64>,
    len: usize,
) -> Vec<Complex<f64>> {
    let mut signal = signal
        .into_iter()
        .map(|sample| Complex::<f64>::new(f64::from(sample), 0.0))
        .collect::<Vec<Complex<f64>>>();

    let fft = fft_planner.plan_fft_forward(len);
    fft.process(&mut signal);

    signal
}

fn inverse_real_fft(
    mut signal: Vec<Complex<f64>>,
    fft_planner: &mut FftPlanner<f64>,
    len: usize,
) -> Vec<f64> {
    let fft = fft_planner.plan_fft_inverse(len);
    fft.process(&mut signal);

    signal
        .into_iter()
        .map(|complex_num| complex_num.re)
        .collect::<Vec<f64>>()
}

/// normalizes and casts to i32
fn finalize(signal: Vec<f64>, target_db: f64) -> Vec<i32> {
    let max = signal.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&1.0);

    let scalar = 10.0_f64.powf(target_db / 20.0) / max;

    #[allow(clippy::cast_possible_truncation)]
    signal
        .into_iter()
        .map(|sample| {
            let sample = sample * scalar * f64::from(i32::MAX);
            sample as i32
        })
        .collect::<Vec<i32>>()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();

    let (mut impulse, data_offset) = read_32_bit_stereo_pcm_wav("impulse.wav")?;
    let (mut impulse_response, _) = read_32_bit_stereo_pcm_wav("impulse_response.wav")?;

    // assuming both channels are the same length
    // next power of two because of divide and conquer algorithms
    let output_len = (impulse.left.len() + impulse_response.left.len() - 1).next_power_of_two();

    // calculate left channel
    let handle = thread::spawn(move || {
        let mut fft_planner = FftPlanner::new();

        impulse.left.resize(output_len, 0);
        impulse_response.left.resize(output_len, 0);

        let impulse_left_f = forward_real_fft(impulse.left, &mut fft_planner, output_len);
        let impulse_response_left_f =
            forward_real_fft(impulse_response.left, &mut fft_planner, output_len);

        let output_y = impulse_left_f
            .into_iter()
            .zip(impulse_response_left_f)
            .map(|(impulse, impulse_response)| impulse * impulse_response)
            .collect::<Vec<Complex<f64>>>();

        finalize(
            inverse_real_fft(output_y, &mut fft_planner, output_len),
            NORMALIZE_TARGET_DB,
        )
    });

    // at the exact same time, calculate right channel
    let mut fft_planner = FftPlanner::new();

    impulse.right.resize(output_len, 0);
    impulse_response.right.resize(output_len, 0);

    let impulse_right_f = forward_real_fft(impulse.right, &mut fft_planner, output_len);
    let impulse_response_right_f =
        forward_real_fft(impulse_response.right, &mut fft_planner, output_len);

    let output_y = impulse_right_f
        .into_iter()
        .zip(impulse_response_right_f)
        .map(|(impulse, impulse_response)| impulse * impulse_response)
        .collect::<Vec<Complex<f64>>>();

    let right_out = finalize(
        inverse_real_fft(output_y, &mut fft_planner, output_len),
        NORMALIZE_TARGET_DB,
    );

    // wait for left to finish if not already done
    // and move final data back into this scope
    let left_out = handle.join().unwrap();

    write_32_bit_stereo_samples_as_pcm_wav(
        "output.wav",
        impulse.header,
        left_out,
        right_out,
        data_offset,
    )?;

    println!("{:?}", now.elapsed());

    Ok(())
}
