use std::array;

use crate::{Frame, frame::Array2D};

pub fn motion_esimtation(
    previous: &Frame,
    current: &Frame,
    block_height: usize,
    block_width: usize,
    max_block_width: usize,
    max_block_height: usize,
) -> Option<(isize, isize, [[u8; 16]; 16], [[u8; 16]; 16])> {
    // Window in pixels
    const WINDOW: usize = 16;

    /// Returns an H.264 macroblock from the given frame
    fn macro_block(
        frame: &Array2D<'_>,
        width_offset: usize,
        heigh_offset: usize,
    ) -> [[u8; 16]; 16] {
        array::from_fn(|i| {
            *(&frame[i + heigh_offset][width_offset..width_offset + WINDOW])
                .as_array()
                .unwrap()
        })
    }

    let mut sums: [[u32; 32]; 32] = [[u32::MAX; _]; _];
    let pixel_width = block_width * 16;
    let pixel_height = block_height * 16;

    let current = macro_block(&current.y(), pixel_width, pixel_height);
    let prev_block = macro_block(&previous.y(), pixel_width, pixel_height);
    if current == prev_block {
        return None;
    }

    // Iteration offset
    let i_offset = if block_width == 0 { 16 } else { 0 };
    let j_offset = if block_height == 0 { 16 } else { 0 };

    let width_upper_bound = if block_width == max_block_width {
        16
    } else {
        32
    };
    let height_upper_bound = if block_height == max_block_height {
        16
    } else {
        32
    };

    for i in i_offset..width_upper_bound {
        for j in j_offset..height_upper_bound {
            // println!("Finding blocks: {} {} {}", pixel_width, i, i_offset);
            // Offset back by 16, as we want to look behind and ahead
            let block = macro_block(&previous.y(), pixel_width + i - 16, pixel_height + j - 16);

            sums[i][j] = block
                .iter()
                .flatten()
                .zip(current.iter().flatten())
                .map(|(&a, &b)| {
                    let a = a as u32;
                    let b = b as u32;
                    u32::max(a, b) - u32::min(b, a)
                })
                .sum();
        }
    }

    let mut prev_sum = u32::MAX;
    let mut mv_y = 0;
    let mut mv_x = 0;

    for (height, sums) in sums.iter().enumerate() {
        for (width, &sum) in sums.iter().enumerate() {
            if sum < prev_sum {
                prev_sum = sum;
                mv_y = width as isize - 16;
                mv_x = height as isize - 16;
            }
        }
    }

    // println!("{} {} {} {} {} {}", mv_x, mv_y, pixel_width, pixel_height, pixel_width.wrapping_add_signed(mv_x), pixel_height.wrapping_add_signed(mv_y));
    let block = macro_block(
        &previous.y(),
        pixel_width.wrapping_add_signed(mv_x),
        pixel_height.wrapping_add_signed(mv_y),
    );

    Some((mv_x, mv_y, block, current))
}

#[cfg(test)]
mod tests {
    use super::motion_esimtation;
    use crate::Frame;

    #[test]
    fn motion_estimate_over_one_down_one() {
        let before: &[u8; 2048] = include_bytes!("me_samples/b.yuv");
        let after: &[u8; 2048] = include_bytes!("me_samples/a.yuv");

        let mut prev = Frame::new(&[0; 32 * 32], &[0; 16 * 16], &[0; 16 * 16], 32, 32);
        let mut curr = Frame::new(&[0; 32 * 32], &[0; 16 * 16], &[0; 16 * 16], 32, 32);
        for i in 0..32 {
            prev.y_mut()[i].copy_from_slice(&before[i * 32..i * 32 + 32]);
            curr.y_mut()[i].copy_from_slice(&after[i * 32..i * 32 + 32]);
        }

        let val = motion_esimtation(&prev, &curr, 0, 0, 1, 1);

        assert!(val.is_some());
        let (x, y, _, _) = val.unwrap();
        assert_eq!(x, 1, "Expected 1, got {x}");
        assert_eq!(y, 1, "Expected 1, got {y}");
    }

    #[test]
    fn motion_estimate_over_two_down_two() {
        let before: &[u8; 2048] = include_bytes!("me_samples/c.yuv");
        let after: &[u8; 2048] = include_bytes!("me_samples/a.yuv");

        let mut prev = Frame::new(&[0; 32 * 32], &[0; 16 * 16], &[0; 16 * 16], 32, 32);
        let mut curr = Frame::new(&[0; 32 * 32], &[0; 16 * 16], &[0; 16 * 16], 32, 32);
        for i in 0..32 {
            prev.y_mut()[i].copy_from_slice(&before[i * 32..i * 32 + 32]);
            curr.y_mut()[i].copy_from_slice(&after[i * 32..i * 32 + 32]);
        }

        let val = motion_esimtation(&prev, &curr, 0, 0, 1, 1);

        assert!(val.is_some());
        let (x, y, _, _) = val.unwrap();
        assert_eq!(x, 2, "Expected 2, got {x}");
        assert_eq!(y, 2, "Expected 2, got {y}");
    }
}
