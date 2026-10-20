use std::num::NonZero;

use crate::bitstream::BitStream;

pub fn cavlc_encode(matrix: [[i32; 4]; 4], nc: i8) -> (u8, BitStream) {
    let mut stream = BitStream::new();

    let array_form = [
        matrix[0][0],
        matrix[0][1],
        matrix[1][0],
        matrix[2][0],
        matrix[1][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][2],
        matrix[2][1],
        matrix[3][0],
        matrix[3][1],
        matrix[2][2],
        matrix[1][3],
        matrix[2][3],
        matrix[3][2],
        matrix[3][3],
    ];

    let mut total_coeffs = 0;
    let mut zeroes = 0;
    let mut last_non_zero = 0;
    for (idx, &forward_i) in array_form.iter().enumerate() {
        if forward_i != 0 {
            total_coeffs += 1;
            last_non_zero = idx;
        } else {
            zeroes += 1
        }
    }

    let total_coeffs = total_coeffs;

    let mut trailing_ones_budget = 3;
    let mut trailing_ones_mask = 0b0;
    let mut trailing_ones_end = false;
    let coeffs = &array_form[..=last_non_zero];
    let trailing_zeroes = 16 - coeffs.len();
    let mut levels = Vec::new();
    for &coeff in coeffs.iter().rev() {
        if (coeff == 1 || coeff == -1) && (!trailing_ones_end && trailing_ones_budget != 0) {
            trailing_ones_budget -= 1;
            trailing_ones_mask |= (coeff as u32) >> (31 - trailing_ones_budget)
        } else if coeff == 0 {
            // Ignore 0's
        } else {
            levels.push(coeff);
            trailing_ones_end = true;
        }
    }

    let total_zeroes = zeroes - trailing_zeroes;
    let trailing_ones = 3 - trailing_ones_budget;

    let (bits, width) = coeff_mapping_new(total_coeffs, trailing_ones, nc);
    for i in (0..width).rev() {
        stream.push(bits >> i & 1 != 0);
    }
    if total_coeffs == 0 {
        return (0, stream);
    }
    for i in (0..trailing_ones).rev() {
        stream.push(trailing_ones_mask >> i & 1 != 0);
    }

    let mut suffix_length = if total_coeffs > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };

    /*
       Source:
       An Efficient Software Implementation of the
       CAVLC Encoder for H.264/AVC

       Marc P. Hoffman, Eric J. Balster Member, IEEE, Frank Scarpino and Kerry Hill
       School of Electrical and Computer Engineering University of Dayton
    */
    for (_, level) in levels.into_iter().enumerate() {
        let level_code = 2 * level.abs() - if level < 0 { 1 } else { 2 };
        let t = if suffix_length == 0 {
            7
        } else {
            15 * (1 << suffix_length - 1)
        };
        if level > t {
            todo!()
        } else {
            let level_prefix = level_code >> suffix_length;
            let level_suffix_size = suffix_length;
            let level_suffix = level_code - (level_prefix << suffix_length);
            for _ in (0..level_prefix).rev() {
                stream.push(false);
            }
            stream.push(true);
            for i in (0..level_suffix_size).rev() {
                stream.push(level_suffix >> i & 1 != 0);
            }
        }

        if suffix_length == 0 {
            suffix_length = 1;
        }

        let t = 3 << suffix_length - 1;
        if level > t {
            suffix_length += 1;
            if suffix_length > 6 {
                suffix_length = 6
            }
        }
    }

    if total_coeffs != 16 {
        let (bits, len) = count_zeroes(total_zeroes as u8, total_coeffs);
        for i in (0..len).rev() {
            stream.push(bits >> i & 1 != 0);
        }
    }

    let mut zeroes_left = total_zeroes as u8;

    let mut rem = array_form[..=last_non_zero].iter().rev().peekable();
    'outer: while let Some(&i) = rem.next() {
        if zeroes_left == 0 {
            break;
        }

        assert!(i != 0);
        let mut run_before = 0;
        loop {
            if let Some(&&peak) = rem.peek() {
                if peak == 0 {
                    run_before += 1;
                    // Consume the zero
                    rem.next();
                    continue;
                }
                break;
            }
            break 'outer;
        }

        let (bit, width) = run_before_table(run_before, zeroes_left);
        zeroes_left -= run_before;
        let width = width as usize;
        for i in (0..width).rev() {
            stream.push(bit >> i & 1 != 0);
        }
    }

    (total_coeffs, stream)
}

fn ilog2_unchecked(num: u8) -> u8 {
    let zeroes = num.leading_zeros() as u8;
    7 - zeroes
}

fn count_zeroes(total_zeroes: u8, total_coeffs: u8) -> (u16, u8) {
    const LUT: [[(u16, u8); 15]; 16] = [
        [
            (0b1, 1),
            (0b111, 3),
            (0b0101, 4),
            (0b00011, 5),
            (0b0101, 4),
            (0b000001, 6),
            (0b000001, 6),
            (0b000001, 6),
            (0b000001, 6),
            (0b000001, 6),
            (0, 4),
            (0, 4),
            (0, 3),
            (0, 2),
            (0, 1),
        ],
        [
            (0b011, 3),
            (0b110, 3),
            (0b111, 3),
            (0b111, 3),
            (0b0100, 4),
            (0b00001, 5),
            (0b00001, 5),
            (0b0001, 4),
            (0, 6),
            (0, 5),
            (0b0001, 4),
            (0b0001, 4),
            (0b001, 3),
            (0b01, 2),
            (0b1, 1),
        ],
        [
            (0b010, 3),
            (0b101, 3),
            (0b110, 3),
            (0b0101, 4),
            (0b0011, 4),
            (0b111, 3),
            (0b101, 3),
            (0b0001, 5),
            (0b0001, 4),
            (0b001, 3),
            (0b001, 3),
            (0b01, 2),
            (0b1, 1),
            (0b1, 1),
            (0, 0),
        ],
        [
            (0b0011, 4),
            (0b011, 3),
            (0b0100, 4),
            (0b110, 3),
            (0b110, 3),
            (0b101, 3),
            (0b101, 3),
            (0b100, 3),
            (0b011, 3),
            (0b11, 2),
            (0b010, 3),
            (0b1, 1),
            (0b01, 2),
            (0, 0),
            (0, 0),
        ],
        [
            (0b0010, 4),
            (0b011, 3),
            (0b0100, 4),
            (0b110, 3),
            (0b110, 3),
            (0b101, 3),
            (0b011, 3),
            (0b11, 2),
            (0b10, 2),
            (0b10, 2),
            (0b1, 1),
            (0b001, 3),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b00011, 5),
            (0b0101, 4),
            (0b0011, 4),
            (0b101, 3),
            (0b101, 3),
            (0b100, 3),
            (0b11, 2),
            (0b10, 2),
            (0b001, 3),
            (0b01, 2),
            (0b011, 3),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b00010, 5),
            (0b0100, 4),
            (0b100, 3),
            (0b100, 3),
            (0b100, 3),
            (0b011, 3),
            (0b010, 3),
            (0b010, 3),
            (0b01, 2),
            (0b0001, 4),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b000011, 6),
            (0b0011, 4),
            (0b011, 3),
            (0b0011, 4),
            (0b011, 3),
            (0b010, 3),
            (0b0001, 4),
            (0b001, 3),
            (0b00001, 5),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b000010, 6),
            (0b0010, 4),
            (0b0010, 4),
            (0b011, 3),
            (0b0010, 4),
            (0b0001, 4),
            (0b001, 3),
            (0b000000, 6),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b0000011, 7),
            (0b00011, 5),
            (0b00011, 5),
            (0b0010, 4),
            (0b00001, 5),
            (0b001, 3),
            (0b000000, 6),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b0000010, 7),
            (0b00010, 5),
            (0b00010, 5),
            (0b00010, 5),
            (0b0001, 4),
            (0b000000, 6),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b00000011, 8),
            (0b000011, 6),
            (0b000001, 6),
            (0b00001, 5),
            (0b00000, 5),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b00000010, 8),
            (0b000010, 6),
            (0b00001, 5),
            (0b00000, 5),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b000000011, 9),
            (0b000001, 6),
            (0b000000, 6),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b000000010, 9),
            (0b000000, 6),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [
            (0b000000001, 9),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
    ];

    // Map to 0..15 rather than 1..=15
    let total_coeffs = total_coeffs - 1;

    LUT[total_zeroes as usize][total_coeffs as usize]
}

// Optimized run before table based on Table 9-10 of the H.264 spec
fn run_before_table(run_before: u8, zeroes_left: u8) -> (u16, u8) {
    if zeroes_left == 0 {
        panic!()
    }

    match (run_before, zeroes_left) {
        (6, 6) => (0b100, 3),
        // When X == Y (excluding pair (6, 6)), ilog2(x) + 1 zeroes are added to the bit stream
        (x, y) if x == y && y < 6 => (0, ilog2_unchecked(y) + 1),
        (0, 1..=2) => (0b1, 1),
        (0, 3..=6) => (0b11, 2),
        (1, 1) => (0, 1),
        (1, 2) => (0b01, 2),
        (1, 3..=5) => (0b10, 2),
        (1, 6) => (0b000, 3),
        (2, 3) | (2, 4) => (0b01, 2),
        (2, 5) => (0b011, 3),
        (2, 6) => (0b001, 3),
        (3, 4) => (0b001, 3),
        (3, 5) => (0b010, 3),
        (3, 6) => (0b011, 3),
        (4, 5) => (0b001, 3),
        (4, 6) => (0b010, 3),
        (5, 6) => (0b101, 3),
        // When `run_before` is less than 6 and `zeroes_left` is greater than 6, `!run_before & 0b111` is added to the bit stream
        (x @ ..=5, 7..) => ((!x as u16) & 0b111, 3),
        (x @ 6.., 7..) => (0b1, x - 3),
        (x, y) => unreachable!("{x} {y}"),
    }
}

const fn count_binary_literal(bytes: &[u8]) -> u8 {
    // Ignore 0b if present
    let mut i = if bytes[0] == b'0' && bytes[1] == b'b' {
        2
    } else {
        0
    };
    let mut bits = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'1' | b'0' => bits += 1,
            b'_' | b' ' => {
                // '_' and ' ' is allowed for fomatting purposes
            }
            _ => {}
        }
        i += 1;
    }

    bits
}

const _: () = assert!(count_binary_literal(b"0b0000_0011_1") == 9);
const _: () = assert!(count_binary_literal(b"0b0000_111") == 7);
const _: () = assert!(count_binary_literal(b"0b0001_11") == 6);
const _: () = assert!(count_binary_literal(b"0b0") == 1);

macro_rules! lit_len {
    (-) => {
        (0, 0)
    };
    ($lit:literal) => {
        ($lit, count_binary_literal(stringify!($lit).as_bytes()))
    };
}

/// The value of `nc` should not be greater than 16, or less than -2. This is defined in the H.264 spec.
/// `total_coeffs` is in range 0..=16 
/// `trailing_ones` is in range 0..=3
/// 
/// The output of this is a 16 bit binary number from Table 9-5 of the H.264 spec. 
pub fn coeff_mapping_new(total_coeffs: u8, trailing_ones: u8, nc: i8) -> (u16, u8) {
    #[rustfmt::skip]
    const COEFF_LUT: [[[(u16, u8); 6]; 4]; 17] = [
        [
            [lit_len!(0b1), lit_len!(0b11), lit_len!(0b1111), lit_len!(0b0000_11), lit_len!(0b01), lit_len!(0b1)],
            [(0, 0); 6],
            [(0, 0); 6],
            [(0, 0); 6],
        ],
        [
            [lit_len!(0b0001_01), lit_len!(0b0010_11), lit_len!(0b0011_11), lit_len!(0b0000_00), lit_len!(0b0001_11), lit_len!(0b0001_111)],
            [lit_len!(0b01), lit_len!(0b10), lit_len!(0b1110), lit_len!(0b0000_01), lit_len!(0b1), lit_len!(0b01)],
            [(0, 0); 6],
            [(0, 0); 6],
        ],
        [
            [lit_len!(0b0000_0111), lit_len!(0b0001_11), lit_len!(0b0010_11), lit_len!(0b0001_00), lit_len!(0b0001_00), lit_len!(0b0001_110)],
            [lit_len!(0b0001_00), lit_len!(0b0011_1), lit_len!(0b0111_1), lit_len!(0b0001_01), lit_len!(0b0001_10), lit_len!(0b0001_101)],
            [lit_len!(0b001), lit_len!(0b011), lit_len!(0b1101), lit_len!(0b0001_10), lit_len!(0b001), lit_len!(0b001)],
            [(0, 0); 6],
        ],
        [
            [lit_len!(0b0000_0011_1), lit_len!(0b0000_111), lit_len!(0b0010_00), lit_len!(0b0010_00), lit_len!(0b0000_11), lit_len!(0b0000_0011_1)],
            [lit_len!(0b0000_0110), lit_len!(0b0010_10), lit_len!(0b0110_0), lit_len!(0b0010_01), lit_len!(0b0000_011), lit_len!(0b0001_100)],
            [lit_len!(0b0000_101), lit_len!(0b0010_01), lit_len!(0b0111_0), lit_len!(0b0010_10), lit_len!(0b0000_010), lit_len!(0b0001_011)],
            [lit_len!(0b0001_1), lit_len!(0b0101), lit_len!(0b1100), lit_len!(0b0010_11), lit_len!(0b0001_01), lit_len!(0b0000_1)],
        ],
        [
            [lit_len!(0b0000_0001_11), lit_len!(0b0000_0111), lit_len!(0b0001_111), lit_len!(0b0011_00), lit_len!(0b0000_10), lit_len!(0b0000_0011_0)],
            [lit_len!(0b0000_0011_0), lit_len!(0b0001_10), lit_len!(0b0101_0), lit_len!(0b0011_01), lit_len!(0b0000_0011), lit_len!(0b0000_0010_1)],
            [lit_len!(0b0000_0101), lit_len!(0b0001_01), lit_len!(0b0101_1), lit_len!(0b0011_10), lit_len!(0b0000_0010), lit_len!(0b0001_010)],
            [lit_len!(0b0000_11), lit_len!(0b0100), lit_len!(0b1011), lit_len!(0b0011_11), lit_len!(0b0000_000), lit_len!(0b0000_01)],
        ],
        [
            [lit_len!(0b0000_0000_111), lit_len!(0b0000_0100), lit_len!(0b0001_011), lit_len!(0b0100_00), lit_len!(-), lit_len!(0b0000_0001_11)],
            [lit_len!(0b0000_0001_10), lit_len!(0b0000_110), lit_len!(0b0100_0), lit_len!(0b0100_01), lit_len!(-), lit_len!(0b0000_0001_10)],
            [lit_len!(0b0000_0010_1), lit_len!(0b0000_101), lit_len!(0b0100_1), lit_len!(0b0100_10), lit_len!(-), lit_len!(0b0000_0010_0)],
            [lit_len!(0b0000_100), lit_len!(0b0011_0), lit_len!(0b1010), lit_len!(0b0100_11), lit_len!(-), lit_len!(0b0001_001)],
        ],
        [
            [lit_len!(0b0000_0000_0111_1), lit_len!(0b0000_0011_1), lit_len!(0b0001_001), lit_len!(0b0101_00), lit_len!(-), lit_len!(0b0000_0000_111)],
            [lit_len!(0b0000_0000_110), lit_len!(0b0000_0110), lit_len!(0b0011_10), lit_len!(0b0101_01), lit_len!(-), lit_len!(0b0000_0000_110)],
            [lit_len!(0b0000_0001_01), lit_len!(0b0000_0101), lit_len!(0b0011_01), lit_len!(0b0101_10), lit_len!(-), lit_len!(0b0000_0001_01)],
            [lit_len!(0b0000_0100), lit_len!(0b0010_00), lit_len!(0b1001), lit_len!(0b0101_11), lit_len!(-), lit_len!(0b0001_000)],
        ],
        [
            [lit_len!(0b0000_0000_0101_1), lit_len!(0b0000_0001_111), lit_len!(0b0001_000), lit_len!(0b0110_00), lit_len!(-), lit_len!(0b0000_0000_0111)],
            [lit_len!(0b0000_0000_0111_0), lit_len!(0b0000_0011_0), lit_len!(0b0010_10), lit_len!(0b0110_01), lit_len!(-), lit_len!(0b0000_0000_0110)],
            [lit_len!(0b0000_0000_101), lit_len!(0b0000_0010_1), lit_len!(0b0010_01), lit_len!(0b0110_10), lit_len!(-), lit_len!(0b0000_0000_101)],
            [lit_len!(0b0000_0010_0), lit_len!(0b0001_00), lit_len!(0b1000), lit_len!(0b0110_11), lit_len!(-), lit_len!(0b0000_0001_00)],
        ],
        [
            [lit_len!(0b0000_0000_0100_0), lit_len!(0b0000_0001_011), lit_len!(0b0000_1111), lit_len!(0b111_00), lit_len!(-), lit_len!(0b0000_0000_0011_1)],
            [lit_len!(0b0000_0000_0101_0), lit_len!(0b0000_0001_110), lit_len!(0b0001_110), lit_len!(0b0111_01), lit_len!(-), lit_len!(0b0000_0000_0101)],
            [lit_len!(0b0000_0000_0110_1), lit_len!(0b0000_0001_101), lit_len!(0b0001_101), lit_len!(0b0111_10), lit_len!(-), lit_len!(0b0000_0000_0100)],
            [lit_len!(0b0000_0001_00), lit_len!(0b0000_100), lit_len!(0b0110_1), lit_len!(0b0111_11), lit_len!(-), lit_len!(0b0000_0000_100)],
        ],
        [
            [lit_len!(0b0000_0000_0011_11), lit_len!(0b0000_0000_1111), lit_len!(0b0000_1011), lit_len!(0b1000_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0011_10), lit_len!(0b0000_0001_010), lit_len!(0b0000_1110), lit_len!(0b1000_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0100_1), lit_len!(0b0000_0001_001), lit_len!(0b0001_010), lit_len!(0b1000_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_100), lit_len!(0b0000_0010_0), lit_len!(0b0011_00), lit_len!(0b1000_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0010_11), lit_len!(0b0000_0000_1011), lit_len!(0b0000_0111_1), lit_len!(0b1001_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0010_10), lit_len!(0b0000_0000_1110), lit_len!(0b0000_1010), lit_len!(0b1001_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0011_01), lit_len!(0b0000_0000_1101), lit_len!(0b0000_1101), lit_len!(0b1001_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0110_0), lit_len!(0b0000_0001_100), lit_len!(0b0001_100), lit_len!(0b0001_100), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0001_111), lit_len!(0b0000_0000_1000), lit_len!(0b0000_0101_1), lit_len!(0b1010_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_110), lit_len!(0b0000_0000_1010), lit_len!(0b0000_0111_0), lit_len!(0b1010_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0010_01), lit_len!(0b0000_0000_1001), lit_len!(0b0000_1001), lit_len!(0b1010_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0011_00), lit_len!(0b0000_0001_000), lit_len!(0b0000_1100), lit_len!(0b1010_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0001_011), lit_len!(0b0000_0000_0111_1), lit_len!(0b0000_0100_0), lit_len!(0b1011_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_010), lit_len!(0b0000_0000_0111_0), lit_len!(0b0000_0101_0), lit_len!(0b1011_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_101), lit_len!(0b0000_0000_0110_1), lit_len!(0b0000_0110_1), lit_len!(0b1011_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0010_00), lit_len!(0b0000_0000_1100), lit_len!(0b0000_1000), lit_len!(0b1011_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0000_1111), lit_len!(0b0000_0000_0101_1), lit_len!(0b0000_0011_01), lit_len!(0b1100_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_001), lit_len!(0b0000_0000_0101_0), lit_len!(0b0000_0011_1), lit_len!(0b1100_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_001), lit_len!(0b0000_0000_0100_1), lit_len!(0b0000_0100_1), lit_len!(0b1100_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_100), lit_len!(0b0000_0000_0110_0), lit_len!(0b0000_0110_0), lit_len!(0b1100_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0000_1011), lit_len!(0b0000_0000_0011_1), lit_len!(0b0000_0010_01), lit_len!(0b1101_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1110), lit_len!(0b0000_0000_0010_11), lit_len!(0b0000_0011_00), lit_len!(0b1101_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1101), lit_len!(0b0000_0000_0011_0), lit_len!(0b0000_0010_11), lit_len!(0b1101_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0001_000), lit_len!(0b0000_0000_0100_0), lit_len!(0b0000_0010_10), lit_len!(0b1101_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0000_0111), lit_len!(0b0000_0000_0010_01), lit_len!(0b0000_0001_01), lit_len!(0b1110_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1010), lit_len!(0b0000_0000_0010_00), lit_len!(0b0000_0010_00), lit_len!(0b1110_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1001), lit_len!(0b0000_0000_0010_10), lit_len!(0b0000_0001_11), lit_len!(0b1110_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1100), lit_len!(0b0000_0000_0000_1), lit_len!(0b0000_0001_10), lit_len!(0b1110_11), lit_len!(-), lit_len!(-)],
        ],
        [
            [lit_len!(0b0000_0000_0000_0100), lit_len!(0b0000_0000_0001_11), lit_len!(0b0000_0000_01), lit_len!(0b1111_00), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_0110), lit_len!(0b0000_0000_0001_10), lit_len!(0b0000_0001_00), lit_len!(0b1111_01), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_0101), lit_len!(0b0000_0000_0001_01), lit_len!(0b0000_0000_11), lit_len!(0b1111_10), lit_len!(-), lit_len!(-)],
            [lit_len!(0b0000_0000_0000_1000), lit_len!(0b0000_0000_0001_00), lit_len!(0b0000_0000_10), lit_len!(0b1111_11), lit_len!(-), lit_len!(-)],
        ],
    ];

    // Converts `nc` to the value of it's index inside the LUT.
    let nc_idx = match nc {
        ..=-2 => 5,
        -1 => 4,
        0..2 => 0,
        2..4 => 1,
        4..8 => 2,
        8.. => 3,
    };

    COEFF_LUT[total_coeffs as usize][trailing_ones as usize][nc_idx]
}
