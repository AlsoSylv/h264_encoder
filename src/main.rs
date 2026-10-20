use std::{array, fs::File, io::Write, process::exit, sync::Once};

use crate::{
    bitstream::BitStream, cavlc::cavlc_encode, frame::Array2D, motion_estimation::motion_esimtation,
};

pub mod bitstream;
mod cavlc;
mod frame;
mod matricies;
mod motion_estimation;
mod vectors;

use matricies::*;

/// Every block in H.264 starts with 0x0 0x0 0x0 0x1
const HEADER: [u8; 4] = [0x0, 0x0, 0x0, 0x1];

use frame::Frame;

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum NalRefIdc {
    Zero,
    One,
    Two,
    Three,
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum NalUnitType {
    Unspecified = 0,
    CodedSliceOfANonIdrPicture = 1,
    CodedSliceDataPartitionA = 2,
    CodedSliceDataPartitionB = 3,
    CodedSliceDataPartitionC = 4,
    CodedSliceOfAnIdrPicture = 5,
    SupplementalEnhancementInformation = 6,
    SequenceParameterSet = 7,
    PictureParameterSet = 8,
}

pub struct NalUnitHeader {
    ref_idc: NalRefIdc,
    unit_type: NalUnitType,
}

impl NalUnitHeader {
    pub fn encode_header(&self) -> u8 {
        0b00000000 | (self.ref_idc as u8) << 5 | self.unit_type as u8
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), std::io::Error> {
        writer.write_all(&HEADER)?;
        writer.write_all(&[self.encode_header()])
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum SliceType {
    P = 0,
    B = 1,
    I = 7,
    SP = 3,
    SI = 4,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum MacroBlockType {
    /// P_L0_16x16
    I_NxN,
    /// P_L0_L0_16x8
    I_16x16_0_0_0,
    /// P_L0_L0_8x16
    I_16x16_1_0_0,
    /// P_8x8
    I_16x16_2_0_0,
    /// P_8x8ref0
    I_16x16_3_0_0,
    I_16x16_0_1_0,
    I_16x16_1_1_0,
    I_16x16_2_1_0,
    I_16x16_3_1_0,
    I_16x16_0_2_0,
    I_16x16_1_2_0,
    I_16x16_2_2_0,
    I_16x16_3_2_0,
    I_16x16_0_0_1,
    I_16x16_1_0_1,
    I_16x16_2_0_1,
    I_16x16_3_0_1,
    I_16x16_0_1_1,
    I_16x16_1_1_1,
    I_16x16_2_1_1,
    I_16x16_3_1_1,
    I_16x16_0_2_1,
    I_16x16_1_2_1,
    I_16x16_2_2_1,
    I_16x16_3_2_1,
    I_PCM,
}

pub struct SliceHeader<'a> {
    first_mb_in_slice: u32,
    slice_type: SliceType,
    sequence_parameter_set_id: u32,
    pic_parameter_set_id: u32,
    frame_num: u32,
    idr_pic_id: u32,
    pic_order_cnt_lsb: u32,
    first_mb_type: MacroBlockType,
    data: Option<&'a [SliceData]>,
}

#[derive(Debug)]
pub enum SliceData {
    Skip(u32),
    Block {
        mv_x: i32,
        mv_y: i32,
        luma: [[i32; 16]; 16],
    },
}

impl SliceHeader<'_> {
    pub fn to_bit_stream(
        &self,
        active_pps: &PicParameterSet,
        active_sps: &SequenceParameterSet,
    ) -> BitStream {
        let mut stream = BitStream::new();
        stream.extend(exponential_golomb_encode(self.first_mb_in_slice));
        stream.extend(exponential_golomb_encode(self.slice_type as u32));
        stream.extend(exponential_golomb_encode(active_pps.pic_parameter_set_id));
        // frame_num
        let bits = active_sps.log2_max_frame_num_minus4 + 4;
        for idx in (0..bits).rev() {
            stream.push(self.frame_num >> idx & 1 == 1);
        }
        if self.slice_type == SliceType::I {
            stream.extend(exponential_golomb_encode(self.idr_pic_id));
        }
        if self.slice_type == SliceType::P {
            // num_ref_idx_active_override_flag
            stream.push(false);
            // ref_pic_list_modification_flag_l0
            stream.push(false);
            // adaptive_ref_pic_marking_mode_flag
            stream.push(false);
        }
        // pic_order_cnt_lsb
        if active_sps.pic_order_cnt_type == 0 {
            let bits = active_sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            for _ in (0..bits).rev() {
                stream.push(false);
            }
        } else {
            todo!()
        }
        if self.slice_type == SliceType::I || self.slice_type == SliceType::SI {
            // no_output_of_prior_pics_flag
            stream.push(true);
            // long_term_reference_flag
            stream.push(false);
        } else {
            // stream.push(false);
        }

        // Slice_qp_delta
        stream.extend(signed_exponential_golomb_encode(0));
        if self.slice_type != SliceType::I {
            let mut p_data = BitStream::new();
            // TODO: Each 8x8 block gets its own nc.
            // TODO: Replace this with a structure that seaches linearly to make the memory footprint smaller.
            let mut ncs: Vec<[[i8; 4]; 4]> = Vec::new();

            if let Some(data) = self.data {
                for data in data {
                    match data {
                        SliceData::Skip(run) => {
                            // skip run
                            p_data.extend(exponential_golomb_encode(*run));
                            for _ in 0..*run {
                                ncs.push([[0; _]; _]);
                            }
                        }

                        SliceData::Block { mv_x, mv_y, luma } => {
                            p_data.extend(exponential_golomb_encode(self.first_mb_type as u32));
                            // mv_x, mv_y
                            p_data.extend(signed_exponential_golomb_encode(*mv_x));
                            p_data.extend(signed_exponential_golomb_encode(*mv_y));
                            let height = ncs.len() / 20;
                            let width = ncs.len() % 20;
                            let coded_block_pattern =
                                coded_block_pattern(*luma, [[0; _]; _], [[0; _]; _]);
                            p_data.extend(exponential_golomb_encode(
                                (CBP_MAPPING_TABLE[(coded_block_pattern & 0b001111) as usize]) as _,
                            ));
                            if coded_block_pattern > 0 {
                                // mb_qp_delta
                                p_data.extend(signed_exponential_golomb_encode(0));
                            }
                            let blocks = subdivide_mb_8x8x4(*luma);
                            let length = ncs.len();
                            ncs.insert(length, [[0; _]; _]);

                            for i in 0..4 {
                                if coded_block_pattern >> i & 1 == 1 {
                                    let blocks = blocks[i];
                                    for (idx, block) in
                                        subdivide_mb_8x8_4x4x4(blocks).into_iter().enumerate()
                                    {
                                        let x = idx / 2;
                                        let y = idx % 2;
                                        let x = if i > 1 { x + 2 } else { x };
                                        let y = if i % 2 == 0 { y } else { y + 2 };
                                        let nc = determine_nc(&*ncs, x, y, width, height, 20, 10);

                                        if width > 8 {
                                            // exit(1)
                                        }
                                        println!("NC: {nc}");
                                        let (non_zero, stream) = cavlc_encode(block, nc);
                                        let array = ncs.get_mut(length).unwrap();
                                        if non_zero > 8 {
                                            println!("{block:?}")
                                        }
                                        array[x][y] = non_zero as i8;
                                        println!(
                                            "idx: {i} x: {x} y: {y} non_zero: {non_zero} {:?}",
                                            &array[0..4]
                                        );
                                        p_data.extend(stream);
                                    }
                                }
                            }
                            let chroma_pattern = coded_block_pattern >> 4;
                            if chroma_pattern == 2 {
                                // todo!("Add Chroma")
                            } else if chroma_pattern == 1 {
                                // todo!("Add Chroma")
                            } else {
                                // todo!("Add Chroma")
                            }
                        }
                    }
                }

                stream.extend(p_data);
            }
        } else {
            stream.extend(exponential_golomb_encode(self.first_mb_type as u32));
        }

        if self.slice_type != SliceType::I && self.slice_type != SliceType::SI {}

        let bits = stream.bits();
        let mut vec = stream.as_slice().to_vec();
        let added = insert_emulation_prevention_bytes(&mut vec);

        let stream = BitStream::from_slice_with_bits(&vec, bits + added);

        stream
    }
}

pub fn insert_emulation_prevention_bytes(data: &mut Vec<u8>) -> usize {
    let mut i = 0;
    let mut insertions = 0;

    let mut last_two = LastTwo::new();

    while i < data.len() {
        if last_two.eq([0; 2]) && data[i] < 0x4 {
            data.insert(i, 0x3);
            last_two.clear();
            i += 1;
            insertions += 1;
        }

        last_two.push(data[i]);
        i += 1;
    }

    insertions
}

pub struct MacroBlock {
    // Macro-block type is the same as slice type as noted by section 7.4.5 of the H.264 spec, table 7-10
    _type: MacroBlockType,
}

impl MacroBlock {
    pub fn to_bit_stream(&self) -> BitStream {
        let mut bits = BitStream::new();
        bits.extend(exponential_golomb_encode(self._type as u32));
        while bits.bits() % u8::BITS as usize != 0 {
            bits.push(false);
        }

        bits
    }
}

pub struct PicParameterSet {
    pic_parameter_set_id: u32,
    seq_parameter_set_id: u32,
    entropy_encoding_mode_flag: bool,
    bottom_field_pic_order_in_frame_present_flag: bool,
    num_slice_groups_minus_1: u32,
    num_ref_idx_l0_default_active_minus1: u32,
    num_ref_idx_l1_default_active_minus1: u32,
    weight_pred: bool,
    weight_bipred_idc: u8,
    pic_init_qp_minus26: i32,
    pic_init_qs_minus26: i32,
    chroma_qp_index_offset: i32,
    deblocking_filter_control_present_flag: bool,
    constrained_intra_pred_flag: bool,
    redundant_pic_cnt_present_flag: bool,
}

impl PicParameterSet {
    pub fn to_bit_stream(&self) -> BitStream {
        let mut stream = BitStream::new();
        stream.extend(exponential_golomb_encode(self.pic_parameter_set_id));
        stream.extend(exponential_golomb_encode(self.seq_parameter_set_id));
        stream.push(self.entropy_encoding_mode_flag);
        stream.push(self.bottom_field_pic_order_in_frame_present_flag);
        stream.extend(exponential_golomb_encode(self.num_slice_groups_minus_1 - 1));
        stream.extend(exponential_golomb_encode(
            self.num_ref_idx_l0_default_active_minus1 - 1,
        ));
        stream.extend(exponential_golomb_encode(
            self.num_ref_idx_l1_default_active_minus1 - 1,
        ));
        stream.push(self.weight_pred);
        stream.push(self.weight_bipred_idc & 0b1 != 0);
        stream.push(self.weight_bipred_idc & 0b10 != 0);
        stream.extend(signed_exponential_golomb_encode(self.pic_init_qp_minus26));
        stream.extend(signed_exponential_golomb_encode(self.pic_init_qs_minus26));
        stream.extend(signed_exponential_golomb_encode(
            self.chroma_qp_index_offset,
        ));
        stream.push(self.deblocking_filter_control_present_flag);
        stream.push(self.constrained_intra_pred_flag);
        stream.push(self.redundant_pic_cnt_present_flag);
        stream.push(true);

        stream
    }
}

pub struct SequenceParameterSet {
    profile_idc: u8,
    flags: u8,
    level_idc: u8,
    seq_parameter_set_id: u32,
    log2_max_frame_num_minus4: u32,
    pic_order_cnt_type: u32,
    log2_max_pic_order_cnt_lsb_minus4: u32,
    num_ref_frames: u32,
    gaps_in_frame_num_value_allowed_flag: bool,
    pick_width_in_macro_blocks_minus_1: u32,
    pick_height_in_macro_blocks_minus_1: u32,
    frame_macro_blocks_only_flag: bool,
    direct_8x8_inference_flag: bool,
    frame_cropping_flag: bool,
    vui_parameter_flag: bool,
}

impl SequenceParameterSet {
    pub fn to_bit_stream(&self) -> BitStream {
        let mut bits: BitStream =
            BitStream::from_slice(&[self.profile_idc, self.flags, self.level_idc]);
        bits.extend(exponential_golomb_encode(self.seq_parameter_set_id));

        bits.extend(exponential_golomb_encode(self.log2_max_frame_num_minus4));
        bits.extend(exponential_golomb_encode(self.pic_order_cnt_type));
        bits.extend(exponential_golomb_encode(
            self.log2_max_pic_order_cnt_lsb_minus4,
        ));
        bits.extend(exponential_golomb_encode(self.num_ref_frames));
        bits.push(self.gaps_in_frame_num_value_allowed_flag);
        bits.extend(exponential_golomb_encode(
            self.pick_width_in_macro_blocks_minus_1,
        ));
        bits.extend(exponential_golomb_encode(
            self.pick_height_in_macro_blocks_minus_1,
        ));
        bits.push(self.frame_macro_blocks_only_flag);
        bits.push(self.direct_8x8_inference_flag);
        bits.push(self.frame_cropping_flag);
        bits.push(self.vui_parameter_flag);
        bits.push(true);

        while bits.bits() % 8 != 0 {
            bits.push(false);
        }

        bits
    }
}

pub struct Encoder<W: Write> {
    writer: W,
    // TODO: Reusable bitstream
    bit_stream: BitStream,
    sps: Vec<SequenceParameterSet>,
    pps: Vec<PicParameterSet>,
}

impl<W: Write> Encoder<W> {
    pub fn new(writer: W) -> Encoder<W> {
        Self {
            writer,
            bit_stream: BitStream::new(),
            sps: Vec::new(),
            pps: Vec::new(),
        }
    }

    pub fn write_sps(&mut self, sps: SequenceParameterSet) {
        let nal_sps = NalUnitHeader {
            ref_idc: NalRefIdc::Three,
            unit_type: NalUnitType::SequenceParameterSet,
        };
        nal_sps.write(&mut self.writer).unwrap();
        self.writer
            .write_all(sps.to_bit_stream().as_slice())
            .unwrap();
        self.sps.push(sps);
    }

    pub fn write_pps(&mut self, pps: PicParameterSet) {
        let nal_pps = NalUnitHeader {
            ref_idc: NalRefIdc::Three,
            unit_type: NalUnitType::PictureParameterSet,
        };
        nal_pps.write(&mut self.writer).unwrap();
        self.writer
            .write_all(pps.to_bit_stream().as_slice())
            .unwrap();
        self.pps.push(pps);
    }

    // TODO: This should also write the frame
    pub fn write_slice_header(&mut self, header: &SliceHeader, nal_unit_type: NalUnitType) {
        let nal_unit = NalUnitHeader {
            ref_idc: NalRefIdc::One,
            unit_type: nal_unit_type,
        };
        nal_unit.write(&mut self.writer).unwrap();
        self.writer
            .write_all(
                header
                    .to_bit_stream(
                        &self.pps[header.pic_parameter_set_id as usize],
                        &self.sps[header.sequence_parameter_set_id as usize],
                    )
                    .as_slice(),
            )
            .unwrap();
    }
}

#[must_use]
pub fn subdivide_macro_block(block: [[i32; 16]; 16]) -> [[[i32; 4]; 4]; 16] {
    const _: () =
        const { assert!(align_of::<[[i32; 16]; 16]>() == align_of::<[[[i32; 4]; 4]; 16]>()) };
    const _: () =
        const { assert!(size_of::<[[i32; 16]; 16]>() == size_of::<[[[i32; 4]; 4]; 16]>()) };

    // SAFETY: The size and alignment are equal
    unsafe { std::mem::transmute(block) }
}

#[must_use]
pub fn concat_macro_block(block: [[[i32; 4]; 4]; 16]) -> [[i32; 16]; 16] {
    unsafe { std::mem::transmute(block) }
}

pub fn subdivide_mb_8x8x4(block: [[i32; 16]; 16]) -> [[[i32; 8]; 8]; 4] {
    const _: () =
        const { assert!(align_of::<[[i32; 16]; 16]>() == align_of::<[[[i32; 8]; 8]; 4]>()) };
    const _: () =
        const { assert!(size_of::<[[i32; 16]; 16]>() == size_of::<[[[i32; 8]; 8]; 4]>()) };
    const _: () = const { assert!(16 * 16 == 8 * 8 * 4) };

    // SAFETY: The size and alignment are equal
    unsafe { std::mem::transmute(block) }
}

#[must_use]
pub fn subdivide_mb_8x8_4x4x4(block: [[i32; 8]; 8]) -> [[[i32; 4]; 4]; 4] {
    unsafe { std::mem::transmute(block) }
}

fn main() {
    // TODO: EMULATION PREVENTION YOU FUCKING LOSER

    let video = std::fs::read("output.yuv").unwrap();
    let mut frames = Vec::with_capacity(912);
    let mut read = 0;
    let res = 320 * 160;
    let y_size = res;
    let chroma_size = 160 * 80;
    while read < video.len() {
        let start = read;
        let chroma = start + y_size;
        frames.push(Frame::new(
            &video[start..start + y_size],
            &video[chroma..chroma + chroma_size],
            &video[chroma + chroma_size..chroma + chroma_size * 2],
            320,
            160,
        ));

        read += res + chroma_size * 2;
    }

    let macro_blocks_width: u32 = 320 / 16;
    let macro_blocks_height: u32 = 160 / 16;

    let file = std::fs::File::create("output.h264").unwrap();
    let mut encoder = Encoder::new(file);

    let sps = SequenceParameterSet {
        profile_idc: 66,
        flags: 0,
        level_idc: 10,
        seq_parameter_set_id: 0,
        log2_max_frame_num_minus4: 0,
        pic_order_cnt_type: 0,
        log2_max_pic_order_cnt_lsb_minus4: 0,
        num_ref_frames: 0,
        gaps_in_frame_num_value_allowed_flag: false,
        pick_width_in_macro_blocks_minus_1: macro_blocks_width - 1,
        pick_height_in_macro_blocks_minus_1: macro_blocks_height - 1,
        frame_macro_blocks_only_flag: true,
        direct_8x8_inference_flag: false,
        frame_cropping_flag: false,
        vui_parameter_flag: false,
    };

    let pps = PicParameterSet {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        entropy_encoding_mode_flag: false,
        bottom_field_pic_order_in_frame_present_flag: false,
        num_slice_groups_minus_1: 1,
        num_ref_idx_l0_default_active_minus1: 1,
        num_ref_idx_l1_default_active_minus1: 1,
        weight_pred: false,
        weight_bipred_idc: 0,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset: 0,
        deblocking_filter_control_present_flag: false,
        constrained_intra_pred_flag: false,
        redundant_pic_cnt_present_flag: false,
    };
    encoder.write_sps(sps);
    encoder.write_pps(pps);

    let mut slice_data = Vec::new();
    let mut prev_frame = Frame::new(&[], &[], &[], 0, 0);
    for (i, frame) in frames.drain(..).enumerate() {
        if i > 1 {
            break;
        }

        println!("{i}");
        if i % 2 == 1 {
            let mut skip_run = 0;
            for h in 0..macro_blocks_height as usize {
                for w in 0..macro_blocks_width as usize {
                    if let Some((x, y, most_similar, original)) = motion_esimtation(
                        &prev_frame,
                        &frame,
                        h,
                        w,
                        macro_blocks_width as usize - 1,
                        macro_blocks_height as usize - 1,
                    ) {
                        let casted_similar = most_similar.map(|v| v.map(|v| v as i32));
                        let casted_original = original.map(|v| v.map(|v| v as i32));
                        let blocks_similar = subdivide_macro_block(casted_similar)
                            .map(|block| IMat4x4::new(block[0], block[1], block[2], block[3]));
                        let blocks_original = subdivide_macro_block(casted_original)
                            .map(|block| IMat4x4::new(block[0], block[1], block[2], block[3]));
                        let diffs: [IMat4x4; 16] =
                            array::from_fn(|i| blocks_original[i] - blocks_similar[i]);
                        let transformed = diffs.map(|block| {
                            transform_matrix_mf(
                                block,
                                (encoder.pps[0].pic_init_qp_minus26 + 26) as u32,
                            )
                            .to_array()
                        });
                        let macro_block = concat_macro_block(transformed.clone());
                        slice_data.push(SliceData::Skip(skip_run));
                        slice_data.push(SliceData::Block {
                            luma: macro_block,
                            mv_x: x as i32,
                            mv_y: y as i32,
                        });
                        skip_run = 0;
                    } else {
                        skip_run += 1;
                    }
                }
            }

            if skip_run != 0 {
                slice_data.push(SliceData::Skip(skip_run));
            }

            let slice_header = SliceHeader {
                first_mb_in_slice: 0,
                slice_type: SliceType::P,
                sequence_parameter_set_id: 0,
                pic_parameter_set_id: 0,
                frame_num: if i != 0 && i % 2 == 1 { 1 } else { 0 },
                idr_pic_id: 0,
                pic_order_cnt_lsb: 0,
                first_mb_type: MacroBlockType::I_NxN,
                data: Some(&slice_data),
            };
            encoder.write_slice_header(&slice_header, NalUnitType::CodedSliceOfANonIdrPicture);
            slice_data.clear();

            // break;
        } else {
            let slice_header = SliceHeader {
                first_mb_in_slice: 0,
                slice_type: SliceType::I,
                sequence_parameter_set_id: 0,
                pic_parameter_set_id: 0,
                frame_num: 0,
                idr_pic_id: 0,
                pic_order_cnt_lsb: 0,
                first_mb_type: MacroBlockType::I_PCM,
                data: None,
            };
            encoder.write_slice_header(&slice_header, NalUnitType::CodedSliceOfAnIdrPicture);

            write_macro_blocks(
                &frame,
                &mut encoder.writer,
                macro_blocks_width as usize,
                macro_blocks_height as usize,
            );
            prev_frame = frame.clone()
        }

        encoder.writer.write(&[0x80]).unwrap();
    }
}

pub struct LastTwo {
    len: u8,
    buf: [u8; 2],
}

impl LastTwo {
    fn new() -> Self {
        Self {
            len: 0,
            buf: [0; 2],
        }
    }

    fn push(&mut self, val: u8) {
        self.buf[1] = self.buf[0];
        self.buf[0] = val;
        if self.len < 2 {
            self.len += 1;
        }
    }

    fn eq(&self, pat: [u8; 2]) -> bool {
        self.len == 2 && pat == self.buf
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

pub fn write_macro_blocks<W: Write>(
    frame: &Frame,
    writer: &mut W,
    block_width: usize,
    block_height: usize,
) {
    let mut first = true;
    const PAT: [u8; 4] = [0, 1, 2, 3];
    let mut last_two = LastTwo::new();

    for i in 0..block_height {
        for j in 0..block_width {
            if first {
                first = false;
            } else {
                writer
                    .write_all(
                        MacroBlock {
                            _type: MacroBlockType::I_PCM,
                        }
                        .to_bit_stream()
                        .as_slice(),
                    )
                    .unwrap();
            }

            for x in 0..16 {
                for y in 0..16 {
                    let byte = frame.y()[i * 16 + x][j * 16 + y];
                    if last_two.eq([0; 2]) && PAT.contains(&byte) {
                        last_two.clear();
                        writer.write(&[0x3]).unwrap();
                    }
                    last_two.push(byte);
                    writer.write(&[byte]).unwrap();
                }
            }

            for x in 0..8 {
                for y in 0..8 {
                    let byte = frame.cb()[i * 8 + x][j * 8 + y];
                    if last_two.eq([0; 2]) && PAT.contains(&byte) {
                        last_two.clear();
                        writer.write(&[0x3]).unwrap();
                    }
                    last_two.push(byte);
                    writer.write(&[byte]).unwrap();
                }
            }
            for x in 0..8 {
                for y in 0..8 {
                    let byte = frame.cr()[i * 8 + x][j * 8 + y];
                    if last_two.eq([0; 2]) && PAT.contains(&byte) {
                        last_two.clear();
                        writer.write(&[0x3]).unwrap();
                    }
                    last_two.push(byte);
                    writer.write(&[byte]).unwrap();
                }
            }
        }
    }
}
/// Returns number of leading zeroes and code num
pub fn exponential_golomb_encode(num: u32) -> BitStream {
    let mut bits = BitStream::new();
    let code_num = num + 1;
    let k = code_num.leading_zeros() ^ 31;
    for _ in 0..k {
        bits.push(false);
    }
    for i in (0..k + 1).rev() {
        bits.push(code_num >> i & 1 != 0);
    }
    bits
}

pub fn signed_exponential_golomb_encode(mut num: i32) -> BitStream {
    if num > 0 {
        num += num - 1;
    } else {
        num *= -2;
    }

    exponential_golomb_encode(num as u32)
}

#[test]
pub fn each_matrix() {
    const LUT: [[i32; 3]; 6] = [
        [13107, 5243, 8066],
        [11916, 4660, 7490],
        [10082, 4194, 6554],
        [9362, 3647, 5825],
        [8192, 3355, 5243],
        [7282, 2893, 4559],
    ];

    for qp in 0..6 {
        let [z, o, t] = LUT[(qp % 6) as usize];
        let matrix = IMat4x4::new([z, o, z, o], [t, o, t, o], [z, o, z, o], [t, o, t, o]);
    }
}

/// Sourced from: https://www.vcodex.com/h264avc-4x4-transform-and-quantization
/// Formula goes as follows, DCT(mat) * MF + (2^qbits / 3) >> qbits
fn transform_matrix_mf(mat: IMat4x4, qp: u32) -> IMat4x4 {
    const LUT: [[i32; 3]; 6] = [
        [13107, 5243, 8066],
        [11916, 4660, 7490],
        [10082, 4194, 6554],
        [9362, 3647, 5825],
        [8192, 3355, 5243],
        [7282, 2893, 4559],
    ];

    let w = {
        #[rustfmt::skip]
        let c = IMat4x4::new(
            [1, 1, 1, 1],
            [2, 1, -1, -2],
            [1, -1, -1, 1],
            [1, -2, 2, -1],
        );
        #[rustfmt::skip]
        let c_t = IMat4x4::new_col(
            [1, 2, 1, 1],
            [1, 1, -1, -2],
            [1, -1, -1, 2],
            [1, -2, 1, -1],
        );
        (c * mat).mul_skip_col_transform(c_t)
    };

    let [z, o, t] = LUT[(qp % 6) as usize];
    #[rustfmt::skip]
    let matrix = IMat4x4::new(
        [z, o, z, o], 
        [t, o, t, o], 
        [z, o, z, o], 
        [t, o, t, o]
    );

    let qbits = 15 + qp / 6;
    let f = pow_2(qbits) / 3;

    (w.abs().component_mul(matrix) + f as i32 >> qbits as i32).sign(w)
}

pub fn pow_2(pow: u32) -> u32 {
    1 << pow
}

#[cfg(test)]
mod benches {}

// Waiting on https://github.com/rust-lang/rust/issues/149298
// const _LUT: [IMat4x4; 6] = [
//     IMat4x4::new(
//         [13107, 5243, 13107, 5243],
//         [8066, 5243, 8066, 5243],
//         [13107, 5243, 13107, 5243],
//         [8066, 5243, 8066, 5243],
//     ),
//     IMat4x4::new(
//         [11916, 4660, 11916, 4660],
//         [7490, 4660, 7490, 4660],
//         [11916, 4660, 11916, 4660],
//         [7490, 4660, 7490, 4660],
//     ),
//     IMat4x4::new(
//         [10082, 4194, 10082, 4194],
//         [6554, 4194, 6554, 4194],
//         [10082, 4194, 10082, 4194],
//         [6554, 4194, 6554, 4194],
//     ),
//     IMat4x4::new(
//         [9362, 3647, 9362, 3647],
//         [5825, 3647, 5825, 3647],
//         [9362, 3647, 9362, 3647],
//         [5825, 3647, 5825, 3647],
//     ),
//     IMat4x4::new(
//         [8192, 3355, 8192, 3355],
//         [5243, 3355, 5243, 3355],
//         [8192, 3355, 8192, 3355],
//         [5243, 3355, 5243, 3355],
//     ),
//     IMat4x4::new(
//         [7282, 2893, 7282, 2893],
//         [4559, 2893, 4559, 2893],
//         [7282, 2893, 7282, 2893],
//         [4559, 2893, 4559, 2893],
//     ),
// ];

pub const CBP_MAPPING_TABLE: [u8; 48] = [
    0, 2, 3, 7, 4, 8, 17, 13, 5, 18, 9, 14, 10, 15, 16, 11, 1, 32, 33, 36, 34, 37, 44, 40, 25, 45,
    38, 41, 39, 42, 43, 19, 6, 24, 25, 20, 26, 21, 46, 28, 27, 47, 22, 29, 23, 30, 31, 12,
];

pub fn coded_block_pattern(
    luma_block: [[i32; 16]; 16],
    cr: [[i32; 8]; 8],
    cb: [[i32; 8]; 8],
) -> u8 {
    let mut top_left = [[0; 8]; 8];
    let mut bottom_left = [[0; 8]; 8];
    let mut top_right = [[0; 8]; 8];
    let mut bottom_right = [[0; 8]; 8];

    for i in 0..8 {
        for j in 0..8 {
            top_left[i][j] = luma_block[i][j];
            top_right[i][j] = luma_block[i][j + 8];
            bottom_left[i][j] = luma_block[i + 8][j];
            bottom_right[i][j] = luma_block[i + 8][j + 8];
        }
    }

    let zero = top_left.iter().flatten().any(|&f| f != 0) as u8;
    let one = top_right.iter().flatten().any(|&f| f != 0) as u8;
    let two = bottom_left.iter().flatten().any(|&f| f != 0) as u8;
    let three = bottom_right.iter().flatten().any(|&f| f != 0) as u8;
    let four = cr.iter().flatten().any(|&f| f != 0);
    let five = cb.iter().flatten().any(|&f| f != 0);

    let pattern = if five {
        2
    } else if four {
        1
    } else {
        0
    };

    zero | one << 1 | two << 2 | three << 3 | pattern << 4
}

fn determine_nc(
    // Non-zero-counts
    ncs: &[[[i8; 4]; 4]],
    sub_block_x: usize,
    sub_block_y: usize,
    block_width: usize,
    block_height: usize,
    max_block_width: usize,
    max_block_height: usize,
) -> i8 {
    let above = if sub_block_x > 0 {
        ncs[block_height * max_block_width + block_width][sub_block_x - 1][sub_block_y]
    } else if block_width > 0 {
        ncs[block_height * max_block_width + (block_width - 1)][3][sub_block_y]
    } else {
        0
    };

    let left = if sub_block_y > 0 {
        ncs[block_height * max_block_width + block_width][sub_block_x][sub_block_y - 1]
    } else if block_height > 0 {
        ncs[(block_height - 1) * max_block_width + block_width][sub_block_x][3]
    } else {
        0
    };

    println!("Block: {block_width} {block_height} {left} {above}");

    match (left == 0, above == 0) {
        (true, true) => 0,
        (true, false) => above,
        (false, true) => left,
        (false, false) => (left + above + 1) >> 1,
    }
}
