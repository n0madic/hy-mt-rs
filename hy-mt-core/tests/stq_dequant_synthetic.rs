//! Hand-built STQ1_0 block decoded against the codebook by inspection.

use half::f16;
use hy_mt_core::quant::{
    dequantize_row_stq1_0_f32, BlockStq1_0, BLOCK_BYTES, QK_K, STQ1_0_CODEBOOK,
};

#[test]
fn dequant_zero_block_is_zero() {
    let block = BlockStq1_0 {
        qs: [0; 32],
        sign: [0; 8],
        d: f16::from_f32(1.0),
    };
    // qs = code 0, sign = 0 → codebook[0] = 0xA9 = 0b10_10_10_01
    // Lanes (LSB-first per pair): {01,10,10,10} → q-values {0,1,1,1} → {-1, 0, 0, 0}? Wait, q is 0..3:
    //   p=0: bits 0..1 = 0b01 → q=1 → q-1 =  0 (zero lane)
    //   p=1: bits 2..3 = 0b10 → q=2 → q-1 = +1
    //   p=2: bits 4..5 = 0b10 → q=2 → q-1 = +1
    //   p=3: bits 6..7 = 0b10 → q=2 → q-1 = +1
    let mut out = vec![0.0f32; QK_K];
    dequantize_row_stq1_0_f32(std::slice::from_ref(&block), &mut out).unwrap();

    // Inspect the first group of four:
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 1.0);
    assert_eq!(out[2], 1.0);
    assert_eq!(out[3], 1.0);

    // All 64 groups have the same code/sign, so the entire 256-weight block
    // should repeat the same pattern.
    for chunk in out.chunks_exact(4) {
        assert_eq!(chunk, [0.0, 1.0, 1.0, 1.0]);
    }
}

#[test]
fn dequant_scales_by_d() {
    let block = BlockStq1_0 {
        qs: [0; 32],
        sign: [0; 8],
        d: f16::from_f32(0.25),
    };
    let mut out = vec![0.0f32; QK_K];
    dequantize_row_stq1_0_f32(std::slice::from_ref(&block), &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(chunk, [0.0, 0.25, 0.25, 0.25]);
    }
}

#[test]
fn dequant_picks_lane_via_codebook_position() {
    // code = 1 → codebook[1] = 0x89 = 0b10_00_10_01
    //   p=0: 0b01 → 0 (zero lane)
    //   p=1: 0b10 → +1
    //   p=2: 0b00 → -1
    //   p=3: 0b10 → +1
    // Place this in group 0; leave the rest zero (=> still uses code 0 for groups 1..)
    let mut qs = [0u8; 32];
    qs[0] = 0x01; // group 0 = nibble 1, group 1 = nibble 0
    let block = BlockStq1_0 {
        qs,
        sign: [0; 8],
        d: f16::from_f32(2.0),
    };

    let mut out = vec![0.0f32; QK_K];
    dequantize_row_stq1_0_f32(std::slice::from_ref(&block), &mut out).unwrap();

    assert_eq!(&out[0..4], &[0.0, 2.0, -2.0, 2.0]);
    // Group 1 falls back to code 0 (codebook[0] = 0xA9), as in the first test.
    assert_eq!(&out[4..8], &[0.0, 2.0, 2.0, 2.0]);
}

#[test]
fn dequant_sign_flips_non_zero_lanes() {
    // code=0, sign=1 → codebook[16] = 0x01 = 0b00_00_00_01
    //   p=0: 0b01 → 0 (zero lane)
    //   p=1: 0b00 → -1
    //   p=2: 0b00 → -1
    //   p=3: 0b00 → -1
    // (the negated counterpart of the all-positive base pattern)
    let mut sign = [0u8; 8];
    sign[0] = 0x01;
    let block = BlockStq1_0 {
        qs: [0; 32],
        sign,
        d: f16::from_f32(1.0),
    };
    let mut out = vec![0.0f32; QK_K];
    dequantize_row_stq1_0_f32(std::slice::from_ref(&block), &mut out).unwrap();

    assert_eq!(&out[0..4], &[0.0, -1.0, -1.0, -1.0]);
    // Subsequent groups use sign bit 0, so the all-positive pattern returns.
    assert_eq!(&out[4..8], &[0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn block_size_constants_are_consistent() {
    assert_eq!(BLOCK_BYTES, std::mem::size_of::<BlockStq1_0>());
    assert_eq!(QK_K, 256);
    // 32 codebook entries (16 per sign half).
    assert_eq!(STQ1_0_CODEBOOK.len(), 32);
}
