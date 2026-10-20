use std::arch::x86_64::_mm_unpackhi_epi32;
use std::arch::x86_64::_mm_unpackhi_epi64;
use std::arch::x86_64::_mm_unpacklo_epi32;
use std::arch::x86_64::_mm_unpacklo_epi64;
use std::ops::Index;

use crate::vectors::IVec4;

#[derive(Clone, Copy)]
pub struct IMat4x4 {
    x: IVec4,
    y: IVec4,
    z: IVec4,
    w: IVec4,
}

impl IMat4x4 {
    pub fn new<T: Into<IVec4>>(row1: T, row2: T, row3: T, row4: T) -> Self {
        IMat4x4 {
            x: row1.into(),
            y: row2.into(),
            z: row3.into(),
            w: row4.into(),
        }
    }

    pub fn new_col<T: Into<IVec4> + Index<usize, Output = i32>>(
        col1: T,
        col2: T,
        col3: T,
        col4: T,
    ) -> Self {
        Self {
            x: [col1[0], col2[0], col3[0], col4[0]].into(),
            y: [col1[1], col2[1], col3[1], col4[1]].into(),
            z: [col1[2], col2[2], col3[2], col4[2]].into(),
            w: [col1[3], col2[3], col3[3], col4[3]].into(),
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
            w: self.w.abs(),
        }
    }

    pub fn sign(&self, mat: Self) -> Self {
        Self {
            x: self.x.sign(mat.x),
            y: self.y.sign(mat.y),
            z: self.z.sign(mat.z),
            w: self.w.sign(mat.w),
        }
    }

    pub fn component_mul(&self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
            w: self.w * rhs.w,
        }
    }

    pub fn col_transform(&self) -> Self {
        let tmp0 = unsafe { _mm_unpacklo_epi32(self.x.0, self.y.0) };
        let tmp1 = unsafe { _mm_unpackhi_epi32(self.x.0, self.y.0) };
        let tmp2 = unsafe { _mm_unpacklo_epi32(self.z.0, self.w.0) };
        let tmp3 = unsafe { _mm_unpackhi_epi32(self.z.0, self.w.0) };

        // Resulting rows
        let x = IVec4(unsafe { _mm_unpacklo_epi64(tmp0, tmp2) });
        let y = IVec4(unsafe { _mm_unpackhi_epi64(tmp0, tmp2) });
        let z = IVec4(unsafe { _mm_unpacklo_epi64(tmp1, tmp3) });
        let w = IVec4(unsafe { _mm_unpackhi_epi64(tmp1, tmp3) });

        Self { x, y, z, w }
    }

    pub fn mul_skip_col_transform(self, rhs: Self) -> Self {
        let row0 = IVec4::new(
            self.x.dot(rhs.x),
            self.x.dot(rhs.y),
            self.x.dot(rhs.z),
            self.x.dot(rhs.w),
        );
        let row1 = IVec4::new(
            self.y.dot(rhs.x),
            self.y.dot(rhs.y),
            self.y.dot(rhs.z),
            self.y.dot(rhs.w),
        );
        let row2 = IVec4::new(
            self.z.dot(rhs.x),
            self.z.dot(rhs.y),
            self.z.dot(rhs.z),
            self.z.dot(rhs.w),
        );
        let row3 = IVec4::new(
            self.w.dot(rhs.x),
            self.w.dot(rhs.y),
            self.w.dot(rhs.z),
            self.w.dot(rhs.w),
        );

        IMat4x4 {
            x: row0,
            y: row1,
            z: row2,
            w: row3,
        }
    }

    #[must_use]
    pub fn to_array(self) -> [[i32; 4]; 4] {
        [
            self.x.to_array(),
            self.y.to_array(),
            self.z.to_array(),
            self.w.to_array(),
        ]
    }
}

impl std::ops::Mul<IMat4x4> for IMat4x4 {
    type Output = IMat4x4;

    fn mul(self, rhs: Self) -> Self::Output {
        let rhs = rhs.col_transform();
        let row0 = IVec4::new(
            self.x.dot(rhs.x),
            self.x.dot(rhs.y),
            self.x.dot(rhs.z),
            self.x.dot(rhs.w),
        );
        let row1 = IVec4::new(
            self.y.dot(rhs.x),
            self.y.dot(rhs.y),
            self.y.dot(rhs.z),
            self.y.dot(rhs.w),
        );
        let row2 = IVec4::new(
            self.z.dot(rhs.x),
            self.z.dot(rhs.y),
            self.z.dot(rhs.z),
            self.z.dot(rhs.w),
        );
        let row3 = IVec4::new(
            self.w.dot(rhs.x),
            self.w.dot(rhs.y),
            self.w.dot(rhs.z),
            self.w.dot(rhs.w),
        );

        IMat4x4 {
            x: row0,
            y: row1,
            z: row2,
            w: row3,
        }
    }
}

impl std::ops::Add<i32> for IMat4x4 {
    type Output = IMat4x4;

    fn add(self, rhs: i32) -> Self::Output {
        IMat4x4::new(
            self.x + IVec4::new(rhs, rhs, rhs, rhs),
            self.y + IVec4::new(rhs, rhs, rhs, rhs),
            self.z + IVec4::new(rhs, rhs, rhs, rhs),
            self.w + IVec4::new(rhs, rhs, rhs, rhs),
        )
    }
}

impl std::ops::Shr<i32> for IMat4x4 {
    type Output = IMat4x4;

    fn shr(self, rhs: i32) -> Self::Output {
        IMat4x4::new(self.x >> rhs, self.y >> rhs, self.z >> rhs, self.w >> rhs)
    }
}

impl std::ops::Shl<i32> for IMat4x4 {
    type Output = IMat4x4;

    fn shl(self, rhs: i32) -> Self::Output {
        IMat4x4::new(self.x << rhs, self.y << rhs, self.z << rhs, self.w << rhs)
    }
}

impl std::ops::BitAnd<IMat4x4> for IMat4x4 {
    type Output = IMat4x4;

    fn bitand(self, rhs: IMat4x4) -> Self::Output {
        IMat4x4::new(
            self.x & rhs.x,
            self.y & rhs.y,
            self.z & rhs.z,
            self.w & rhs.w,
        )
    }
}

impl std::ops::BitOr<IMat4x4> for IMat4x4 {
    type Output = IMat4x4;

    fn bitor(self, rhs: IMat4x4) -> Self::Output {
        IMat4x4::new(
            self.x | rhs.x,
            self.y | rhs.y,
            self.z | rhs.z,
            self.w | rhs.w,
        )
    }
}

impl std::ops::Div<i32> for IMat4x4 {
    type Output = IMat4x4;

    fn div(self, rhs: i32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl std::ops::Sub<IMat4x4> for IMat4x4 {
    type Output = IMat4x4;

    fn sub(self, rhs: IMat4x4) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}
