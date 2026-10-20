use std::{
    arch::x86_64::{
        __m128, __m128i, _mm_abs_epi32, _mm_add_epi32, _mm_add_ps, _mm_and_si128, _mm_broadcast_ss,
        _mm_castps_si128, _mm_castsi128_ps, _mm_cvtsi128_si32, _mm_div_ps, _mm_hadd_epi32,
        _mm_load_epi32, _mm_load_si128, _mm_mul_ps, _mm_mullo_epi32, _mm_or_epi32, _mm_set_epi32,
        _mm_setr_ps, _mm_shuffle_epi32, _mm_shufflelo_epi16, _mm_sign_epi32, _mm_sll_epi32,
        _mm_srl_epi32, _mm_store_epi32, _mm_storeu_epi32, _mm_storeu_ps, _mm_sub_epi32,
    },
    ops::Sub,
};

#[derive(Clone, Copy)]
pub struct IVec4(pub(crate) __m128i);

impl IVec4 {
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        #[repr(align(16))]
        struct Align16([i32; 4]);
        let arr = Align16([x, y, z, w]);
        let vec = unsafe { _mm_load_si128(arr.0.as_ptr() as _) };
        Self(vec)
    }

    pub fn splat(x: i32) -> Self {
        let arr = [x; 4];
        let vec = unsafe { _mm_load_epi32(arr.as_ptr() as _) };
        Self(vec)
    }

    pub fn abs(&self) -> Self {
        Self(unsafe { _mm_abs_epi32(self.0) })
    }

    pub fn sign(&self, vec: Self) -> Self {
        Self(unsafe { _mm_sign_epi32(self.0, vec.0) })
    }

    pub fn dot(&self, vec: Self) -> i32 {
        let mul = *self * vec;
        mul.reduce_sum()
    }

    /// https://stackoverflow.com/a/35270026
    pub fn reduce_sum(&self) -> i32 {
        let hi64 = Self(unsafe { _mm_shuffle_epi32::<{ _mm_shuffle(1, 0, 3, 2) }>(self.0) });
        let sum64 = hi64 + *self;
        let hi32 = Self(unsafe { _mm_shufflelo_epi16::<{ _mm_shuffle(1, 0, 3, 2) }>(sum64.0) });
        let sum32 = sum64 + hi32;
        unsafe { _mm_cvtsi128_si32(sum32.0) }
    }

    pub fn to_array(self) -> [i32; 4] {
        #[repr(align(16))]
        struct Align16([i32; 4]);
        let mut arr = Align16([0; 4]);
        unsafe { _mm_store_epi32(arr.0.as_mut_ptr(), self.0) };
        arr.0
    }
}

pub const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

impl std::ops::Mul for IVec4 {
    type Output = IVec4;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_mullo_epi32(self.0, rhs.0) })
    }
}

impl std::ops::Add for IVec4 {
    type Output = IVec4;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_add_epi32(self.0, rhs.0) })
    }
}

impl std::ops::Shr<i32> for IVec4 {
    type Output = IVec4;

    fn shr(self, rhs: i32) -> Self::Output {
        let load = unsafe { _mm_set_epi32(0, 0, 0, rhs) };
        Self(unsafe { _mm_srl_epi32(self.0, load) })
    }
}

impl std::ops::Shl<i32> for IVec4 {
    type Output = IVec4;

    fn shl(self, rhs: i32) -> Self::Output {
        let load = unsafe { _mm_set_epi32(0, 0, 0, rhs) };
        Self(unsafe { _mm_sll_epi32(self.0, load) })
    }
}

impl std::ops::BitOr<IVec4> for IVec4 {
    type Output = IVec4;

    fn bitor(self, rhs: IVec4) -> Self::Output {
        Self(unsafe { _mm_or_epi32(self.0, rhs.0) })
    }
}

impl std::ops::Div<i32> for IVec4 {
    type Output = IVec4;

    fn div(self, rhs: i32) -> Self::Output {
        let float_a = unsafe { _mm_castsi128_ps(self.0) };
        let float_b = unsafe { _mm_broadcast_ss(&(rhs as f32)) };

        Self(unsafe { _mm_castps_si128(_mm_div_ps(float_a, float_b)) })
    }
}

impl std::ops::BitAnd<IVec4> for IVec4 {
    type Output = IVec4;

    fn bitand(self, rhs: IVec4) -> Self::Output {
        Self(unsafe { _mm_and_si128(self.0, rhs.0) })
    }
}

impl Into<IVec4> for [i32; 4] {
    fn into(self) -> IVec4 {
        IVec4::new(self[0], self[1], self[2], self[3])
    }
}

impl Sub<IVec4> for IVec4 {
    type Output = IVec4;

    fn sub(self, rhs: IVec4) -> Self::Output {
        IVec4(unsafe { _mm_sub_epi32(self.0, rhs.0) })
    }
}

#[derive(Clone, Copy)]
pub struct Vec4(__m128);

impl std::fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut store: [f32; 4] = [0.0; 4];
        unsafe {
            _mm_storeu_ps(store.as_mut_ptr(), self.0);
        }
        f.debug_tuple("IVec4")
            .field(&store[0])
            .field(&store[1])
            .field(&store[2])
            .field(&store[3])
            .finish()
    }
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_setr_ps(x, y, z, w) })
    }

    pub fn splat(x: f32) -> Self {
        Self(unsafe { _mm_broadcast_ss(&x) })
    }
}

impl std::ops::Mul for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }
}

impl std::ops::Div for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_div_ps(self.0, rhs.0) })
    }
}

impl std::ops::Add for Vec4 {
    type Output = Vec4;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_add_ps(self.0, rhs.0) })
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(unsafe { _mm_mul_ps(self.0, Self::splat(rhs).0) })
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: f32) -> Self::Output {
        Self(unsafe { _mm_div_ps(self.0, Self::splat(rhs).0) })
    }
}

impl std::ops::Add<f32> for Vec4 {
    type Output = Vec4;

    fn add(self, rhs: f32) -> Self::Output {
        Self(unsafe { _mm_add_ps(self.0, Self::splat(rhs).0) })
    }
}
