//! This only supports u32 as the storage type at the moment

use std::{alloc::Layout, fmt::Display, ptr::NonNull};

pub struct BitStream {
    ptr: NonNull<u8>,
    cap: usize,
    bits: usize,
}

impl BitStream {
    pub const fn new() -> BitStream {
        BitStream {
            ptr: NonNull::dangling(),
            cap: 0,
            bits: 0,
        }
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    /// `bit` is either true or false
    pub fn push(&mut self, bit: bool) {
        let len_elms = self.bits / u8::BITS as usize;
        let idx_bit = self.bits % u8::BITS as usize;

        if self.cap == len_elms {
            self.ptr = NonNull::new(self.grow()).expect("alloc did not fail");
        }

        let elm = unsafe { self.ptr.add(len_elms) };
        if idx_bit == 0 {
            unsafe {
                elm.write(0);
            }
        }
        unsafe {
            *elm.as_ptr() |= (bit as u8) << (7 - idx_bit);
        };
        self.bits += 1;
    }

    pub fn get(&self, idx: usize) -> Option<bool> {
        if idx >= self.bits {
            None
        } else {
            let len_elms = idx / u8::BITS as usize;
            let bit = idx % u8::BITS as usize;
            let elm = unsafe { self.ptr.add(len_elms) };
            let copy = unsafe { elm.read() };
            Some(copy >> 7 - bit & 1 != 0)
        }
    }

    pub fn extend(&mut self, stream: BitStream) {
        for bit in stream.into_iter() {
            self.push(bit);
        }
    }

    pub fn reserve(&mut self, len: usize) {
        let layout = Layout::new::<u8>();
        let ptr = unsafe { std::alloc::realloc(self.ptr.as_ptr(), layout, self.cap + len) };
        self.ptr = NonNull::new(ptr).unwrap();
    }

    pub fn from_slice(slice: &[u8]) -> BitStream {
        let alloc = unsafe { std::alloc::alloc(Layout::array::<u8>(slice.len()).unwrap()) };
        let cap = slice.len();
        let bits = slice.len() * u8::BITS as usize;
        unsafe { alloc.copy_from(slice.as_ptr(), slice.len()) };
        BitStream {
            ptr: NonNull::new(alloc).unwrap(),
            cap,
            bits,
        }
    }

    /// Produces a byte aligned slice
    pub fn as_slice(&self) -> &[u8] {
        let len = self.bits / u8::BITS as usize;
        let len = if self.bits % u8::BITS as usize > 0 {
            len + 1
        } else {
            len
        };
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    pub fn from_slice_with_bits(slice: &[u8], bits: usize) -> Self {
        let mut slice = Self::from_slice(slice);
        slice.bits = bits;
        slice
    }

    pub fn clear(&mut self) {
        self.bits = 0;
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.cap);
        }
    }
}

pub struct BitStreamIter<'a> {
    vec: &'a BitStream,
    idx: usize,
}

impl Iterator for BitStreamIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.vec.get(self.idx);
        self.idx += 1;
        item
    }
}

impl<'a> IntoIterator for &'a BitStream {
    type Item = bool;

    type IntoIter = BitStreamIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { vec: self, idx: 0 }
    }
}
impl BitStream {
    fn grow(&mut self) -> *mut u8 {
        let new_cap = if self.cap == 0 { 1 } else { self.cap * 2 };

        let ptr = if self.bits == 0 && self.cap == 0 {
            let layout = Layout::new::<u8>();
            unsafe { std::alloc::alloc(layout) as *mut u8 }
        } else {
            let layout = Layout::array::<u8>(self.cap).unwrap();
            unsafe { std::alloc::realloc(self.ptr.as_ptr() as _, layout, new_cap) as *mut u8 }
        };

        self.cap = new_cap;

        ptr
    }
}

impl Drop for BitStream {
    fn drop(&mut self) {
        let layout = Layout::array::<u8>(self.cap).unwrap();
        if self.cap != 0 {
            unsafe {
                std::alloc::dealloc(self.ptr.as_ptr() as _, layout);
            }
        }
    }
}
