use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Frame {
    data: Box<[u8]>,
    width: usize,
    height: usize,
}

impl Frame {
    #[must_use]
    pub fn new(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize) -> Frame {
        let size = width * height + (width * height) / 2;
        let data = vec![0u8; size].into_boxed_slice();

        let mut frame = Frame {
            data,
            width,
            height,
        };

        frame.replace(y, cb, cr);

        frame
    }

    #[must_use]
    pub fn y(&self) -> Array2D<'_> {
        Array2D {
            data: &self.data[..self.width * self.height],
            width: self.width,
        }
    }

    #[must_use]
    pub fn y_mut(&mut self) -> Array2DMut<'_> {
        Array2DMut {
            data: &mut self.data[..self.width * self.height],
            width: self.width,
        }
    }

    #[must_use]
    pub fn cb(&self) -> Array2D<'_> {
        let y_offset = self.width * self.height;
        let chroma_width = self.width / 2;
        let chroma_height = self.height / 2;
        let cr_size = chroma_width * chroma_height;
        Array2D {
            data: &self.data[y_offset..y_offset + cr_size],
            width: chroma_width,
        }
    }

    #[must_use]
    pub fn cb_mut(&mut self) -> Array2DMut<'_> {
        let y_offset = self.width * self.height;
        let chroma_width = self.width / 2;
        let chroma_height = self.height / 2;
        let cr_size = chroma_width * chroma_height;
        Array2DMut {
            data: &mut self.data[y_offset..y_offset + cr_size],
            width: chroma_width,
        }
    }

    #[must_use]
    pub fn cr(&self) -> Array2D<'_> {
        let y_offset = self.width * self.height;
        let chroma_width = self.width / 2;
        let chroma_height = self.height / 2;
        let cr_size = chroma_width * chroma_height;
        Array2D {
            data: &self.data[y_offset + cr_size..y_offset + cr_size + cr_size],
            width: chroma_width,
        }
    }

    #[must_use]
    pub fn cr_mut(&mut self) -> Array2DMut<'_> {
        let y_offset = self.width * self.height;
        let chroma_width = self.width / 2;
        let chroma_height = self.height / 2;
        let cr_size = chroma_width * chroma_height;
        Array2DMut {
            data: &mut self.data[y_offset + cr_size..y_offset + cr_size + cr_size],
            width: chroma_width,
        }
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn replace(&mut self, y: &[u8], cb: &[u8], cr: &[u8]) {
        let y_size = self.y_size();
        let chroma_size = self.chroma_size();

        self.data[..y_size].copy_from_slice(y);
        self.data[y_size..y_size + chroma_size].copy_from_slice(cb);
        self.data[y_size + chroma_size..y_size + chroma_size * 2].copy_from_slice(cr);
    }

    #[must_use]
    fn y_size(&self) -> usize {
        self.width * self.height
    }

    #[must_use]
    fn chroma_size(&self) -> usize {
        (self.width / 2) * (self.height / 2)
    }
}

pub struct Array2D<'a> {
    data: &'a [u8],
    width: usize,
}

impl Array2D<'_> {
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.data.len() / self.width
    }
}

impl Index<usize> for Array2D<'_> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let row = self.width * index;
        &self.data[row..(row + self.width)]
    }
}

pub struct Array2DMut<'a> {
    data: &'a mut [u8],
    width: usize,
}

impl Array2DMut<'_> {
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.data.len() / self.width
    }
}

impl Index<usize> for Array2DMut<'_> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        let row = self.width * index;
        &self.data[row..(row + self.width)]
    }
}

impl IndexMut<usize> for Array2DMut<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = self.width * index;
        &mut self.data[row..(row + self.width)]
    }
}
