#!/usr/bin/env python

import math
from PIL import Image
from imagehash import ImageHash
import numpy as np

# --- REMOVED ---
# from matrix import MatrixUtil

class FINDHasher:

    LUMA_FROM_R_COEFF = float(0.299)
    LUMA_FROM_G_COEFF = float(0.587)
    LUMA_FROM_B_COEFF = float(0.114)

    FIND_WINDOW_SIZE_DIVISOR = 64

    def compute_dct_matrix(self):
        matrix_scale_factor = math.sqrt(2.0 / 64.0)
        d = [0] * 16
        for i in range(0, 16):
            di = [0] * 64
            for j in range(0, 64):
                di[j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))
            d[i] = di
        return d

    def __init__(self):
        # --- CHANGED: store as numpy once ---
        self.DCT_matrix = np.array(self.compute_dct_matrix(), dtype=np.float64)

    def fromFile(self, filepath):
        img = Image.open(filepath)
        return self.fromImage(img)

    def fromImage(self, img):
        img = img.copy()
        img.thumbnail((512, 512))

        numCols, numRows = img.size

        # --- CHANGED: use numpy arrays instead of MatrixUtil ---
        buffer1 = np.empty((numRows, numCols), dtype=np.float64)
        buffer2 = np.empty((numRows, numCols), dtype=np.float64)
        buffer64x64 = np.empty((64, 64), dtype=np.float64)
        buffer16x64 = np.empty((16, 64), dtype=np.float64)
        buffer16x16 = np.empty((16, 16), dtype=np.float64)

        self.fillFloatLumaFromBufferImage(img, buffer1)

        return self.findHash256FromFloatLuma(
            buffer1, buffer2, numRows, numCols,
            buffer64x64, buffer16x64, buffer16x16
        )

    def fillFloatLumaFromBufferImage(self, img, luma):
        # --- CHANGED: fully numpy, no flatten/list ---
        rgb_image = img.convert("RGB")
        arr = np.asarray(rgb_image, dtype=np.float64)

        luma[:, :] = (
            self.LUMA_FROM_R_COEFF * arr[:, :, 0]
            + self.LUMA_FROM_G_COEFF * arr[:, :, 1]
            + self.LUMA_FROM_B_COEFF * arr[:, :, 2]
        )

    def findHash256FromFloatLuma(
        self,
        fullBuffer1,
        fullBuffer2,
        numRows,
        numCols,
        buffer64x64,
        buffer16x64,
        buffer16x16,
    ):
        windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)

        self.boxFilter(fullBuffer1, fullBuffer2, numRows, numCols,
                       windowSizeAlongRows, windowSizeAlongCols)

        fullBuffer1 = fullBuffer2

        self.decimateFloat(fullBuffer1, numRows, numCols, buffer64x64)
        self.dct64To16(buffer64x64, buffer16x64, buffer16x16)
        return self.dctOutput2hash(buffer16x16)

    @classmethod
    def decimateFloat(cls, in_, inNumRows, inNumCols, out):
        # --- CHANGED: vectorized ---
        i = ((np.arange(64) + 0.5) * inNumRows / 64).astype(int)
        j = ((np.arange(64) + 0.5) * inNumCols / 64).astype(int)
        out[:, :] = in_[i[:, None], j[None, :]]

    def dct64To16(self, A, T, B):
        # --- CHANGED: pure numpy matmul ---
        B[:, :] = self.DCT_matrix @ A @ self.DCT_matrix.T

    def dctOutput2hash(self, dctOutput16x16):
        # --- CHANGED: fully vectorized ---
        median = np.median(dctOutput16x16)
        hash_arr = (dctOutput16x16 > median).astype(int)

        # flip to match original indexing [15-i,15-j]
        hash_arr = np.flipud(np.fliplr(hash_arr))

        return ImageHash(hash_arr.reshape((256,)))

    @classmethod
    def computeBoxFilterWindowSize(cls, dimension):
        return int(
            (dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
            / cls.FIND_WINDOW_SIZE_DIVISOR
        )

    @classmethod
    def boxFilter(cls, input, output, rows, cols, rowWin, colWin):
        # --- CHANGED: fully numpy, no reshape/list conversion ---
        halfColWin = int((colWin + 2) / 2)
        halfRowWin = int((rowWin + 2) / 2)

        arr = input  

        sat = np.zeros((rows + 1, cols + 1), dtype=np.float64)
        sat[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)

        i_idx = np.arange(rows)
        j_idx = np.arange(cols)

        xmin = np.maximum(0, i_idx - halfRowWin)
        xmax = np.minimum(rows, i_idx + halfRowWin)
        ymin = np.maximum(0, j_idx - halfColWin)
        ymax = np.minimum(cols, j_idx + halfColWin)

        s = (sat[xmax[:, None], ymax[None, :]]
           - sat[xmin[:, None], ymax[None, :]]
           - sat[xmax[:, None], ymin[None, :]]
           + sat[xmin[:, None], ymin[None, :]])

        area = (xmax - xmin)[:, None] * (ymax - ymin)[None, :]

        output[:, :] = s / area  # --- CHANGED ---

    @classmethod
    def prettyHash(cls, hash):
        if len(hash.hash) != 256:
            print("This function only works with 256-bit hashes.")
            return
        return np.array(hash.hash).astype(int).reshape((16, 16))


if __name__ == "__main__":
    import sys
    find = FINDHasher()
    for filename in sys.argv[1:]:
        h = find.fromFile(filename)
        print("{},{}".format(h, filename))
        print(find.prettyHash(h))