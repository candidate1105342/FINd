import unittest
import numpy as np
from PIL import Image

from find.FINd_opt import FINDHasher as FINDHasher_opt
from find.FINd import FINDHasher as FINDHasher_orig

def _random_image(seed=42, size=(256, 256)):
    """Reproducible random RGB images. Size is (width, height)."""
    rng = np.random.default_rng(seed)
    w, h = size
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class TestFINDHasherOpt(unittest.TestCase):

    def setUp(self):
        self.opt = FINDHasher_opt()
        self.orig = FINDHasher_orig()

    def _hash(self, hasher, img):
        """Convert hash output to NumPy array for comparison."""
        return np.array(hasher.fromImage(img).hash)

    def _hamming(self, a, b):
        """Compute Hamming distance between two hashes."""
        return int(np.sum(a ^ b))

    def test_matches_original(self):
        """
        Verify that the optimized implementation produces identical hashes
        to the reference implementation across a range of inputs.

        Covers:
        - Different random seeds
        - Different aspect ratios
        - Small and large images
        - Different image modes (RGB, grayscale, RGBA)
        """
        cases = [
            ("seed=1",      _random_image(seed=1)),
            ("seed=42",     _random_image(seed=42)),
            ("seed=99",     _random_image(seed=99)),
            ("wide",        _random_image(seed=7, size=(320, 200))),
            ("tall",        _random_image(seed=7, size=(200, 320))),
            ("tiny",        _random_image(seed=7, size=(50, 50))),
            ("oversized",   _random_image(seed=7, size=(700, 500))),
            ("grayscale",   _random_image(seed=3).convert("L")),
            ("rgba",        _random_image(seed=4).convert("RGBA")),
        ]

        for name, img in cases:
            with self.subTest(case=name):
                h_opt = self._hash(self.opt, img)
                h_orig = self._hash(self.orig, img)

                self.assertTrue(
                    np.array_equal(h_opt, h_orig),
                    f"Optimized hash diverged from original on case '{name}'",
                )

    def test_identical_images_distance_zero(self):
        """Same image should produce zero Hamming distance."""
        img = _random_image(seed=123)

        h1 = self._hash(self.opt, img)
        h2 = self._hash(self.opt, img)

        self.assertEqual(self._hamming(h1, h2), 0)

    def test_hash_length(self):
        """Hash should always be 256 bits."""
        img = _random_image()
        h = self.opt.fromImage(img)

        self.assertEqual(h.hash.size, 256)

if __name__ == "__main__":
    unittest.main(verbosity=2)