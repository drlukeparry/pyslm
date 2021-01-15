# -*- coding: utf-8 -*-
from .context import pyslm

import unittest
import platform
import tempfile

class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        assert True


if __name__ == '__main__':
    unittest.main()
