import unittest
from polygrav import Polyhedron
from pathlib import Path
import numpy as np


test_dir = Path(__file__).parent.absolute()


class TestEuropa(unittest.TestCase):
    def test_europa(self):
        location = test_dir / 'samples' / 'europa.obj'
        with open(location, 'r') as f:
            p = np.array([100000, 100000, 100000])
            eu = Polyhedron.init_from_obj_file(f, scale=1000)
            assert np.isclose(eu.U(p), -0.4943641100160493)
            gp = np.array([-1.6507534739799598e-06, -1.6080133022596955e-06, -1.6414782897683391e-06])
            assert np.alltrue(np.isclose(eu.g(p), gp))


if __name__ == '__main__':
    unittest.main()
