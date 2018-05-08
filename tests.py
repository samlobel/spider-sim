import unittest
from unittest import TestCase

import numpy as np

from spider_web import Node, Edge

class EdgeTest(TestCase):
    def test_update_point_forces(self):
        """
        Makes the simplest possible points, at 0 and 1.
        That puts the length at 1. That puts the force at 1. That means that the
        force on p1 should be [1,0,0] and the force on p2 should be [-1,0,0]
        """
        p1 = Node([0,0,0], [0,0,0], 1.0)
        p2 = Node([1,0,0], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        np.testing.assert_array_equal(p1.acc, np.asarray([1,0,0]))
        np.testing.assert_array_equal(p2.acc, np.asarray([-1,0,0]))

    def test_update_point_forces_backwards(self):
        """
        Same as above, but backwards.
        """
        p1 = Node([1,0,0], [0,0,0], 1.0)
        p2 = Node([0,0,0], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        np.testing.assert_array_equal(p1.acc, np.asarray([-1,0,0]))
        np.testing.assert_array_equal(p2.acc, np.asarray([1,0,0]))

    def test_update_point_forces_does_nothing_when_points_overlap(self):
        p1 = Node([1,2,3], [0,0,0], 1.0)
        p2 = Node([1,2,3], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        np.testing.assert_array_equal(p1.acc, np.asarray([0,0,0]))
        np.testing.assert_array_equal(p2.acc, np.asarray([0,0,0]))

    def test_update_point_forces_works_when_points_have_three_dimensions(self):
        """
        I was hoping this was easy but I'm not sure. Let's say one is at the origin and one is
        at [1,1,0]. That means that the spring-force should be sqrt(2).
        But then you need to multiply by the sine and cosine. So it should be sqrt(2)*sqrt(2)/2,
        which is one. So, I do think it works out....
        """
        p1 = Node([1,1,0], [0,0,0], 1.0)
        p2 = Node([0,0,-1], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        np.testing.assert_array_equal(p1.acc, np.asarray([-1,-1,-1]))
        np.testing.assert_array_equal(p2.acc, np.asarray([1,1,1]))


    def test_update_loc_vel(self):
        p1 = Node([0,0,0], [0,0,0], 1.0)
        p2 = Node([1,0,0], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        p1.update_loc_vel()
        p2.update_loc_vel()
        np.testing.assert_array_equal(p1.loc, np.asarray([0,0,0]))
        np.testing.assert_array_equal(p2.loc, np.asarray([1,0,0]))
        np.testing.assert_array_equal(p1.vel, np.asarray([1,0,0]))
        np.testing.assert_array_equal(p2.vel, np.asarray([-1,0,0]))


if __name__ == '__main__':
    unittest.main()
