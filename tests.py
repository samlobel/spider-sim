import unittest
from unittest import TestCase

import numpy as np

from spider_web import Node, Edge

class EdgeTest(TestCase):

    @unittest.expectedFailure
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

    @unittest.expectedFailure
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

    @unittest.expectedFailure
    def test_update_point_forces_does_nothing_when_points_overlap(self):
        p1 = Node([1,2,3], [0,0,0], 1.0)
        p2 = Node([1,2,3], [0,0,0], 1.0)
        e1 = Edge(p1, p2, 1.0)
        e1.update_point_forces()
        np.testing.assert_array_equal(p1.acc, np.asarray([0,0,0]))
        np.testing.assert_array_equal(p2.acc, np.asarray([0,0,0]))

    @unittest.expectedFailure
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


    @unittest.expectedFailure
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

from web_zoo import radial_web
from spider_web import get_distance_between_two_points

class DistanceTest(TestCase):
    """
    This is going to test the distance function.
    """
    def setUp(self):
        self.web = radial_web(radius=10,
                         num_radial=8,
                         num_azimuthal=8,
                         stiffness_radial=10,
                         tension_radial=10,
                         stiffness_azimuthal=10,
                         tension_azimuthal=10,
                         damping_coefficient=0.0,
                         edge_type='stiffness_tension',
                         num_segments_per_radial=5,
                         num_segments_per_azimuthal=5)


    def test_center_point_out_radius(self):
        p1 = self.web.center_point
        p2 = self.web.center_point.radial_after.radial_after.radial_after.radial_after
        d = get_distance_between_two_points(p1, p2)
        d2 = get_distance_between_two_points(p2, p1)
        self.assertEqual(d, 4)
        self.assertEqual(d, d2)

    def test_center_down_works(self):
        p1 = self.web.center_point
        p2 = self.web.center_point.radial_before.radial_after.radial_after.radial_after
        d = get_distance_between_two_points(p1, p2)
        d2 = get_distance_between_two_points(p2, p1)
        self.assertEqual(d, 4)
        self.assertEqual(d, d2)

    def test_up_vs_down_works(self):
        p1 = self.web.center_point.radial_after.radial_after.radial_after.radial_after
        p2 = self.web.center_point.radial_before.radial_after.radial_after.radial_after
        d = get_distance_between_two_points(p1, p2)
        d2 = get_distance_between_two_points(p2, p1)
        self.assertEqual(d, 4)
        self.assertEqual(d, d2)

    def test_works_traversing_azimuthal(self):
        """
        This one is a bit more complicated. If we go out a bunch, and then test two on the
        same azimuthal, it should be that number away.
        """
        out = self.web.center_point.radial_after.radial_after.radial_after.radial_after
        about = out.azimuthal_after.azimuthal_after
        d = get_distance_between_two_points(out, about)
        d2 = get_distance_between_two_points(about, out)
        self.assertEqual(d, 2)
        self.assertEqual(d, d2)

    def test_works_on_different_radial_and_azimuthal(self):
        out = self.web.center_point.radial_after.radial_after.radial_after.radial_after
        about = out.radial_before.azimuthal_after.azimuthal_after
        d = get_distance_between_two_points(out, about)
        d2 = get_distance_between_two_points(about, out)
        self.assertEqual(d, 3)
        self.assertEqual(d, d2)


if __name__ == '__main__':
    unittest.main()
