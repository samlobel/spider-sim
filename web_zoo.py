import math
import random
import numpy as np
from spider_web import Node, Edge, Web, EdgeOfMaterial


def square_web_with_offset_center():
    p1 = Node([-1,-1,0], [0,0,0], 1.0, pinned=True)
    p2 = Node([1,-1,0], [0,0,0], 1.0, pinned=True)
    p3 = Node([1,1,0], [0,0,0], 1.0, pinned=True)
    p4 = Node([-1,1,0], [0,0,0], 1.0, pinned=True)
    p5 = Node([0.5,0.5,0], [0,0,0], 1.0, pinned=False)

    e1 = Edge(p1, p2, spring_coefficient=1.0)
    e2 = Edge(p2, p3, spring_coefficient=1.0)
    e3 = Edge(p3, p4, spring_coefficient=1.0)
    e4 = Edge(p4, p1, spring_coefficient=1.0)

    e5 = Edge(p1, p5, spring_coefficient=1.0)
    e6 = Edge(p2, p5, spring_coefficient=1.0)
    e7 = Edge(p3, p5, spring_coefficient=1.0)
    e8 = Edge(p4, p5, spring_coefficient=1.0)

    web = Web([e1, e2, e3, e4, e5, e6, e7, e8])
    return web

def square_web_with_z_offset_center():
    p1 = Node([-1,-1,0], [0,0,0], 1.0, pinned=True)
    p2 = Node([1,-1,0], [0,0,0], 1.0, pinned=True)
    p3 = Node([1,1,0], [0,0,0], 1.0, pinned=True)
    p4 = Node([-1,1,0], [0,0,0], 1.0, pinned=True)
    p5 = Node([0.5,0.5,0.5], [0,0,0], 1.0, pinned=False)

    e1 = Edge(p1, p2, spring_coefficient=1.0)
    e2 = Edge(p2, p3, spring_coefficient=1.0)
    e3 = Edge(p3, p4, spring_coefficient=1.0)
    e4 = Edge(p4, p1, spring_coefficient=1.0)

    e5 = Edge(p1, p5, spring_coefficient=1.0)
    e6 = Edge(p2, p5, spring_coefficient=1.0)
    e7 = Edge(p3, p5, spring_coefficient=1.0)
    e8 = Edge(p4, p5, spring_coefficient=1.0)

    web = Web([e1, e2, e3, e4, e5, e6, e7, e8])
    return web

def wiggle_line(num_points=6, length=5):
    from random import random
    points = []
    points.append(Node([0,0,0], [0,0,0], 1.0, pinned=True, damping_coefficient=1.0))
    for i in range(1, num_points-1):
        x = (i/num_points)*length
        y = random()
        points.append(Node([x,y,0],[0,0,0],1.0, pinned=False, damping_coefficient=1.0))

    points.append(Node([length,0,0],[0,0,0],1.0, pinned=True, damping_coefficient=1.0))

    edges = []
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        e = Edge(p1, p2, spring_coefficient=5.0)
        edges.append(e)
    web = Web(edges)
    return web


# def nodes_edges_around_circle(num_points, rad_circle, spr_c, pinned=False, damping_coefficient=0.0):
#     points = []
#     for i in range(num_points):
#         radians = 2*(math.pi)*(i / num_points)
#         x = rad_circle*math.cos(radians)
#         y = rad_circle*math.sin(radians)
#         points.append(Node([x,y,0.0],[0,0,0], pinned=pinned, damping_coefficient=damping_coefficient))
#
#     edges = [Edge(points[i], points[(i+1)%num_points], spring_coefficient=5.0) for i in range(num_points)]
#     return points, edges

def connect_points_with_stiffness_tension(list_of_points, stiffness=None, tension=None, complete_circle=False):
    assert stiffness is not None
    assert tension is not None
    assert len(list_of_points) > 1

    edges = []
    for i in range(len(list_of_points)-1):
        p1 = list_of_points[i]
        p2 = list_of_points[i+1]
        edges.append(EdgeOfMaterial(p1, p2, stiffness=stiffness, tension=tension))
    if complete_circle:
        edges.append(EdgeOfMaterial(list_of_points[-1], list_of_points[0], stiffness=stiffness, tension=tension))
    return edges

def nodes_edges_around_circle_tension_stiffness(num_points,
                                                rad_circle,
                                                tension,
                                                stiffness,
                                                pinned=False,
                                                num_segments_per_line=1,
                                                damping_coefficient=0.0):
    print("Number of segments per line: {}".format(num_segments_per_line))
    points = []
    for i in range(num_points):
        radians = 2*(math.pi)*(i / num_points)
        next_radians = 2*(math.pi)*((i+1) / num_points)

        x = rad_circle*math.cos(radians)
        y = rad_circle*math.sin(radians)
        points.append(Node([x,y,0.0],[0,0,0], pinned=pinned, damping_coefficient=damping_coefficient))

        next_x = rad_circle*math.cos(next_radians)
        next_y = rad_circle*math.sin(next_radians)
        for i in range(1, num_segments_per_line):
            pos = [x + (next_x-x)*(i/num_segments_per_line), y + (next_y-y)*(i/num_segments_per_line), 0]
            points.append(Node(pos, [0,0,0], pinned=pinned, damping_coefficient=damping_coefficient))

    edges = connect_points_with_stiffness_tension(points, stiffness=stiffness, tension=tension, complete_circle=True)
    # edges = [EdgeOfMaterial(points[i], points[(i+1)%len(points)], tension=tension, stiffness=stiffness) for i in range(len_points)]
    return points, edges

def connect_two_circles_stiffness_tension(c1, c2, tension=None, stiffness=None, connect_every=1, num_segments_per_line=1, damping_coefficient=0.0):
    """
    connect_every tells it to connect every N points
    """
    assert tension is not None
    assert stiffness is not None
    zipped = zip(c1, c2)
    edges = []
    for i, (p1, p2) in enumerate(zipped):
        if i % connect_every == 0:
            # print("Connecting point #{}".format(i))
            points = []
            points.append(p1)
            for j in range(1, num_segments_per_line):
                pos = p1.loc + ((p2.loc - p1.loc) * (j/num_segments_per_line))
                pos = pos.tolist()
                # print(pos)
                points.append(Node(pos, [0,0,0], pinned=False, damping_coefficient=damping_coefficient))
            points.append(p2)
            new_edges = connect_points_with_stiffness_tension(points, stiffness=stiffness, tension=tension, complete_circle=False)
            edges.extend(new_edges)
    print("Returning connection points")
    return edges

# def connect_two_circles_stiffness_tension(c1, c2, tension, stiffness, step_size=1):
#     """
#     Step size is how many you skip over before you make an attachment.
#     """
#     zipped = zip(c1, c2)
#     return [EdgeOfMaterial(p1, p2, tension=tension, stiffness=stiffness) for p1,p2 in zipped]

# def radial_web(radius, num_radial, num_azimuthal, spr_c_radial, spr_c_azimuthal, damping_coefficient=0.0):
#     """
#     Problem with this one: The inner-layers have more inward-pull than the outer ones, because
#     """
#     center_point = Node([0,0,0],[0,0,0], pinned=False, damping_coefficient=damping_coefficient)
#     center_degen_circle = [center_point]*num_radial
#     edges = []
#     point_array = []
#     point_array.append(center_degen_circle)
#     for i in range(1, num_azimuthal):
#         pinned = (i==(num_azimuthal-1))
#         ps, es = nodes_edges_around_circle(num_radial, (radius*(i/num_azimuthal)), spr_c_azimuthal, pinned=pinned, damping_coefficient=damping_coefficient)
#         edges.extend(es)
#         point_array.append(ps) #APPEND!!!!
#
#     for i in range(len(point_array)-1):
#         ce = connect_two_circles(point_array[i], point_array[i+1], spr_c_radial)
#         edges.extend(ce)
#
#     web = Web(edges)
#     return web

def radial_web_tension_stiffness(radius,
                                 num_radial,
                                 num_azimuthal,
                                 tension_radial=None,
                                 tension_azimuthal=None,
                                 stiffness_radial=None,
                                 stiffness_azimuthal=None,
                                 damping_coefficient=0.0,
                                 num_segments_per_radial=1,
                                 num_segments_per_azimuthal=1):

    center_point = Node([0,0,0],[0,0,0], pinned=False, damping_coefficient=damping_coefficient)
    center_degen_circle = [center_point]*(num_radial*num_segments_per_azimuthal)
    edges = []
    point_array = []
    point_array.append(center_degen_circle)
    for i in range(1, num_azimuthal):
        pinned = (i==(num_azimuthal-1))
        ps, es = nodes_edges_around_circle_tension_stiffness(num_radial,
                                                             (radius*(i/num_azimuthal)),
                                                             tension=tension_azimuthal,
                                                             stiffness=stiffness_azimuthal,
                                                             pinned=pinned,
                                                             damping_coefficient=damping_coefficient,
                                                             num_segments_per_line=num_segments_per_azimuthal)
        edges.extend(es)
        point_array.append(ps) #APPEND!!!!

    for i in range(len(point_array)-1):
        ce = connect_two_circles_stiffness_tension(point_array[i],
                                                   point_array[i+1],
                                                   stiffness=stiffness_radial,
                                                   tension=tension_radial,
                                                   connect_every=num_segments_per_azimuthal,
                                                   damping_coefficient=damping_coefficient,
                                                   num_segments_per_line=num_segments_per_radial)
        edges.extend(ce)

    web = Web(edges)
    web.center_point = center_point
    return web



def many_segment_line(num_points=5, length=5, per_spring_rest_length=0.8, wiggle_size=0.0):
    from random import random
    points = []
    points.append(Node([0,0,0], [0,0,0], 1.0, pinned=True, damping_coefficient=0.0))
    for i in range(1, num_points-1):
        x = (i/num_points)*length
        y = random()*wiggle_size
        points.append(Node([x,y,0],[0,0,0],1.0, pinned=False, damping_coefficient=0.0))

    points.append(Node([length,0,0],[0,0,0],1.0, pinned=True, damping_coefficient=0.0))

    edges = []
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        e = Edge(p1, p2, rest_length=per_spring_rest_length, spring_coefficient=12.0)
        edges.append(e)
    web = Web(edges)
    return web

def single_segment_tension_stiffness():
    p1, p2 = [Node([0,0,0],[0,0,0], pinned=True), Node([1,0,0],[0,0,0],pinned=False)]
    edges = [EdgeOfMaterial(p1, p2, tension=2.0, stiffness=4.0)]
    web = Web(edges)
    return web


def single_segment():
    p1, p2 = [Node([0,0,0],[0,0,0], pinned=True), Node([1,0,0],[0,0,0],pinned=False)]
    edges = [Edge(p1, p2, rest_length=0.9, spring_coefficient=50.0)]
    web = Web(edges)
    return web


def deform_web(web, func):
    # Takes in a function, which will take in a point and alter its position at the very beginning.
    points = web.point_set
    for p in points:
        func(p)
    pass

def move_point_to_cosine(point, radius):
    x, y = point.loc[0:2]
    dist_from_center = (x**2 + y**2)**0.5
    print("dist from center: {}".format(dist_from_center))
    prop_from_center = dist_from_center / radius
    print("prop dist from center: {}".format(prop_from_center))
    z = math.cos(prop_from_center*math.pi/2)
    print("cos: {}".format(z))
    point.loc[2] = z*2

def random_from_sphere():
    while True:
        random_vector = 2*(np.random.rand(3)-0.5)
        if np.sum(random_vector**2) <= 1:
            return random_vector
        else:
            print("Chose {}, which is outside of the unit sphere.".format(random_vector))
            continue

def sine_oscillate_point(point, direction_vector=[0.0, 0, 0.1], max_force=0.0, period=1.0):
    """
    Period is the number of timesteps that elapse before it repeats itself.

    This returns a function, that is just dependent on timestep...
    """
    def function_of_concern(timestep):
        # TODO: Normalize direction vector...
        _direction_vector = np.asarray(direction_vector)
        assert _direction_vector.shape == (3,)
        phase = (2*math.pi*timestep) / period
        # print(phase)
        sined_phase = math.sin(phase)
        force = sined_phase * max_force
        force_vector = _direction_vector * force
        point.acc += force_vector

    return function_of_concern

def random_oscillate_point_one_dimension(point, direction_vector=[1.0,0,0], max_force=0.0):
    # Random force, From uniform distribution, -1 to 1.
    def function_of_concern(*args):
        # TODO: Normalize direction vector...
        _direction_vector = np.asarray(direction_vector)
        assert _direction_vector.shape == (3,)
        random_range = 2*(random.random() - 0.5)
        # print(random_range)
        rand_force = random_range * max_force
        force_vector = _direction_vector * rand_force
        point.acc += force_vector

    return function_of_concern

def random_oscillate_point_three_dimensions(point, max_force=0.0):
    # Random force, From uniform distribution, -1 to 1.
    # TODO: Normalize direction vector...
    def function_of_concern(*args):
        random_vector = random_from_sphere()
        # print(random_range)
        rand_force = random_range * max_force
        force_vector = max_force * rand_force
        point.acc += force_vector

    return function_of_concern



# def embedded_squares():
#     squares = []
#     for i in range(3):
#         square = []
#         for vs in [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]:
#             vs = np.asarray(vs)
#             square.append(Node(vs, [0,0,0,] 1.0), pinned=(i==2))