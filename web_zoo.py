import math
import random
import numpy as np
from spider_web import Node, Edge, Web
# from spider_web import EdgeOfMaterial


def square_web_with_offset_center():
    p1 = Node([-1,-1,0], [0,0,0], 1.0, pinned=True)
    p2 = Node([1,-1,0], [0,0,0], 1.0, pinned=True)
    p3 = Node([1,1,0], [0,0,0], 1.0, pinned=True)
    p4 = Node([-1,1,0], [0,0,0], 1.0, pinned=True)
    p5 = Node([0.5,0.5,0], [0,0,0], 1.0, pinned=False)

    e1 = Edge(p1, p2, spring_constant=1.0)
    e2 = Edge(p2, p3, spring_constant=1.0)
    e3 = Edge(p3, p4, spring_constant=1.0)
    e4 = Edge(p4, p1, spring_constant=1.0)

    e5 = Edge(p1, p5, spring_constant=1.0)
    e6 = Edge(p2, p5, spring_constant=1.0)
    e7 = Edge(p3, p5, spring_constant=1.0)
    e8 = Edge(p4, p5, spring_constant=1.0)

    web = Web([e1, e2, e3, e4, e5, e6, e7, e8])
    return web

def square_web_with_z_offset_center():
    p1 = Node([-1,-1,0], [0,0,0], 1.0, pinned=True)
    p2 = Node([1,-1,0], [0,0,0], 1.0, pinned=True)
    p3 = Node([1,1,0], [0,0,0], 1.0, pinned=True)
    p4 = Node([-1,1,0], [0,0,0], 1.0, pinned=True)
    p5 = Node([0.5,0.5,0.5], [0,0,0], 1.0, pinned=False)

    e1 = Edge(p1, p2, spring_constant=1.0)
    e2 = Edge(p2, p3, spring_constant=1.0)
    e3 = Edge(p3, p4, spring_constant=1.0)
    e4 = Edge(p4, p1, spring_constant=1.0)

    e5 = Edge(p1, p5, spring_constant=1.0)
    e6 = Edge(p2, p5, spring_constant=1.0)
    e7 = Edge(p3, p5, spring_constant=1.0)
    e8 = Edge(p4, p5, spring_constant=1.0)

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
        e = Edge(p1, p2, spring_constant=5.0)
        edges.append(e)
    web = Web(edges)
    return web


def _validate_edge_type(edge_type):
    """
    Maybe should update this to validate all the inputs, like rest_length etc.
    """
    assert edge_type is not None
    assert edge_type in ['spring_constant', 'stiffness_tension']


def nodes_around_circle(num_points=None,
                        rad_circle=None,
                        num_segments_per_line=None,
                        pinned=False,
                        damping_coefficient=0.0):
    assert None not in [num_points, rad_circle, num_segments_per_line]
    points = []
    for i in range(num_points):
        radians = 2*(math.pi)*(i / num_points)
        next_radians = 2*(math.pi)*((i+1) / num_points)

        x = rad_circle*math.cos(radians)
        y = rad_circle*math.sin(radians)
        points.append(Node([x,y,0.0],[0,0,0], pinned=pinned, damping_coefficient=damping_coefficient, intersection=True))

        next_x = rad_circle*math.cos(next_radians)
        next_y = rad_circle*math.sin(next_radians)
        for i in range(1, num_segments_per_line):
            pos = [x + (next_x-x)*(i/num_segments_per_line), y + (next_y-y)*(i/num_segments_per_line), 0]
            points.append(Node(pos, [0,0,0], pinned=pinned, damping_coefficient=damping_coefficient, intersection=False))
    return points


def connect_points(list_of_points,
                   rest_length=None,
                   spring_constant=None,
                   stiffness=None,
                   tension=None,
                   edge_type=None,
                   complete_circle=False,
                   connect_points_with=None
                  ):
    """
    I'm still trying to figure out the best way to do this... At the moment, I think I should
    have connect_points always have a list_of_points that are center-out, or clockwise.

    Then, I scan through for intersection points.

    The first one, it's before will be itself to start. The last one, its after will be itself.
    If it's complete circle, that'll be overwritten.

    But, what's the implications? For example, if my guy is on the edge, and he moves right, what happens?
    That's actually perfect. It'll just stay still.

    So, if the first point doesn't have a before, I set it to itself. If it does, I don't!
    Same thing with the last point, pretty much.
    """
    # print(connect_points_with)
    assert len(list_of_points) > 1
    assert edge_type in ['spring_constant', 'stiffness_tension']
    assert connect_points_with in [None, 'radial', 'azimuthal']

    edges = []

    if connect_points_with is not None:
        most_recent_intersection = list_of_points[0]
        assert most_recent_intersection.intersection == True
        if connect_points_with == 'radial':
            if not hasattr(most_recent_intersection, 'radial_before'):
                most_recent_intersection.radial_before = most_recent_intersection
            if not hasattr(most_recent_intersection, 'radial_after'):
                most_recent_intersection.radial_after = most_recent_intersection
        if connect_points_with == 'azimuthal':
            if not hasattr(most_recent_intersection, 'azimuthal_before'):
                most_recent_intersection.azimuthal_before = most_recent_intersection
            if not hasattr(most_recent_intersection, 'azimuthal_after'):
                most_recent_intersection.azimuthal_after = most_recent_intersection

    for i in range(len(list_of_points) - 1):
        p1 = list_of_points[i]
        p2 = list_of_points[i+1]

        if connect_points_with is not None:
            if p2.intersection == True:
                # print('connecting point with {}'.format(connect_points_with))
                if connect_points_with == 'radial':
                    p2.radial_before = most_recent_intersection
                    most_recent_intersection.radial_after = p2
                    if not hasattr(p2, 'radial_after'):
                        p2.radial_after = p2
                    p2.radial_after = p2 # This will get overwritten if there's another, and won't if there isn't
                    most_recent_intersection = p2
                elif connect_points_with == 'azimuthal':
                    p2.azimuthal_before = most_recent_intersection
                    most_recent_intersection.azimuthal_after = p2
                    if not hasattr(p2, 'azimuthal_after'):
                        p2.azimuthal_after = p2
                    most_recent_intersection = p2

        edge = Edge(p1, p2,
                    rest_length=rest_length,
                    spring_constant=spring_constant,
                    stiffness=stiffness,
                    tension=tension,
                    edge_type=edge_type)
        edges.append(edge)

    if complete_circle:
        if connect_points_with == 'radial':
            list_of_points[0].radial_before = most_recent_intersection
            most_recent_intersection.radial_after = list_of_points[0]
        elif connect_points_with == 'azimuthal':
            list_of_points[0].azimuthal_before = most_recent_intersection
            most_recent_intersection.azimuthal_after = list_of_points[0]

        edge = Edge(list_of_points[-1], list_of_points[0],
                    rest_length=rest_length,
                    spring_constant=spring_constant,
                    stiffness=stiffness,
                    tension=tension,
                    edge_type=edge_type)
        edges.append(edge)
    # else:
        # Design choice here: if it's not completing circle, then the "after" will connect to itself...

    return edges

def nodes_edges_around_circle(num_points=None,
                              rad_circle=None,
                              rest_length=None,
                              spring_constant=None,
                              tension=None,
                              stiffness=None,
                              edge_type=None,
                              pinned=False,
                              num_segments_per_line=None,
                              damping_coefficient=0.0):
    # Input Validation.
    assert None not in [num_points, rad_circle, num_segments_per_line, edge_type]

    assert edge_type in ['spring_constant', 'stiffness_tension']
    if edge_type == 'spring_constant':
        assert spring_constant is not None and rest_length is not None
    if edge_type == 'stiffness_tension':
        assert stiffness is not None and tension is not None

    points = nodes_around_circle(num_points=num_points,
                                 rad_circle=rad_circle,
                                 num_segments_per_line=num_segments_per_line,
                                 pinned=pinned,
                                 damping_coefficient=damping_coefficient)

    # print('boom?')
    edges = connect_points(points,
                           rest_length=rest_length,
                           spring_constant=spring_constant,
                           stiffness=stiffness,
                           tension=tension,
                           edge_type=edge_type,
                           complete_circle=True,
                           connect_points_with='azimuthal')
    return points, edges


def connect_two_circles(c1, c2,
                        rest_length=None,
                        spring_constant=None,
                        stiffness=None,
                        tension=None,
                        edge_type=None,
                        connect_every=1,
                        num_segments_per_line=1,
                        damping_coefficient=0.0):
    """
    connect_every is there because you only want to connect the points that are along
    radial lines, not intermediary points. num_segments_per_line gives the connections some
    wiggle-power.

    Assume that c1 is "Inner" and c2 is "Outer". Then, since we work our way out,
    So, we start with the center piece. We set its prior to itself, and its future to itself.
    Then, we set its future to the next one.

    Then, the next time, we set its prior to itself, and its future to itself.

    That's bad! Because, on the 2-3 circle connect, we'll be overwriting how 2 connects back to 1.

    How to deal with this? One easy way would be to connect the circles all at once. But that's
    a lot of code change. I think I should just only overwrite if it doesn't have it already.


    W
    """
    assert edge_type in ['spring_constant', 'stiffness_tension']
    zipped = zip(c1, c2)
    edges = []
    for i, (p1, p2) in enumerate(zipped):
        if i % connect_every != 0:
            continue
        assert p1.intersection and p2.intersection #Fair enough for now.
        points = []
        points.append(p1)
        for j in range(1, num_segments_per_line):
            pos = p1.loc + ((p2.loc - p1.loc) * (j / num_segments_per_line))
            pos = pos.tolist()
            points.append(Node(pos, [0,0,0], pinned=False, damping_coefficient=damping_coefficient, intersection=False))
        points.append(p2)

        new_edges = connect_points(points,
                                   rest_length=rest_length,
                                   spring_constant=spring_constant,
                                   stiffness=stiffness,
                                   tension=tension,
                                   edge_type=edge_type,
                                   connect_points_with='radial')
        edges.extend(new_edges)
    # print('returning connection points')
    return edges

def connect_center_point_to_first_circle(center_point, first_circle):
    # print('connect center point called...')
    # first_circle should be a list of points.
    intersection_points = [p for p in first_circle if getattr(p, 'intersection', False)]
    # Ideally, this would be divisible by 4. Right? For now, I can decide to enforce that...
    if len(intersection_points) == 0 or len(intersection_points) % 4 != 0:
        raise Exception("We got {} intersection points, when we wanted something divisible by 4!")
    num_to_count_by = len(intersection_points) / 4
    points = intersection_points[::int(num_to_count_by)]
    assert len(points) == 4
    center_point.radial_after = points[0]
    center_point.radial_before = points[2]
    center_point.azimuthal_after = points[1]
    center_point.azimuthal_before = points[3]
    # if len(intersection_points)

def radial_web(radius=None,
               num_radial=None,
               num_azimuthal=None,
               stiffness_radial=None,
               tension_radial=None,
               spring_constant_radial=None,
               rest_length_radial=None,
               stiffness_azimuthal=None,
               tension_azimuthal=None,
               spring_constant_azimuthal=None,
               rest_length_azimuthal=None,
               damping_coefficient=0.0,
               edge_type=None,
               num_segments_per_radial=1,
               num_segments_per_azimuthal=1,
              ):
    assert edge_type in ["spring_constant", "stiffness_tension"]
    assert None not in [radius, num_radial, num_azimuthal,
                        num_segments_per_radial, num_segments_per_azimuthal]

    center_point = Node([0,0,0],[0,0,0], pinned=False, damping_coefficient=damping_coefficient, intersection=True)
    center_degen_circle = [center_point]*(num_radial*num_segments_per_azimuthal)

    edges = []
    list_of_circle_points = []
    list_of_circle_points.append(center_degen_circle)
    for i in range(1, num_azimuthal):
        # These set up the circles, not the lines
        pinned = (i==(num_azimuthal-1)) #Pins only the last layer...
        ps, es = nodes_edges_around_circle(num_points=num_radial,
                                           rad_circle=(radius * (i/num_azimuthal)),
                                           rest_length=rest_length_azimuthal,
                                           spring_constant=spring_constant_azimuthal,
                                           stiffness=stiffness_azimuthal,
                                           tension=tension_azimuthal,
                                           pinned=pinned,
                                           edge_type=edge_type,
                                           damping_coefficient=damping_coefficient,
                                           num_segments_per_line=num_segments_per_azimuthal
                                          )
        edges.extend(es)
        list_of_circle_points.append(ps)

    for j in range(len(list_of_circle_points)-1):
        ce = connect_two_circles(list_of_circle_points[j], list_of_circle_points[j+1],
                                 rest_length=rest_length_radial,
                                 spring_constant=spring_constant_radial,
                                 tension=tension_radial,
                                 stiffness=stiffness_radial,
                                 edge_type=edge_type,
                                 damping_coefficient=damping_coefficient,
                                 connect_every=num_segments_per_azimuthal,
                                 num_segments_per_line=num_segments_per_radial)
        edges.extend(ce)


    web = Web(edges, center_point=center_point)

    connect_center_point_to_first_circle(center_point, list_of_circle_points[1])

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
        e = Edge(p1, p2, rest_length=per_spring_rest_length, spring_constant=12.0)
        edges.append(e)


    web = Web(edges)
    return web


def single_segment_stiffness_tension():
    p1, p2 = [Node([0,0,0],[0,0,0], pinned=True), Node([1,0,0],[0,0,0],pinned=False)]
    edges = [EdgeOfMaterial(p1, p2, tension=2.0, stiffness=4.0)]
    web = Web(edges)
    return web


def single_segment():
    p1, p2 = [Node([0,0,0],[0,0,0], pinned=True), Node([1,0,0],[0,0,0],pinned=False)]
    edges = [Edge(p1, p2, rest_length=0.9, spring_constant=50.0)]
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


def force_point_to_sine(point, direction_vector=[0.0, 0.0, 1.0], amplitude=0.0, period=1.0, delay=0.0):
    """
    Badly named. It forces it to move to sine, it doesn't apply a force.
    """
    def function_of_concern(timestep):
        timestep = max(timestep - delay, 0)
        _direction_vector = np.asarray(direction_vector)
        assert _direction_vector.shape == (3,)
        phase = (2*math.pi*timestep) / period
        sined_phase = math.sin(phase)
        new_pos = _direction_vector * (sined_phase * amplitude)
        point.loc[...] = new_pos

    return function_of_concern

def sine_oscillate_point(point, direction_vector=[0.0, 0, 1.0], max_force=0.0, period=1.0, delay=0.0):
    """
    Period is the number of timesteps that elapse before it repeats itself.

    This returns a function, that is just dependent on timestep...
    """
    def function_of_concern(timestep):
        timestep = max(timestep - delay, 0)
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



def random_oscillate_point_one_dimension(point, direction_vector=[1.0,0,0], max_force=0.0, delay=0.0):
    # Random force, From uniform distribution, -1 to 1.
    def function_of_concern(timestep):
        # TODO: Normalize direction vector...
        if timestep < delay:
            return
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

def impulse_to_point(point, direction_vector=[0,0,1.0], force=0.0, force_time=0.0, delay=0.0):
    def function_of_concern(timestep):
        shifted_timestep = timestep - delay
        if shifted_timestep < 0:
            return
        if shifted_timestep > force_time:
            return

        _direction_vector = np.asarray(direction_vector)
        assert _direction_vector.shape == (3,)
        force_vector = _direction_vector * force
        point.acc += force_vector

    return function_of_concern
