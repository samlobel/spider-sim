"""
This will be where the meat of the spider-web simulation lives. It's essentially just a graph, and
there are going to be two points that the edges are attached to. Then the web will have a set of points,
and a set of edges. It'll have a force-accumulate option, a force-reset option, etc.

https://en.wikipedia.org/wiki/Verlet_integration might be a good place to look
if I can't stabilize my simulation.

https://en.wikipedia.org/wiki/Symplectic_Euler is actually better, just becaues it's way easier.

Or, just using the average acceleration might do it too.
"""


import numpy as np
from numpy import linalg as LA

from collections import OrderedDict

class Node(object):
    """
    NOTE: I don't use the mass at all, which is really stupid. It should be that I only store the
    force, and I calculate the mass part at the end. Should be an easy change... But for now I'll
    leave it.
    """
    def __init__(self,
                 starting_point,
                 starting_velocity,
                 mass=1.0,
                 damping_coefficient=0.0,
                 pinned=False,
                 **kwargs):
        assert len(starting_point) == 3
        assert len(starting_velocity) == 3
        assert type(mass) == float
        self.loc = np.asarray(starting_point, dtype=np.float32)
        self.vel = np.asarray(starting_velocity, dtype=np.float32)
        self.acc = np.zeros((3,), dtype=np.float32)

        self.mass = mass
        self.damping_coefficient=damping_coefficient
        self.pinned = pinned

        # Now, set all KWARGS to the right values.
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def update_loc_vel(self, timestep=1):
        """
        Timestep is a tough one, but what it means is it's going to use that to determine
        the change in loc and vel.

        If it's pinned, then you just keep it fixed.

        This assumes that the acc refers to all EXTERNAL forces, and has been calculated. We still need
        to add in damping.
        """
        if not self.pinned:
            if self.damping_coefficient:
                self.acc -= (self.vel * self.damping_coefficient)
            self.vel += (self.acc * timestep)
            self.loc += (self.vel * timestep)
        # self._zero_acc()


    def _zero_acc(self):
        self.acc.fill(0)
        # self.acc[...] = np.zeros((3,), dtype=np.float32)


def _validate_edge_type(edge_type):
    assert edge_type is not None
    assert edge_type in ['spring_constant', 'stiffness_tension']

class Edge(object):
    """
    The basic edge object. Can update its point's forces based on its stretchiness.

    NOTE: Set stiffness to 0 if you want rest_length to be zero... It just works out...

    Setting stiffness==tension means that it starts stretched twice as far as it would have been.
    Setting stiffness==3*tension means that its rest length is 3/4 the original length.
    """
    def __init__(self,
                 p1,
                 p2,
                 rest_length=None,
                 spring_constant=None,
                 stiffness=None,
                 tension=None,
                 edge_type=None):
        assert isinstance(p1, Node)
        assert isinstance(p2, Node)
        _validate_edge_type(edge_type)

        self.p1 = p1
        self.p2 = p2
        self.both_pinned = self.p1.pinned and self.p2.pinned
        if edge_type == 'spring_constant':
            assert spring_constant is not None and rest_length is not None
            self._initialize_spring_constant(spring_constant, rest_length)
        if edge_type == 'stiffness_tension':
            assert stiffness is not None and tension is not None
            self._initialize_stiffness_tension(tension, stiffness)

    def _initialize_spring_constant(self, spring_constant, rest_length):
        self.spring_constant = spring_constant
        self.rest_length = rest_length

    def _initialize_stiffness_tension(self, tension, stiffness):
        starting_stretched_length = self.edge_vector_norm
        if stiffness == 0:
            self.rest_length = 0.0
            self.spring_constant = (tension / starting_stretched_length)
        else:
            self.rest_length = (stiffness*starting_stretched_length) / (tension + stiffness) #TODO: Proof???
            self.spring_constant = (tension / (starting_stretched_length - self.rest_length)) #Pretty clear.

    def _update_point_forces_zero_rest_length(self):
        """
        if rest-length is zero, the computation becomes much quicker.
        FORCE = (edge_vector * spring_constant) is all you need to do.
        """
        force = (self.edge_vector * self.spring_constant)
        self.p1.acc += force
        self.p2.acc -= force

    def _update_point_forces_with_rest_length(self):
        """

        The starting formula is (edge_vector - rest_vector)  * spring_constant. But we don't know
        rest_vector, only rest_length. So, this is equivalent to

        FORCE = (edge_vector - (rest_length * normalized_edge_vector)) * spring_constant ==
        normalized_edge_vector * (edge_vector_norm - rest_length) * spring_constant ==
        (edge_vector / edge_vector_norm) * (edge_vector_norm - rest_length) * spring_constant ==
        (edge_vector) * ((edge_vector_norm - rest_length) * spring_constant / edge_vector_norm)

        This formulation only requires taking the norm once, and has the fewest vector algebra
        steps.
        """

        edge_vector = self.edge_vector
        edge_vector_norm = LA.norm(edge_vector)
        scaling_constant = ((edge_vector_norm - self.rest_length) * (self.spring_constant / (edge_vector_norm + 1e-8)))
        force = scaling_constant * edge_vector

        self.p1.acc += force
        self.p2.acc -= force

    def update_point_forces(self):
        if self.both_pinned:
            return
        if self.rest_length:
            self._update_point_forces_with_rest_length()
        else:
            self._update_point_forces_zero_rest_length()


    @property
    def edge_vector(self):
        return self.p2.loc - self.p1.loc

    @property
    def edge_vector_norm(self):
        edge_vector = self.edge_vector
        return LA.norm(edge_vector)

# class EdgeOfMaterial(Edge):
#     """
#     A material can be defined by its stiffness. A spider probably puts a constant tension on a material
#     as it weaves.
#     """
#
#     def __init__(self, p1, tension=None, stiffness=None):
#         assert isinstance(p1, Node)
#         assert isinstance(p2, Node)
#         assert tension is not None
#         assert stiffness is not None
#         starting_stretched_length = self.edge_vector_norm
#         rest_length = (stiffness*starting_stretched_length) / (tension + stiffness) #Do the units even work?! I think so actually.
#         spring_constant = (stiffness/rest_length)
#         self.rest_length = rest_length
#         self.spring_constant = spring_constant
#
#
# class EdgeOfMaterial(Edge):
#     def __init__(self, p1, p2, tension, stiffness):
#         self.p1 = p1
#         self.p2 = p2
#         starting_stretched_length = self.edge_vector_norm
#         rest_length = (stiffness*starting_stretched_length) / (tension + stiffness) #Do the units even work?! I think so actually.
#         spring_constant = (stiffness/rest_length)
#         self.rest_length = rest_length
#         self.spring_constant = spring_constant
#         # print("Spring coefficient: {}".format(spring_constant))
#         # print("Rest Length: {}".format(rest_length))
#
#
# class ZeroRestLengthEdge(Edge):
#     """
#     This should be much faster to compute, for starters. How is it going to work? Well, we'll use
#     the starting length and starting tension to figure out a spring-constant. Then, we'll
#     be able to efficiently calculate where it goes
#     """
#     def __init__(self, p1, p2, tension=None):
#         assert tension is not None
#         self.p1 = p1
#         self.p2 = p2
#         self.rest_length = 0
#         starting_stretched_length = self.edge_vector_norm
#         self.spring_constant = (tension / starting_stretched_length)
#
#     def update_point_forces(self):
#         force = (self.edge_vector * self.spring_constant)
#         self.p1.acc += force
#         self.p2.acc -= force




class Web(object):
    def __init__(self, edges, force_func=None, movement_func=None, center_point=None):
        """
        Force func is something that applies a force. For example,
        a bug that's trapped, or an oscillator in the center.
        """
        self.edge_set = set(edges)
        self.point_set = self._collect_points()
        self.edge_list = list(self.edge_set)
        self.point_list = list(self.point_set)
        self.num_steps = 0
        self.force_func = force_func
        self.movement_func = movement_func
        self.center_point = center_point

        # The keys are points. The values should be a tuple of lists...
        # The first is loc, the second for vel, the third is acc. You can't actually measure pos though.
        self.gather_points = OrderedDict()
        pass

    def reset_gather_points(self):
        for p in self.gather_points:
            self.gather_points[p] = ([],[],[])

    def set_gather_points(self, point_list):
        self.gather_points = OrderedDict()
        for p in point_list:
            self.gather_points[p] = ([],[],[])

    def number_of_gathered_samples(self):
        for p in self.gather_points:
            return len(self.gather_points[p][0])

    def record_gather_points(self):
        for p in self.gather_points:
            self.gather_points[p][0].append(np.copy(p.loc))
            self.gather_points[p][1].append(np.copy(p.vel))
            self.gather_points[p][2].append(np.copy(p.acc))

    def step(self, timestep=1.0):
        """
        First, you collect the forces. Then, you update pos, vel, acc of each point.
        """
        self.num_steps += timestep

        for p in self.point_list:
            p._zero_acc()

        for e in self.edge_set:
            e.update_point_forces()

        if self.force_func is not None:
            self.force_func(self.num_steps)

        for p in self.point_set:
            p.update_loc_vel(timestep=timestep)

        if self.movement_func is not None:
            self.movement_func(self.num_steps)

    def _collect_points(self):
        point_set = set([])
        for e in self.edge_set:
            point_set.add(e.p1)
            point_set.add(e.p2)

        return point_set

class Spider(object):
    def __init__(self, web, starting_point):
        # we really don't need the web argument, do we. I guess it doesn't hurt?
        self.web = web
        self.current_point = starting_point
        assert starting_point.intersection == True
        # import ipdb; ipdb.set_trace()

        assert starting_point.radial_before
        assert starting_point.radial_after
        assert starting_point.azimuthal_before
        assert starting_point.azimuthal_after

        # self.gather_points = OrderedDict()
        self.set_gather_points()


    def move(self, direction):
        assert direction in ["radial_before", "radial_after", "azimuthal_before", "azimuthal_after"]
        self.current_point = getattr(self.current_point, direction)

        assert self.current_point.radial_before
        assert self.current_point.radial_after
        assert self.current_point.azimuthal_before
        assert self.current_point.azimuthal_after

        # self.gather_points = OrderedDict()
        self.set_gather_points()
        pass

    def get_gather_points(self):
        return [self.current_point.radial_before,
                self.current_point.radial_after,
                self.current_point.azimuthal_before,
                self.current_point.azimuthal_after]

    def reset_gather_points(self):
        for p in self.gather_points:
            self.gather_points[p] = ([],[],[])

    def set_gather_points(self):
        # new_gather_points = [self.current_point.radial_before,
        #                      self.current_point.radial_after,
        #                      self.current_point.azimuthal_before,
        #                      self.current_point.azimuthal_after]
        new_gather_points = self.get_gather_points()
        # It should set gather points in a particular order, the order of the directions.
        self.gather_points = OrderedDict()
        for p in new_gather_points:
            self.gather_points[p] = ([],[],[])

    def number_of_gathered_samples(self):
        for p in self.gather_points:
            return len(self.gather_points[p][0])

    def record_gather_points(self):
        for p in self.gather_points:
            # print("p vel: {}".format(p.vel))
            # print("p acc: {}".format(p.acc))
            self.gather_points[p][0].append(np.copy(p.loc))
            self.gather_points[p][1].append(np.copy(p.vel))
            self.gather_points[p][2].append(np.copy(p.acc))

    def vectorize_gather_point_dict(
        self,
        recording_size=500,
        squash_to_energy=True,
        include_radial_distance=False):

        """Each leg is a channel, each sense is a channel, and then time is a type of channel.
        I think, the we we could do it, is just say vel and acc are the channels, and concatenate them.
        And then, do that for each leg. So, you get 6 dim * 4 legs = 24 channels. So, it'll be a
        conv1d, with 24 channels.
        It would be better if we really did it with the right dimensionality, but I don't want to.

        Fork, I have to get rid of the accelerations!

        So, it's 4 points, so dim-1 is 4. Then there's 3 dimensions, loc, vel, acc. We want to get rid
        of loc. So, that's the second dimension. The 3rd is the 500. The 4th is the xyz measuremnets,
        for example of velocity.
        So, it should be a[:,1:,:,:]
        """

        assert self.number_of_gathered_samples() == recording_size

        if include_radial_distance and not squash_to_energy:
            raise Exception("Option not supported at this time.")

        if include_radial_distance:
            assert hasattr(self.current_point, 'distance_from_center')

        values = self.gather_points.values()
        values_np = np.asarray(list(values))
        without_loc = values_np[:,1:,:,:]
        time_series_last = without_loc.transpose(0,1,3,2) # Switch the last two dimensions.
        flattened = np.reshape(time_series_last, (24, recording_size))

        if not squash_to_energy:
            return flattened

        flattened_energy = np.mean(flattened * flattened, axis=-1)
        assert flattened_energy.shape == (24,)
        if not include_radial_distance:
            return flattened_energy

        new_array = np.empty((25,), np.float32)
        new_array[0:24] = flattened_energy
        new_array[24] = self.current_point.distance_from_center #That's the last one. Zero indexing...

        return new_array


VALID_DIRECTIONS = ["radial_before", "radial_after", "azimuthal_before", "azimuthal_after"]

def get_points_neighbors(point):
    connection_points = [getattr(point, val, None) for val in VALID_DIRECTIONS]
    assert None not in connection_points
    return connection_points

def ensure_point_is_valid_intersection(point):
    assert point.intersection == True
    connection_points = get_points_neighbors(point)
    assert None not in connection_points


def get_distance_between_two_points(p1, p2):
    # NOTE This NEEDS tests! Too complicated to trust.

    ensure_point_is_valid_intersection(p1)
    ensure_point_is_valid_intersection(p2)

    if p1 == p2:
        # print("Trying to compare distance between two points that are the same point!!!")
        return 0
        # raise Exception()

    seen = set()
    frontier = set()
    new_frontier = set()
    frontier.add(p1)
    distance = 0
    while True:
        # End condition: if p2 is in the frontier, you're done!
        if p2 in frontier:
            return distance
        # Now, you've seen and checked the frontier
        seen = seen.union(frontier)
        # First, get the new frontier
        for point in frontier:
            neighbors = get_points_neighbors(point)
            new_frontier = new_frontier.union(neighbors)
        # which involves only choosing new stuff.
        new_frontier = new_frontier.difference(seen)
        # Make sure they're all good...
        for point in new_frontier:
            ensure_point_is_valid_intersection(point)
        # Then, set the frontier to the new_frontier, and increment your count.
        frontier = new_frontier
        new_frontier = set()
        distance += 1
