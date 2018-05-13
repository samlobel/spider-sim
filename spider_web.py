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
                 update_method='symplectic_euler'):
        assert update_method in ['euler', 'symplectic_euler']
        assert len(starting_point) == 3
        assert len(starting_velocity) == 3
        assert type(mass) == float
        self.loc = np.asarray(starting_point, dtype=np.float32)
        self.vel = np.asarray(starting_velocity, dtype=np.float32)
        self.acc = np.zeros((3,), dtype=np.float32)

        self.mass = mass
        self.damping_coefficient=damping_coefficient
        self.pinned = pinned
        self.update_method = update_method

    def _update_euler(self, timestep=1):
        if not self.pinned:
            self.loc += (self.vel * timestep)
            self.vel += (self.acc * timestep)
        self._zero_acc()

    def _update_symplectic(self, timestep=1):
        if not self.pinned:
            self.vel += (self.acc * timestep)
            self.loc += (self.vel * timestep)
        self._zero_acc()

    def update_loc_vel(self, timestep=1):
        """
        Timestep is a tough one, but what it means is it's going to use that to determine
        the change in loc and vel.

        If it's pinned, then you just keep it fixed.

        This assumes that the acc refers to all EXTERNAL forces, and has been calculated. We still need
        to add in damping.
        """

        self.acc -= (self.vel * self.damping_coefficient)
        if self.update_method == 'euler':
            self._update_euler(timestep=timestep)
        else:
            self._update_symplectic(timestep=timestep)

    def _zero_acc(self):
        self.acc[...] = np.zeros((3,), dtype=np.float32)


class Edge(object):
    def __init__(self, p1, p2, rest_length=0.0, spring_coefficient=1.0):
        assert isinstance(p1, Node)
        assert isinstance(p2, Node)
        assert isinstance(rest_length, float)
        assert isinstance(spring_coefficient, float)
        self.p1 = p1
        self.p2 = p2
        self.spring_coefficient = spring_coefficient
        self.rest_length = rest_length

    def initialize_from_tension_and_stiffness(self, p1, p2, tension, stiffness):
        """
        NOTE: I actually think this should be a subclass of Edge... It should just overwrite the constructor.
        NOTE: Another word for stiffness is Elastic Modulus.
        Tension:
            It's how much force it has when it's attached to the two points.
        Stiffness:
            I don't know exactly. I'm thinking something like, how much force you would get
            per unit stretch, if it were unit length. Because if it were half the length then half the
            stretching makes it the same force.

        Here's an interesting thing: If something is length 1, and you stretch it to 2, you get force F.
        If it's length 2, you stretch it to 4 to get the same force.
        Let's say stiffness is the amount you have to multiply it by to get force 1. That's a good def.
        No, it isn't. Bigger stiffness should mean less multiplying... Maybe it should be the amount you
        need to divide by to get the rest length. Does that make sense?

        The point is, I need a quantity that's invariant to length. Spring constant isn't.
        SC/RL is.

        Let's call stiffness = SF = SC*RL. When SC doubles, SF doubles.
            If SC were to stay the same but RL doubled, that means SF doubles.
        Also, SC = SF/RL
        Then, if we have tension=T, and stretch length = SL:
        We know T = (SL-RL)*SC = (SL-RL)*(SF/RL).
        T = (SL*SF/RL) - SF
        T + SF = SL*SF/RL
        (T+SF)/(SL*SF)=(1/RL)
        RL=(SL*SF)/(T+SF)

        One more time:

        And, I have a formula for tension as well... T = (SL - RL) * (SF / RL)

        T = (SL - RL) * SC => T = (SL - RL) * (SF / RL) => T = (SL*SF/RL) - SF =>
        T + SF = (SF*SL/RL) => (T + SF) / (SF * SL) = (1/RL) =>
        RL = (SF*SL)/ (T + SF).
        So, lets say p1 and p2 are 1 apart. And tension is 1, and stiffness is 2. That means you need to
        multiply the length by 2 to get the tension. Meaning that the rest length is 0.1

        If they're 1 apart, tension is 1, stiffness is 3. That means that you need to mul
        """


    def update_point_forces(self):
        """
        One way to do this: Get the magnitude of the force, multiply it by the direction of the force.
        Direction of the force will be: -kX.

        So, you get the vector. And then it's always pulling, so the force always goes in the opposite
        direction.

        This is isn't right when it gets short...

        Somehow, we need to work this out a bit better.

        I should redirect the rest-length in the direction of the real edge. That makes sense.

        """

        # you know the rest-length. You know how long the edge-vector norm is.
        # edge_vector_norm - rest_length is stretched distance.
        # force = Stretched distance * spring_coefficient * edge_vector / edge_vector_norm
        # Force = ((edge_vector_norm - rest_length) * spring_coefficient / edge_vector_norm) * edge_vector

        # I guess the real thing is that it's just too weird for it to compress past zero. Why is it doing that in
        # the first place?

        # First, you get the direction of the force. That's the normalized edge_vector.
        edge_vector = self.edge_vector
        edge_vector_norm = LA.norm(edge_vector)
        # if edge_vector_norm < 0.001:
        #     print('boom')
        #     import ipdb; ipdb.set_trace()
        #     print(edge_vector_norm)
        # edge_vector_norm = (np.sum(edge_vector**2))**0.5

        scaling_constant = ((edge_vector_norm - self.rest_length) * self.spring_coefficient / (edge_vector_norm + 1e-8))
        force = scaling_constant * edge_vector


        # edge_vector = self.edge_vector
        # edge_vector_norm = (np.sum(edge_vector**2))**0.5
        # normalized_edge_vector = edge_vector / edge_vector_norm #Direction...
        # rest_length_vector = normalized_edge_vector*self.rest_length
        # vector_diff = edge_vector - rest_length_vector
        # force = vector_diff * self.spring_coefficient



        # # Then, you get the amount that it stretches. That's the length, minus the rest-length.
        # length_diff = edge_vector_norm - self.rest_length
        # print(length_diff)
        # # To get the magnitude of the force, you do length_diff*spring_coefficient
        # force_magnitude = self.spring_coefficient * length_diff
        # force = normalized_edge_vector * force_magnitude

        # If p1=[0,0,0], p2=[1,0,0], and rl=0, then force will point right. But, we need a negative
        # sign in there. It's going to pull both towards the center. It should add the force to p1
        # and subtract it from p2. That's right.
        self.p1.acc += force
        self.p2.acc -= force

    @property
    def edge_vector(self):
        return self.p2.loc - self.p1.loc

    @property
    def edge_vector_norm(self):
        edge_vector = self.edge_vector
        edge_vector_norm = (np.sum(edge_vector**2))**0.5
        return edge_vector_norm

    # @property
    # def normalized_edge_vector(self):
    #     edge_vector = self.edge_vector
    #     edge_vector_norm = (np.sum(edge_vector**2))**0.5
    #     normalized_edge_vector = edge_vector / edge_vector_norm
    #     return normalized_edge_vector

class EdgeOfMaterial(Edge):
    def __init__(self, p1, p2, tension, stiffness):
        self.p1 = p1
        self.p2 = p2
        starting_stretched_length = self.edge_vector_norm
        rest_length = (stiffness*starting_stretched_length) / (tension + stiffness) #Do the units even work?! I think so actually.
        spring_coefficient = (stiffness/rest_length)
        self.rest_length = rest_length
        self.spring_coefficient = spring_coefficient
        # print("Spring coefficient: {}".format(spring_coefficient))
        # print("Rest Length: {}".format(rest_length))






class Web(object):
    def __init__(self, edges, force_func=None):
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
        pass

    def step(self, timestep=1.0):
        """
        First, you collect the forces. Then, you update pos, vel, acc of each point.
        """
        self.num_steps += timestep

        for e in self.edge_set:
            e.update_point_forces()

        if self.force_func is not None:
            self.force_func(self.num_steps)

        for p in self.point_set:
            p.update_loc_vel(timestep=timestep)

    def _collect_points(self):
        point_set = set([])
        for e in self.edge_set:
            point_set.add(e.p1)
            point_set.add(e.p2)
        return point_set
