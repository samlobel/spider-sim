# import vpython
# import gc
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
#
# class WebDisplay(object):
#     """
#     NOTE: So that the set stays in the same order, we're assuming that the Edge objects are ALWAYS the same
#     in the web...
#     """
#     def __init__(self, web, display_rate):
#         self.web = web
#         self.edges = list(self.web.edge_set)
#         self.rate = display_rate
#         self.draw_elements = []
#         self.colors = [vpython.color.red, vpython.color.yellow, vpython.color.green, vpython.color.blue]
#
#     def clear_web(self):
#         while len(self.draw_elements):
#             first = self.draw_elements[0]
#             first.visible = False
#             del self.draw_elements[0]
#         gc.collect()
#
#     def draw_web(self):
#         self.clear_web()
#         for i, edge in enumerate(self.edges):
#             p1, p2 = edge.p1, edge.p2
#             c = vpython.curve(vpython.vector(*p1.loc.tolist()), vpython.vector(*p2.loc.tolist()), color=self.colors[i%4])
#             self.draw_elements.append(c)
#
#     def run(self):
#         vpython.scene.center = vpython.vector(0,0,0)
#         self.draw_web()
#         vpython.scene.autoscale = False
#         i = 0
#         while True:
#             print("Timestep: {}".format(i))
#             i += 1
#             vpython.rate(self.rate)
#             self.draw_web()
#             for _ in range(10):
#                 self.web.step(0.001)
#
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
#
# class MPLWebDisplay(object):
#     def __init__(self, web, display_rate):
#         self.web = web
#         self.edges = list(web.edge_set)
#         self.rate = display_rate
#         self.set_up_3d_figure()
#
#     def set_up_3d_figure(self):
#         fig = plt.figure()
#         ax = p3.Axes3D(fig)
#         ax.set_xlim3d([-1.0, 1.0])
#         ax.set_xlabel('X')
#
#         ax.set_ylim3d([-1.0, 1.0])
#         ax.set_ylabel('Y')
#
#         ax.set_zlim3d([-1.0, 1.0])
#         ax.set_zlabel('Z')
#
#         ax.set_title('3D Test')
#         self.fig = fig
#         self.ax = ax
#
#
#
#     def draw_web(self, step=True):
#         for i, l in enumerate(self.ax.lines):
#             ax.lines.pop(i)
#             l.remove()
#         for e in self.edge:
#             zipped = list(zip(e.p1.loc, e.p2.loc))
#             ax.plot(*zipped)
#             plt.show()
#         # for e in self.edges:
#
#
#
# if __name__ == '__main__':
#     import web_zoo
#     from web_zoo import square_web_with_z_offset_center
#     # web = web_zoo.radial_web(5, 10, 5, 5.0, 1.0, damping_coefficient=0.0)
#     web = web_zoo.radial_web_tension_stiffness(radius=5,
#                                                num_radial=6,
#                                                num_azimuthal=5,
#                                                tension_radial=10,
#                                                # tension_radial=1,
#                                                tension_azimuthal=5,
#                                                stiffness_radial=40,
#                                                # stiffness_radial=2,
#                                                stiffness_azimuthal=20,
#                                                damping_coefficient=0.0)
#
#
#     from web_zoo import deform_web, move_point_to_cosine
#     def func(point):
#         return move_point_to_cosine(point, 1)
#     deform_web(web, func)
#
#     wd = MPLWebDisplay(web, 100)
#     wd.draw_web()
#     exit()
#     # web = square_web_with_z_offset_center()
#     # web = web_zoo.many_segment_line(num_points=10,length=1, per_spring_rest_length=0.3, wiggle_size=0.5)
#     # web = web_zoo.single_segment()
#     # web = web_zoo.single_segment_tension_stiffness()
#
#     # i = 0
#     # while True:
#     #     print('timestep: {}'.format(i))
#     #     i+=1
#     #     web.step(0.001)
#     wd = WebDisplay(web, 100)
#     wd.run()
