import gc
import time

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class MPLWebDisplay(object):
    def __init__(self, web, steps_per_frame=100, frames_to_write=100, step_size=0.001, blit=True):
        self.web = web
        self.edges = list(web.edge_set)
        self.step_size = step_size
        self.steps_per_frame = steps_per_frame
        self.frames_to_write = frames_to_write
        self.blit = blit
        self.set_up_3d_figure()

    def set_up_3d_figure(self):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_xlim3d([-4.0, 4.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-4.0, 4.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-4.0, 4.0])
        ax.set_zlabel('Z')

        ax.set_title('3D Test')
        self.fig = fig
        self.ax = ax
        if self.blit:
            self.set_up_line_drawings()

    def set_up_line_drawings(self):
        self.all_lines = []
        for e in self.edges:
            zipped = list(zip(e.p1.loc, e.p2.loc))
            line = self.ax.plot(*zipped, color='cyan')
            self.all_lines.extend(line)

    def update_drawing_blit(self):
        for e, l in zip(self.edges, self.all_lines):
            zipped = list(zip(e.p1.loc, e.p2.loc))
            l.set_data(zipped[0:2])
            l.set_3d_properties(zipped[2])
        return self.all_lines

    def draw_lines_blit(self, frame, step=True):
        start_time = time.time()
        print("Draw lines blit. Frame {}".format(frame))
        for _ in range(self.steps_per_frame):
            self.web.step(self.step_size)

        print("Took {} seconds to draw web WITH blit".format(time.time()-start_time))
        return self.update_drawing_blit()


    def draw_web(self, frame, step=True):
        start_time = time.time()
        print("Draw lines without blit")
        self.ax.lines = []
        # for i, l in enumerate(self.ax.lines):
        #     asdf = self.ax.lines.pop(i)
        #     l.remove()
        for e in self.edges:
            zipped = list(zip(e.p1.loc, e.p2.loc))
            self.ax.plot(*zipped, color='cyan')
        if step:
            for _ in range(self.steps_per_frame):
                print("Stepping time number {}".format(_))
                self.web.step(self.step_size)
        print("Took {} seconds to draw web without blit".format(time.time()-start_time))

    def run(self):
        draw_func = self.draw_lines_blit if self.blit else self.draw_web
        web_ani = animation.FuncAnimation(self.fig,
                                       # self.draw_lines_blit,
                                       draw_func,
                                       frames=self.frames_to_write,
                                       blit=self.blit, #NOTE: May break everything...
                                       interval=1)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Sam Lobel'), bitrate=1800)
        web_ani.save('./videos/v26_no_external_force_with_damping_no_azimuthal_force_no_azimuthal.mp4', writer=writer)
        # plt.show()
        # plt.show()

        # for e in self.edges:



if __name__ == '__main__':
    import web_zoo
    from web_zoo import square_web_with_z_offset_center
    # web = web_zoo.radial_web(5, 10, 5, 5.0, 1.0, damping_coefficient=0.0)
    radius=10
    web = web_zoo.radial_web_tension_stiffness(radius=radius,
                                               num_radial=20,
                                               # num_azimuthal=10, #ORIG
                                               num_azimuthal=10,
                                               tension_radial=50,
                                               # tension_radial=10,
                                               tension_azimuthal=1,
                                               # tension_azimuthal=50, #ORIG
                                               stiffness_radial=100,
                                               # stiffness_radial=20,
                                               stiffness_azimuthal=1,
                                               # stiffness_azimuthal=100, #ORIG
                                               damping_coefficient=1.0,
                                               # num_segments_per_radial=5, #ORIG
                                               num_segments_per_radial=2, #ORIG
                                               # num_segments_per_azimuthal=5 #ORIG
                                               num_segments_per_azimuthal=1
                                           )
    from web_zoo import sine_oscillate_point
    # force_func = sine_oscillate_point(web.center_point, [0,0,1.0], max_force=100.0, period=10.0)
    # web.force_func = force_func
    # force_func = sine_oscillate_point(web.center_point, [0,1.0,0.0], max_force=100.0, period=10.0)
    # web.force_func = force_func

    # from web_zoo import deform_web, move_point_to_cosine
    # def func(point):
    #     return move_point_to_cosine(point, radius)
    # deform_web(web, func)

    wd = MPLWebDisplay(web, steps_per_frame=25, frames_to_write=100, step_size=0.004, blit=True)
    # wd = MPLWebDisplay(web, steps_per_frame=100, frames_to_write=100)
    # wd.draw_web()
    wd.run()
