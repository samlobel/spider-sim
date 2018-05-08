import gc

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class MPLWebDisplay(object):
    def __init__(self, web, steps_per_frame=100, frames_to_write=100):
        self.web = web
        self.edges = list(web.edge_set)
        self.steps_per_frame = steps_per_frame
        self.frames_to_write = frames_to_write
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



    def draw_web(self, frame, step=True):
        print("drawing frame number {}. Step={}".format(frame, step))
        self.ax.lines = []
        # for i, l in enumerate(self.ax.lines):
        #     asdf = self.ax.lines.pop(i)
        #     l.remove()
        for e in self.edges:
            zipped = list(zip(e.p1.loc, e.p2.loc))
            self.ax.plot(*zipped, color='cyan')
        if step:
            for _ in range(self.steps_per_frame):
                self.web.step(0.001)

    def run(self):
        web_ani = animation.FuncAnimation(self.fig,
                                       self.draw_web,
                                       frames=self.frames_to_write,
                                       interval=1)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        web_ani.save('./videos/v10_symp.mp4', writer=writer)
        # plt.show()
        # plt.show()

        # for e in self.edges:



if __name__ == '__main__':
    import web_zoo
    from web_zoo import square_web_with_z_offset_center
    # web = web_zoo.radial_web(5, 10, 5, 5.0, 1.0, damping_coefficient=0.0)
    web = web_zoo.radial_web_tension_stiffness(radius=10,
                                               num_radial=12,
                                               num_azimuthal=20,
                                               tension_radial=50,
                                               # tension_radial=1,
                                               tension_azimuthal=5,
                                               stiffness_radial=100,
                                               # stiffness_radial=2,
                                               stiffness_azimuthal=20,
                                               damping_coefficient=0.0)
    from web_zoo import sine_oscillate_point
    force_func = sine_oscillate_point(web.center_point, [0,0,1.0], max_force=50.0, period=4.0)
    web.force_func = force_func

    # from web_zoo import deform_web, move_point_to_cosine
    # def func(point):
    #     return move_point_to_cosine(point, 4)
    # deform_web(web, func)

    wd = MPLWebDisplay(web, steps_per_frame=500, frames_to_write=50)
    # wd = MPLWebDisplay(web, steps_per_frame=100, frames_to_write=100)
    # wd.draw_web()
    wd.run()
