import time

from mpl_visualization import  MPLWebDisplay
from spider_web import Spider

class MPLWebDisplayWithSpider(MPLWebDisplay):
    def __init__(self, web, spider, **kwargs):
        print('huzzah!')
        print(kwargs)
        self.spider = spider
        super().__init__(web, **kwargs)

    def set_up_line_drawings(self):
        super().set_up_line_drawings()
        cp = self.spider.current_point

        xs, ys, zs = [[v] for v in self.spider.current_point.loc]
        self.spider_point_draw = self.ax.scatter(xs, ys, zs, color='green')

        leg_locs = [getattr(cp, val).loc for val in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']]

        leg_xs, leg_ys, leg_zs = zip(*leg_locs)
        self.spider_legs_draw = self.ax.scatter(leg_xs, leg_ys, leg_zs, color="black")

    def draw_web(self, frame, step=True):
        if hasattr(self, 'center_draw'):
            self.center_draw.remove()
        if hasattr(self, 'leg_draw'):
            self.leg_draw.remove()

        start_time = time.time()
        print("Draw lines without blit")
        cp = self.spider.current_point
        self.ax.lines = []
        # for i, l in enumerate(self.ax.lines):
        #     asdf = self.ax.lines.pop(i)
        #     l.remove()
        for e in self.edges:
            zipped = list(zip(e.p1.loc, e.p2.loc))
            self.ax.plot(*zipped, color='cyan')

        xs, ys, zs = [[v] for v in self.spider.current_point.loc]

        self.center_draw = self.ax.scatter(xs, ys, zs, color='green')

        leg_locs = [getattr(cp, val).loc for val in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']]

        leg_xs, leg_ys, leg_zs = zip(*leg_locs)
        self.leg_draw = self.ax.scatter(leg_xs, leg_ys, leg_zs, color="red")


        if step:
            for _ in range(self.steps_per_frame):
                # print("Stepping time number {}".format(_))
                self.web.step(self.step_size)
        print("Took {} seconds to draw web without blit".format(time.time()-start_time))


    def update_drawing_blit(self):
        for e, l in zip(self.edges, self.all_lines):
            zipped = list(zip(e.p1.loc, e.p2.loc))
            l.set_data(zipped[0:2])
            l.set_3d_properties(zipped[2])

        # import ipdb; ipdb.set_trace()
        # self.spider_point_draw.set_array([[0],[0],[0]])
        # point = self.spider_point_draw[0]
        # point.set_array(self.spider.current_point.loc[0:2])
        # point.set_3d_properties(self.spider.current_point.loc[3])

        # leg_locs = [getattr(self.spider.current_point, val).loc for val in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']]
        # for draw_point, real_point in zip(self.spider_legs_draw, leg_locs):
        #     draw_point.set_data(real_point.loc[0:2])
        #     draw_point.set_3d_properties(real_point.loc[3])

        return self.all_lines + [self.spider_point_draw]

    def draw_lines_blit(self, frame, step=True):
        start_time = time.time()
        print("Draw lines blit. Frame {}".format(frame))
        for _ in range(self.steps_per_frame):
            self.web.step(self.step_size)

        print("Took {} seconds to draw web WITH blit".format(time.time()-start_time))
        return self.update_drawing_blit()


if __name__ == '__main__':
    import web_zoo

    radius=10
    web = web_zoo.radial_web(radius=radius,
                             num_radial=8,
                             num_azimuthal=16,
                             stiffness_radial=0,
                             tension_radial=300,
                             stiffness_azimuthal=0,
                             # stiffness_azimuthal=90,
                             tension_azimuthal=30,
                             damping_coefficient=0.1,
                             edge_type='stiffness_tension',
                             num_segments_per_radial=2,
                             num_segments_per_azimuthal=5,
                            )

    spider = Spider(web, web.center_point)
    # wd = MPLWebDisplayWithSpider(web, spider, steps_per_frame=25, frames_to_write=20, step_size=0.002, blit=False, start_drawing_at=0.0)
    wd = MPLWebDisplayWithSpider(web, spider, steps_per_frame=25, frames_to_write=20, step_size=0.01, blit=False, start_drawing_at=2.0)

    wd.run()



    # from web_zoo import sine_oscillate_point, force_point_to_sine
    # movement_func = force_point_to_sine(web.center_point,
    #                                     direction_vector=[0.0, 0.0, 1.0],
    #                                     amplitude=3.0,
    #                                     period=8.0, delay=2.0)
    # web.movement_func = movement_func
    # off_center_point = list(filter(lambda p : p.loc[0] == 5.0, web.point_set))[0]
    # force_func = web_zoo.random_oscillate_point_one_dimension(off_center_point, [0,0,1.0], max_force=750.0, delay=3.0)
    # web.force_func = force_func
    # force_func = web_zoo.impulse_to_point(off_center_point, [0,0,1.0], force=2000, force_time=0.5, delay=3.0)
    # web.force_func = force_func
    # force_func = sine_oscillate_point(web.center_point, [0,0,1.0], max_force=750.0, period=8.0, delay=2.0)
    # web.force_func = force_func

    # from web_zoo import deform_web, move_point_to_cosine
    # def func(point):
    #     return move_point_to_cosine(point, radius)
    # deform_web(web, func)

    # wd = MPLWebDisplay(web, steps_per_frame=50, frames_to_write=100, step_size=0.002, blit=True)
    # wd = MPLWebDisplay(web, steps_per_frame=100, frames_to_write=100)
    # wd.draw_web()
