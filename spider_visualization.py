import time

from mpl_visualization import  MPLWebDisplay
from spider_web import Spider

import numpy as np

import matplotlib.animation as animation
from collections import defaultdict

def get_all_points_in_line_from_spider(spider, web):
    # This is only useful for now. I'm going to do a center-web type thing,
    # where I give it a center we
    # I just realized, that we don't want to include that last one. Becuase it's pinned,
    # so it'll never move. Luckily, it seems I did that by accident before.
    cp = spider.current_point
    assert web.center_point == cp
    point_vals = defaultdict(list)
    for direction in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']:
        this_point = getattr(cp, direction)
        point_vals[direction].append(this_point)
        while this_point.radial_after != this_point:
            this_point = this_point.radial_after
            point_vals[direction].append(this_point)

    sizes = set([len(point_vals[k]) for k in point_vals])
    assert len(sizes) == 1 # Ensures they're all the same length.

    return point_vals

class MPLWebDisplayWithSpider(MPLWebDisplay):
    def __init__(self, web, spider, **kwargs):
        print('huzzah!')
        print(kwargs)
        self.spider = spider
        super().__init__(web, **kwargs)

    # def set_up_line_drawings(self):
    #     super().set_up_line_drawings()
    #     cp = self.spider.current_point
    #
    #     xs, ys, zs = [[v] for v in self.spider.current_point.loc]
    #     self.spider_point_draw = self.ax.scatter(xs, ys, zs, color='red', linewidth=2)
    #
    #     leg_locs = [getattr(cp, val).loc for val in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']]
    #
    #     leg_xs, leg_ys, leg_zs = zip(*leg_locs)
    #     self.spider_legs_draw = self.ax.scatter(leg_xs, leg_ys, leg_zs, color="black", linewidth=2)

    def draw_web(self, frame, step=True):
        # self.ax.clear()
        if hasattr(self, 'center_draw'):
            self.center_draw.remove()
        if hasattr(self, 'leg_draw'):
            self.leg_draw.remove()

        start_time = time.time()
        # print("Draw lines without blit")
        cp = self.spider.current_point
        self.ax.lines = []
        # for i, l in enumerate(self.ax.lines):
        #     asdf = self.ax.lines.pop(i)
        #     l.remove()
        for e in self.edges:
            zipped = list(zip(e.p1.loc, e.p2.loc))
            self.ax.plot(*zipped, color='cyan')

        xs, ys, zs = [np.asarray([v]) for v in self.spider.current_point.loc]
        self.center_draw = self.ax.scatter(xs, ys, zs, color='green', linewidth=10, depthshade=False)

        leg_locs = [getattr(cp, val).loc for val in ['radial_before', 'radial_after', 'azimuthal_before', 'azimuthal_after']]

        leg_xs, leg_ys, leg_zs = zip(*leg_locs)
        self.leg_draw = self.ax.scatter(leg_xs, leg_ys, leg_zs, color="red", linewidth=10, depthshade=False)




        # if step:
        #     for _ in range(self.steps_per_frame):
        #         # print("Stepping time number {}".format(_))
        #         self.web.step(self.step_size)
        # print("Took {} seconds to draw web without blit".format(time.time()-start_time))


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

    def run(self):
        # This will actually happen as part of a data-generating phase...
        while self.web.num_steps < self.start_drawing_at:
            self.web.step(self.step_size)

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15, metadata=dict(artist='Sam Lobel'), bitrate=1000)
        with writer.saving(self.fig, "writer_test.mp4", 100):
            for frame in range(self.frames_to_write):
                print('frame {}'.format(frame))
                for _ in range(self.steps_per_frame):
                    self.web.step(self.step_size)
                self.draw_web(frame)
                writer.grab_frame()


if __name__ == '__main__':
    import web_zoo

    radius=10
    web = web_zoo.radial_web(radius=radius,
                             num_radial=16,
                             num_azimuthal=8,
                             stiffness_radial=0,
                             tension_radial=100,
                             stiffness_azimuthal=0,
                             # stiffness_azimuthal=90,
                             tension_azimuthal=20,
                             damping_coefficient=0.1,
                             edge_type='stiffness_tension',
                             num_segments_per_radial=2,
                             num_segments_per_azimuthal=5,
                            )

    spider = Spider(web, web.center_point)
    # wd = MPLWebDisplayWithSpider(web, spider, steps_per_frame=25, frames_to_write=20, step_size=0.002, blit=False, start_drawing_at=0.0)
    wd = MPLWebDisplayWithSpider(web, spider, steps_per_frame=25, frames_to_write=20, step_size=0.002, blit=False, start_drawing_at=2.0)

    gather_points = [spider.current_point.radial_before,
                     spider.current_point.radial_after,
                     spider.current_point.azimuthal_before,
                     spider.current_point.azimuthal_after]

    web.set_gather_points(gather_points)

    point_dict = get_all_points_in_line_from_spider(spider, web)

    print(point_dict)

    off_center_point = list(filter(lambda p : p.loc[0] == 5.0, web.point_set))[0]

    assert off_center_point

    force_func = web_zoo.random_oscillate_point_one_dimension(off_center_point, [0,0,1.0], max_force=250.0, delay=3.0)
    web.force_func = force_func

    # exit()

    while wd.web.num_steps < wd.start_drawing_at:
        wd.web.step(wd.step_size)

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Sam Lobel'), bitrate=1000)
    with writer.saving(wd.fig, "walking_spider.mp4", 100):
        for frame in range(500):
            print("Drawing frame {}".format(frame))
            for _ in range(wd.steps_per_frame):
                wd.web.step(wd.step_size)
            # web.record_gather_points()
            # print(web.gather_points)
            if frame % 10 == 0 and frame < 50:
                wd.spider.current_point = wd.spider.current_point.radial_after
            elif frame % 10 == 0 and frame >= 50:
                wd.spider.current_point = wd.spider.current_point.azimuthal_after

            wd.draw_web(frame)
            writer.grab_frame()

        # for frame in range(wd.frames_to_write):
        #     if frame % 3 == 0:
        #         print('moving spider')
        #         wd.spider.current_point = wd.spider.current_point.radial_after
        #     print('frame {}'.format(frame))
        #     for _ in range(wd.steps_per_frame):
        #         wd.web.step(wd.step_size)
        #     wd.draw_web(frame)
        #     writer.grab_frame()


    # wd.run()



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
