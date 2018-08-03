import numpy as np
import time
import web_zoo
from spider_web import Spider
from spider_visualization import get_all_points_in_line_from_spider

from collections import OrderedDict

def morph_gather_points_dict_to_something_useable(gather_points_dict):
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
    values = gather_points_dict.values()
    values_np = np.asarray(list(values))
    without_loc = values_np[:,1:,:,:]
    time_series_last = without_loc.transpose(0,1,3,2) # Switch the last two dimensions.
    flattened = np.reshape(time_series_last, (24, 500))
    return flattened

if __name__ == '__main__':

    direction_map = OrderedDict([
        ('radial_before', 0),
        ('radial_after', 1),
        ('azimuthal_before', 2),
        ('azimuthal_after', 3),
    ])


    radius = 10
    # num_radial = 16
    # num_azimuthal = 16
    num_radial=8
    num_azimuthal=8
    stiffness_radial = 0
    tension_radial = 300
    stiffness_azimuthal = 0
    tension_azimuthal = 20
    damping_coefficient=0.1
    edge_type='stiffness_tension'
    num_segments_per_radial=2
    num_segments_per_azimuthal=5

    step_size = 0.002
    start_recording_at = 0.1
    recording_size = 500 #How many samples to record.

    """
    So, I do want to reset the web every time I change points. But, what I can easily do is,
    just use the index. Recreate each time, only fetch the point by index...

    So, if there are 16 azimuthal, then there should be 15 azimuthals that aren't the center.
    So, just do it by index.
    """

    # FINAL_GATHER_LOC = {}

    SAMPLES = [] # A list of 24x500 samples
    TARGETS = [] # A list of scalars, which correspond to the target-direction.
    NUM_RECORDINGS_PER_POINT=2 #change to 100 at some point.

    for direction in direction_map.keys():
        for index in range(num_azimuthal - 1):
            # Re-create the web...
            web = web_zoo.radial_web(radius=radius,
                                     num_radial=num_radial,
                                     num_azimuthal=num_azimuthal,
                                     stiffness_radial=stiffness_radial,
                                     tension_radial=tension_radial,
                                     stiffness_azimuthal=stiffness_azimuthal,
                                     tension_azimuthal=tension_azimuthal,
                                     damping_coefficient=damping_coefficient,
                                     edge_type=edge_type,
                                     num_segments_per_radial=num_segments_per_radial,
                                     num_segments_per_azimuthal=num_segments_per_azimuthal,
                                    )


            spider = Spider(web, web.center_point)

            point_dict = get_all_points_in_line_from_spider(spider, web)

            off_center_point = point_dict[direction][index]
            assert off_center_point

            gather_points = [spider.current_point.radial_before,
                             spider.current_point.radial_after,
                             spider.current_point.azimuthal_before,
                             spider.current_point.azimuthal_after]

            web.set_gather_points(gather_points)

            force_func = web_zoo.random_oscillate_point_one_dimension(off_center_point, [0,0,1.0], max_force=250.0, delay=3.0)
            web.force_func = force_func

            print("stepping along")
            while web.num_steps < start_recording_at:
                # print(web.num_steps)
                web.step(step_size)
            print("stepped along until size was {}".format(web.num_steps))

            for record_number in range(NUM_RECORDINGS_PER_POINT): #Number of recordings per oscillating point.
                print("Record number {}".format(record_number))
                web.reset_gather_points()
                for number_of_samples in range(recording_size):
                    web.step(step_size)
                    web.record_gather_points()
                something_useable = morph_gather_points_dict_to_something_useable(web.gather_points)
                SAMPLES.append(something_useable)
                TARGETS.append(direction_map[direction])
                print("Num things collected: {}".format(len(SAMPLES)))
            print("Done with index {} of direction {}.".format(index, direction))
        print("Altogether done with direction {}".format(direction))
    print("Done with everything. Waow.")
    samples_as_numpy = np.asarray(SAMPLES)
    targets_as_numpy = np.asarray(TARGETS)
    np.save('data/train_samples', samples_as_numpy)
    np.save('data/train_targets', samples_as_numpy)
