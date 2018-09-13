"""
Alright, here's the outline of what I'll do here.

First, you make the web.

To the web, you need to add to each node a value, which is the distance from the center that
the node is.

Then, you gather all the intersection points in one list. You gather all the points along one
one direction in another list.

Finally, you gather data from a "spider" at EACH point (except the one being oscillated).

From the last one, it's difficult to do this, because every time you want to reset the web, you
get all new points. But, for this one, since I'm just randomly vibrating one of the nodes, it's
fine for me to just restart it, and use a count to get the next radial point.

So, you'd say: 0 is center, 1 is center.radial_after, 2 is center.radial_after.radial_after

Also note that we don't want the outside one to vibrate at all, because it's pinned, so
nothing will happen either way.

"""

import web_zoo
from spider_web import get_distance_between_two_points, Spider
import json
import os
import pickle
import numpy as np

def assign_each_point_in_web_a_radial_fraction(web, num_azimuthal):
    center_point = web.center_point
    center_point.distance_from_center = 0.0
    radial_starting_point = center_point
    for i in range(1, num_azimuthal):
        print('boom, {}'.format(i))
        assert radial_starting_point != radial_starting_point.radial_after
        radial_starting_point = radial_starting_point.radial_after
        around_in_circle = radial_starting_point
        while True:
            if hasattr(around_in_circle, 'distance_from_center'):
                break
            around_in_circle.distance_from_center = (i / (num_azimuthal - 1))
            around_in_circle = around_in_circle.azimuthal_after

    assert radial_starting_point.radial_after == radial_starting_point

def get_points_a_spider_might_stand(web, point_that_is_oscillating):
    # All intersections
    to_return = [p for p in web.point_set if getattr(p, 'intersection', False)]
    # Except pinned ones
    to_return = [p for p in to_return if not getattr(p, 'pinned', False)]
    # Or the one that's oscillating.
    to_return = [p for p in to_return if p != point_that_is_oscillating]

    return to_return

def get_goal_vector_for_spider_and_oscillating_point(spider, oscillating_point):
    # What's the order of point that we're using always? r_b, r_a, a_b, a_a
    gather_points = spider.get_gather_points()
    d_orig = get_distance_between_two_points(spider.current_point, oscillating_point)
    gather_point_distances = [get_distance_between_two_points(p, oscillating_point) for p in gather_points]
    goal_array = []
    for g_d in gather_point_distances:
        if g_d > d_orig:
            goal_array.append(-1.0) #You dont want to move away
        elif g_d < d_orig:
            goal_array.append(1.0) #You do want to move towards
        else:
            goal_array.append(0.0) #If its the same. Not sure if this will happen...
    return goal_array

def record_goals_and_samples_for_point(web,
                                       point_that_is_oscillating,
                                       samples_per_recording=None,
                                       num_recordings=None,
                                       start_recording_at=None,
                                       step_size=None):
    """
    What's it going to do? Well, first, it's going to put a spider on EVERY point
    a spider might stand on. Then, it starts the oscillation. Eventually, start recording.
    Keep doing that until you have enough samples.

    Also, for each point that you're recording on, make a GOAL vector. Make a concatenation of
    these, too.

    Finally, return two arrays, one of data, and one of goals, that line up.
    """
    assert None not in [samples_per_recording, num_recordings, start_recording_at, step_size]

    spider_points = get_points_a_spider_might_stand(web, point_that_is_oscillating)

    spiders = [Spider(web, p) for p in spider_points]

    ALL_RECORDINGS = []
    ALL_TARGETS = []

    print("stepping along")
    while web.num_steps < start_recording_at:
        # print(web.num_steps)
        web.step(step_size)
    print("stepped along until size was {}".format(web.num_steps))

    for recording_num in range(num_recordings):
        print("Recording record number {}".format(recording_num))
        # First, clear the spiders memories.
        for spider in spiders:
            spider.reset_gather_points()
        # Then, you step for the number of steps that is needed, recording as you go.
        for sample in range(samples_per_recording):
            web.step(step_size)
            for spider in spiders:
                spider.record_gather_points()
        # Now, it should be done recording ONE sample.
        recordings_per_spider = []
        targets_per_spider = []
        for spider in spiders:
            recording = spider.vectorize_gather_point_dict(recording_size=samples_per_recording,
                                                           squash_to_energy=True,
                                                           include_radial_distance=True)
            target = get_goal_vector_for_spider_and_oscillating_point(spider,
                                                                      point_that_is_oscillating)
            recordings_per_spider.append(recording)
            targets_per_spider.append(target)

            spider.reset_gather_points()
        ALL_RECORDINGS.extend(recordings_per_spider)
        ALL_TARGETS.extend(targets_per_spider)
    return ALL_RECORDINGS, ALL_TARGETS




def test_it_out(
    samples_per_recording=100,
    num_recordings=10,
    start_recording_at=3.0,
    step_size=0.0002):
    # num_azimuthal = 20
    # web = web_zoo.radial_web(radius=10,
    #                          num_radial=16,
    #                          num_azimuthal=num_azimuthal,
    #                          stiffness_radial=30,
    #                          tension_radial=30,
    #                          stiffness_azimuthal=30,
    #                          tension_azimuthal=30,
    #                          damping_coefficient=0.1,
    #                          edge_type='stiffness_tension',
    #                          num_segments_per_radial=5,
    #                          num_segments_per_azimuthal=5,
    #                         )
    # assign_each_point_in_web_a_radial_fraction(web, 20)
    num_azimuthal = 8
    ALL_RECORDINGS, ALL_TARGETS = [], []
    for i in range(num_azimuthal-1):
        print("Vibrating azimuthal number {}".format(i))
        web = web_zoo.radial_web(radius=10,
                                num_radial=16,
                                num_azimuthal=num_azimuthal,
                                stiffness_radial=0,
                                tension_radial=100,
                                stiffness_azimuthal=0,
                                tension_azimuthal=20,
                                damping_coefficient=0.1,
                                edge_type='stiffness_tension',
                                num_segments_per_radial=5,
                                num_segments_per_azimuthal=5,
                                )

        print("Finding a force-point...")
        assign_each_point_in_web_a_radial_fraction(web, num_azimuthal)
        point_to_vibrate = web.center_point
        for _ in range(i):
            point_to_vibrate = point_to_vibrate.radial_after

        print("Assigning a force_func")

        force_func = web_zoo.random_oscillate_point_one_dimension(point_to_vibrate, [0,0,1.0], max_force=250.0, delay=2.0)
        web.force_func = force_func

        recordings, targets = record_goals_and_samples_for_point(web,
                                    point_to_vibrate,
                                    samples_per_recording=samples_per_recording,
                                    num_recordings=num_recordings,
                                    start_recording_at=start_recording_at,
                                    step_size=step_size)
        ALL_RECORDINGS.extend(recordings)
        ALL_TARGETS.extend(targets)
        print("Number of recordings: {}".format(len(ALL_RECORDINGS)))
        # print(asdf.distance_from_center)
        # # print(asdf.azimuthal_after.azimuthal_after.azimuthal_after.distance_from_center)
        # asdf = asdf.radial_after

    # 
    stacked_recordings = np.stack(ALL_RECORDINGS)
    stacked_targets = np.stack(ALL_TARGETS).astype(np.float32)
    return stacked_recordings, stacked_targets
    # return ALL_RECORDINGS, ALL_TARGETS
    # recordings = []
    # targets = []


    # import ipdb; ipdb.set_trace()
    # record_goals_and_samples_for_point(web, )
    print('goodbye')

def write_data(inputs, targets, target_directory, config={}, overwrite=False):

    path_to_config = os.path.join(target_directory, "config.json")
    path_to_data = os.path.join(target_directory, "data.pkl")

    if not overwrite and os.path.exists(path_to_data):
        print("Going to exit early, because it already exists.")
        return


    dictionary = {
        'inputs': inputs,
        'targets': targets
    }
    
    print('writing config')
    with open(path_to_config, 'w') as f:
        f.write(json.dumps(config))

    print('config written. writing data')
    with open(path_to_data, 'wb') as f:
        pickle.dump(dictionary, f)
    print('written')
    


if __name__ == '__main__':
    # pass
    # all_recordings, all_targets = test_it_out()
    # This could be annoying, because the writing maybe won't happen after a long training run. Oh well.
    config = {
        'samples_per_recording': 100,
        'num_recordings': 100,
        'start_recording_at': 5.0,
        'step_size': 0.0002,
    }
    
    all_recordings, all_targets = test_it_out(**config)
    write_data(all_recordings, all_targets, target_directory="./data/locating/", config=config, overwrite=False)

    print('goodbye for real...')
