import timeit
import datetime
import sys


def estimate_time(length: int, i: int, start_time: float, output: bool = True):
    """
    Returns the estimate of when a looping operation will be finished.

    :param length: total number of iterations for the loop
    :param i: iterator for the loop
    :param start_time: time created from timeit.default_timer(), see examples
    :param output: False to suppress printing estimated time. Use this if you want to just store the time
        for some other use or custom output.
    :return:

    :Examples:

    This function goes at the end of the loop to be timed. Outside of this function at the beginning of the
    loop, you must start a timer object as follows::

        start_time = timeit.default_timer()

    So the entire loop will look like this::

        my_start_time = timeit.default_timer()
        for i, item in enumerate(my_list):

            #Do loop stuff here

             estimate_time(len(my_list),i,my_start_time)


    """
    avg_time = (timeit.default_timer() - start_time) / (i + 1)
    loops_left = length - (i + 1)
    est_time_remaining = avg_time * loops_left
    est_finish_time = datetime.datetime.now() + datetime.timedelta(0, est_time_remaining)

    if output == True:
        print("Estimated finish time: {}. Completed {}/{}, ({:.0%})".format(est_finish_time, i, length, i / length),
              end="\r")
        sys.stdout.flush()

    return est_finish_time
