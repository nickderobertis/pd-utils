import timeit
import datetime
import sys


def estimate_time(length, i, start_time, output=True):
    '''
    Returns the estimate of when a looping operation will be finished.

    HOW TO USE:
    This function goes at the end of the loop to be timed. Outside of this function at the beginning of the
    loop, you must start a timer object as follows:

    start_time = timeit.default_timer()

    So the entire loop will look like this:

    my_start_time = timeit.default_timer()
    for i, item in enumerate(my_list):

        #Do loop stuff here

         estimate_time(len(my_list),i,my_start_time)

    REQUIRED OPTIONS:
    length:     total number of iterations for the loop
    i:          iterator for the loop
    start_time: timer object, to be started outside of this function (SEE ABOVE)

    OPTIONAL OPTIONS:
    output: specify other than True to suppress printing estimated time. Use this if you want to just store the time
            for some other use or custom output. The syntax then is as follows:

    my_start_time = timeit.default_timer()
    for i, item in enumerate(my_list):

        #Do loop stuff here

        my_timer = estimate_time(len(my_list),i,my_start_time, output=False)
        print("I like my output sentence better! Here's the estimate: {}".format(my_timer))

    '''
    avg_time = (timeit.default_timer() - start_time) / (i + 1)
    loops_left = length - (i + 1)
    est_time_remaining = avg_time * loops_left
    est_finish_time = datetime.datetime.now() + datetime.timedelta(0, est_time_remaining)

    if output == True:
        print("Estimated finish time: {}. Completed {}/{}, ({:.0%})".format(est_finish_time, i, length, i / length),
              end="\r")
        sys.stdout.flush()

    return est_finish_time
