
def format_time(total_time, index, total):
    """
    Formats an ETA as a string given the total time so far,
    which index in the loop we're in and the total num of steps
    """
    tt = (float(total_time)/(index+1)) * (total-index)

    m, s = divmod(tt, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return "{0:.2f}h {1:.2f}m {2:.2f}s".format(h, m, s)
    elif m > 0:
        return "{0:.2f}m {1:.2f}s".format(m, s)
    else:
        return "{0:.2f}s".format(s)