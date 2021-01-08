

def show_progress_py2(title, index, maximum):
    """
    :param title:
    :param index:
    :param maximum:
    :return:
    """
    assert index >= 0 and maximum > 0
    # Show progress
    # (1) Get progress
    progress = int((index + 1) * 100.0 / maximum)
    # (2) Show progress
    if progress == 100:
        print '\r%s: [%s%s] %d%%' % (title, '>' * (progress / 5), ' ' * (20 - progress / 5), progress)
    else:
        print '\r%s: [%s%s] %d%%' % (title, '>' * (progress / 5), ' ' * (20 - progress / 5), progress),
