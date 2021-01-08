

def show_progress_py3(title, index, maximum):
    """
    :param title:
    :param index:
    :param maximum:
    :return:
    """
    assert index >= 0 and maximum > 0
    # Show progress
    # (1) Get progress
    progress = min(int((index + 1) * 100.0 / maximum), 100)
    # (2) Show progress
    print('\r%s: [%s%s] %d%%' % (title, '>' * int(progress / 5), ' ' * (20 - int(progress / 5)), progress),
          end='\n' if progress == 100 else '')
