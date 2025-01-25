import cProfile
from functools import wraps


def profile(filename=None):
    """
    A decorator to profile a function using cProfile.
    Saves the profile results to a file if a filename is provided.
    """

    def prof_decorator(f):
        @wraps(f)
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename:
                pr.dump_stats(filename)
            else:
                pr.print_stats()

            return result

        return wrap_f

    return prof_decorator
