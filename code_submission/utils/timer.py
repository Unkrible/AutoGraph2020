import time
# import functools
# import multiprocessing


TIME_BUDGET = None

# multiprocessing.get_context("forkserver")
# manager = multiprocessing.Manager()


def set_time_budget(time_budget):
    global TIME_BUDGET
    TIME_BUDGET = TimeBudget(time_budget)


def get_time_budget():
    global TIME_BUDGET
    return TIME_BUDGET


def _wrapped_method(m, return_dict, args, kwargs):
    res = m(*args, **kwargs)
    return_dict["res"] = res
    return return_dict


# def time_limit(milliseconds=1000):
#     def wrapper(method):
#         @functools.wraps(method)
#         def timed(*args, **kwargs):
#             global TIME_BUDGET
#             time_budget = TIME_BUDGET
#             return_dict = manager.dict()
#
#             p = multiprocessing.Process(
#                 target=_wrapped_method,
#                 args=(method, return_dict, args, kwargs),
#             )
#             p.start()
#             # use 80% time for fitting
#             running_time = min(time_budget.remain * 0.8, milliseconds)
#             p.join(running_time)
#             if p.is_alive():
#                 p.kill()
#                 print(f"Task exceeds the time budget {running_time} and has been cancelled")
#                 res = None
#             else:
#                 res = return_dict.get("res")
#                 p.terminate()
#             print(f"After running, there is {time_budget.remain: .4f}s remaining time")
#             return res
#         return timed
#     return wrapper


class TimeOutError(Exception):
    pass


class TimeBudget:
    def __init__(self, time_budget):
        self._time_budget = time_budget
        self._start_time = time.time()

    def reset(self):
        self._start_time = time.time()

    @property
    def remain(self):
        escape_time = time.time() - self._start_time
        return self._time_budget - escape_time

    @remain.setter
    def remain(self, value):
        self._time_budget = value

    def timing(self, seconds=None, frac=1.0):
        if seconds is None:
            seconds = self.remain * frac
        else:
            seconds = min(seconds, self.remain * frac)
        return TimeBudget(seconds)

    def check(self):
        if self.remain < 0:
            raise TimeOutError(f"Time out {self.remain: 0.4f}")

    def __add__(self, other):
        # self._time_budget += other
        return self

    def __sub__(self, other):
        # self._time_budget -= other
        return self

    def __str__(self):
        return str(self.remain)

    def __repr__(self):
        return repr(self.remain)

    def __format__(self, format_spec):
        return format(self.remain, format_spec)
