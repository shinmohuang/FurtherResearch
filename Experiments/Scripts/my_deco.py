import functools
import time


def running_time(func):
    """
    calculate the running time of a function

    Args:
        func: the function to be calculated

    Returns:
        wrapper: the wrapper function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Start time: {start_time}")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"End time: {end_time}")
        print(f"Running time: {end_time - start_time}")
        return result

    return wrapper


def debug(func):
    """
    print the name of the function

    Args:
        func: the function to be debugged

    Returns:
        wrapper: the wrapper function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function name: {func.__name__} -args: {args} -kwargs: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


def suppress_error(func):
    """
    suppress the error of a function

    Args:
        func: the function to be suppressed

    Returns:
        wrapper: the wrapper function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f'Error in function {func.__name__}:{e}')
            return None

    return wrapper
