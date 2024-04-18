from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain
from datetime import datetime
import random
import math
from functools import wraps
import time

SAMPLE_RATE = 16_000

# class JupyterProfiler:
#     def __init__(self):
#         self.profiler = pyinstrument.Profiler()

#     def __enter__(self):
#         self.profiler.start()
#         return True


#     def __exit__(self, type, value, traceback):
#         self.profiler.stop()
#         ipd.display(ipd.HTML(self.profiler.output_html()))


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log(msg):
    print(f"[{get_timestamp()}]\t{msg}")


def border_msg(msg):
    row = len(msg)
    h = "".join(["+"] + ["-" * row] + ["+"])
    result = "\n" + h + "\n" "|" + msg + "|" "\n" + h
    return result


def time_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print()
        start_time = time.perf_counter()
        message = f"Beginning {func.__name__}"
        log(border_msg(message))
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        time_took = end_time - start_time
        unit = "seconds"
        if time_took > 60:
            time_took /= 60
            unit = "minutes"

        message = f"{func.__name__} took {time_took:.2f} {unit}"
        log(border_msg(message))
        print()
        return result

    return wrapper


def process_chunk(ix, tot_chunks, chunk, fn, num_threads, fn_kwargs):
    if num_threads > 1:
        with tqdm(total=len(chunk), leave=True, desc=f"Worker: {ix}") as progress:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for elem in chunk:
                    future = executor.submit(fn, ix, tot_chunks, elem, **fn_kwargs)
                    future.add_done_callback(lambda _: progress.update())
                    futures.append(future)

                results = list(as_completed(futures))
        results = list((map(lambda x: x.result(), results)))
    else:
        results = fn(ix, tot_chunks, chunk, **fn_kwargs)
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@time_me
def multicore_thread_process(num_workers, num_threads, chunked_args, fn, **fn_kwargs):
    assert num_workers > 0
    assert num_threads > 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunked_args):
            future = executor.submit(
                process_chunk,
                ix=i,
                tot_chunks=len(chunked_args),
                chunk=chunk,
                fn=fn,
                num_threads=num_threads,
                fn_kwargs=fn_kwargs,
            )
            futures.append(future)
        results = list(as_completed(futures))
        # results = list(chain(*map(lambda x: x.result(), results)))
    return results


def split_to_chunks_of_size(a: list, chunk_size: int, shuffle: bool = False):
    if shuffle:
        random.shuffle(a)
    for i in range(0, len(a), chunk_size):
        yield a[i : i + chunk_size]


def split_to_n_chunks(a: list, n: int, shuffle: bool = False):
    chunk_size = math.ceil(len(a) / n)
    return split_to_chunks_of_size(a, chunk_size=chunk_size, shuffle=shuffle)
