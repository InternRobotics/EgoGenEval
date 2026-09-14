"""Run independent source-scene jobs, joining workers before staging cleanup."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context


def scene_jobs(function, jobs, workers, *, processes=False):
    jobs = list(jobs)
    if workers == 1 or len(jobs) < 2:
        return [function(job) for job in jobs]
    # EGL contexts must be created in fresh processes, never shared by threads
    # or inherited through fork from the environment preflight renderer.
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if processes else ThreadPoolExecutor(max_workers=workers))
    futures = {pool.submit(function, job): index for index, job in enumerate(jobs)}
    results = [None] * len(jobs)
    try:
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results
