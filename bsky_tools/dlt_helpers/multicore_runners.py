from multiprocessing import set_start_method
# set_start_method("forkserver", True)

import os
import threading
from multiprocessing import Process, Manager
from typing import Callable, List
from tqdm import tqdm
import time

from ..dlt_helpers.parsing import get_posts_adaptive_sliding_window_reverse as posts_func
from ..dlt_helpers.parsing import get_posts_count_adaptive_sliding_window_reverse as post_count_func

def pool_func_posts(task_queue, progress_queue, result_queue):
    """Worker function for the process pool."""
    while True:
        task = task_queue.get()
        if task is None:
            break

        query, start_date, end_date = task
        try:
            result = [*posts_func(query, start_date, end_date)]
            progress_queue.put(1)  # Signal completion
            result_queue.put(result)
            
        except Exception as e:
            print(f"Error processing task: {task}, error: {e}")
            result = []  # Or handle the error as needed
            result_queue.put(result)
            progress_queue.put(1) # Signal to move to next task (or end threads)
        # task_queue.task_done()  # Signal that the task is done here -> IMPORTANT

import logging
import random

def pool_func_post_count(task_queue, progress_queue, result_queue):
    """Worker function for the process pool."""
    while True:
        task = task_queue.get()
        if task is None:
            break

        query, start_date, end_date = task
        try:
            result = [*post_count_func(query, start_date, end_date)]
            progress_queue.put(1)  # Signal completion
            result_queue.put(result)
            
        except Exception as e:
            log_file_name = f"{query}_{start_date}_{end_date}.log"
            logging.basicConfig(filename=log_file_name, level=logging.ERROR,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            error_message = f"Error processing task: {task}, error: {e}"
            logging.exception(error_message) # Log the entire exception, including stack trace
            result = []  # Or handle the error as needed
            result_queue.put(result)
            progress_queue.put(1) # Signal to move to next task (or end threads)
        # task_queue.task_done()  # Signal that the task is done here -> IMPORTANT
        
def _run_query_pool(
    param_tuple: tuple,
    pool_func: Callable,
    n_cpus: int = os.cpu_count(),
    yield_flag: bool = False
) -> List[List[dict]]:
    """
    Executes the given function in parallel using a multiprocessing pool,
    along with a progress bar updated via a queue.
    Args:
        param_tuple (tuple): A tuple of parameters to pass to the `pool_func`.
        pool_func (Callable): The function to execute in parallel.
                          Defaults to `get_posts_count_adaptive_sliding_window_reverse`
        n_cpus (int): The number of CPUs to use for the pool. Defaults to `os.cpu_count()`.
    Returns:
        List[List[dict]]: A list of results from the `pool_func` for each input parameter set.
    """

    results = []
    manager = Manager()
    task_queue = manager.Queue()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    tasks = [*param_tuple]
    total_tasks = len(tasks)


    def progress_updater(queue, total):
        with tqdm(total=total) as progress_bar:
            count = 0
            while count < total:
                try:
                    queue.get()  # Set a timeout
                    progress_bar.update(1)
                    count += 1
                except Exception as e: # queue.Empty:
                    # Handle timeout or empty queue as needed
                    #print("Progress queue empty, checking for completion...")
                    #time.sleep(0.1)
                    pass

    # Start the progress bar thread
    progress_thread = threading.Thread(
        target=progress_updater, args=(progress_queue, total_tasks)
    )
    progress_thread.daemon = True  # Thread will exit when the main program exits
    progress_thread.start()

    # Create and start worker processes
    processes = []
    for _ in range(n_cpus):
        p = Process(target=pool_func, args=(task_queue, progress_queue, result_queue))
        p.start()
        processes.append(p)
    # Queue them up to the work queue
    [task_queue.put(task) for task in tasks]

    # Signal the workers to terminate
    for _ in range(n_cpus):
       task_queue.put(None)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Wait for all tasks to be completed by each of the child processes
    while any([p.is_alive() for p in processes]):
        for i,p in enumerate(processes):
            p.join()
            if p.is_alive():
                logging.info(f"Process {i} - alive")
                
    progress_queue.put(None)
    # Signal all jobs processed
    # progress_queue.put("DONE")
    progress_thread.join()


    # Collect results from the result queue
    while not result_queue.empty():
        results.append(result_queue.get())
        # print(f"Remaining: {result_queue.qsize()}")

    #result_queue.put(None) # Not needed
    
    if yield_flag:
        # yield from results
        def yield_results():
            yield from results
            
        return yield_results()

    if not yield_flag:
        return results