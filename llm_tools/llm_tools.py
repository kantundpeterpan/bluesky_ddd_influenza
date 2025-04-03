from ..utils import ChatOpenRouter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, List, Iterable
from multiprocessing import Pool, Queue, Process, Manager
import threading
from tqdm import tqdm  # For the progress bar
from pathlib import Path
import os
import pandas as pd

llm_tool_dir = Path(__file__).parent

class ILIAnswer(TypedDict):
    ili_related: Annotated[bool, ..., "Indicator whether message is ILI related"]
    symptoms: Annotated[List[str], ..., "List of symptoms mentioned in the the messages, regardless if ILI related or not"]
    
class SymptomExtractor():
    
    def __init__(self, model: str = 'google/gemini-2.0-flash-001'):
        self.model = ChatOpenRouter(model = model, temperature = 0)
        with open(llm_tool_dir / "symptom_extraction_prompt_en", "r") as f:
            self.prompt = SystemMessage(f.read())
            
        self.msg_template ="""Extract the symptoms from the following message :
        {msg}
        """
        
    def _pool_func(self, task_queue, progress_queue, result_queue):
        """Worker function for the process pool."""
        while True:
            task = task_queue.get()
            if task is None:
                break

            msg_id, msg = task
            try:
                result = self.extract_symptoms(msg_id, msg)
                progress_queue.put(1)  # Signal completion
                result_queue.put(result)
                
            except Exception as e:
                print(f"Error processing task: {task}, error: {e}")
                result = []  # Or handle the error as needed
                progress_queue.put(1) # Signal to move to next task (or end threads)
                
    def _run_extraction_pool(
        self,
        param_tuples: List[tuple],
        # pool_func: Callable = pool_func,
        n_jobs: int = os.cpu_count()
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
        tasks = param_tuples
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
        # progress_thread.daemon = True  # Thread will exit when the main program exits
        progress_thread.start()

        # Create and start worker processes
        processes = []
        for _ in range(n_jobs):
            p = Process(target=self._pool_func, args=(task_queue, progress_queue, result_queue))
            p.start()
            processes.append(p)
        # Queue them up to the work queue
        [task_queue.put(task) for task in tasks]

        # Signal the workers to terminate
        for _ in range(n_jobs):
            task_queue.put(None)

        # Wait for all tasks to be completed by each of the child processes
        for p in processes:
            p.join()
        # It is important to join all processes before proceeding

        progress_queue.put(None)
        # Signal all jobs processed
        # progress_queue.put("DONE")
        progress_thread.join()


        # Collect results from the result queue
        while not result_queue.empty():
            results.append(result_queue.get())
            # print(f"Remaining: {result_queue.qsize()}")

        #result_queue.put(None) # Not needed

        return results
                
    def extract_symptoms(self, msg_id: str, msg: str):
        messages = [
            self.prompt,
            HumanMessage(self.msg_template.format(msg = msg))
        ]
    
        response = self.model.with_structured_output(ILIAnswer).invoke(
                    messages
                )
        
        response['uri'] = msg_id
        
        return response
    
    def multi_extract(self, msgs: Iterable, n_jobs: int = os.cpu_count()):
        
        self.results = self._run_extraction_pool(n_jobs=n_jobs, param_tuples=msgs)
        self.resdf = pd.DataFrame.from_records(self.results)
        