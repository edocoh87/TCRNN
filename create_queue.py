from worker import ProcessWorker, ThreadWorker
import multiprocessing
from pdb import set_trace as trace
import time

def create_queue_and_workers(is_debug, files, batch_size, n_features, pos_files_path, oversample_pos, ignore_string_loc,
                             take_last_k_cycles, num_workers, pos_replacement=True, prev_workers=None):
    if prev_workers and hasattr(prev_workers[0], 'terminate'):
        for prev_worker in prev_workers:
            prev_worker.terminate()

    my_worker = ThreadWorker if is_debug else ProcessWorker

    batches_queue = multiprocessing.Queue(maxsize=10)
    workers = []
    for worker_num in range(num_workers):
        first_nonfail_file_idx = int(worker_num * len(files) / num_workers)
        last_nonfail_file_idx = int((worker_num + 1) * len(files) / num_workers)
        if pos_files_path:
            first_fail_file_idx = int(worker_num * len(pos_files_path) / num_workers)
            last_fail_file_idx = int((worker_num + 1) * len(pos_files_path) / num_workers)
        curr_worker = my_worker(queue=batches_queue,
                                batch_size=batch_size,
                                files=files[first_nonfail_file_idx: last_nonfail_file_idx],
                                n_features=n_features,
                                pos_files=pos_files_path[first_fail_file_idx: last_fail_file_idx] if pos_files_path
                                else None,
                                take_last_k_cycles=take_last_k_cycles,
                                use_string_loc=not ignore_string_loc,
                                pos_replacement=pos_replacement)
        workers.append(curr_worker)

    for worker_id, worker in enumerate(workers):
        worker.start()

    return batches_queue, workers
