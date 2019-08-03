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
        first_file_idx = int(worker_num * len(files) / num_workers)
        last_file_idx = int((worker_num + 1) * len(files) / num_workers)
        cur_pos_file = pos_files_path if (pos_replacement or (not pos_replacement and worker_num == 0)) else None
        curr_worker = my_worker(queue=batches_queue,
                                batch_size=batch_size,
                                files=files[first_file_idx: last_file_idx],
                                n_features=n_features,
                                pos_file=cur_pos_file,
                                take_last_k_cycles=take_last_k_cycles,
                                use_string_loc=not ignore_string_loc,
                                pos_replacement=pos_replacement if worker_num == 0 else True)
        workers.append(curr_worker)

    for worker_id, worker in enumerate(workers):
        worker.start()
        # give the first thread some more time to load all of the positive examples
        if not pos_replacement and worker_id == 0:
            time.sleep(5)

    return batches_queue, workers
