class FeedDictGenerator(object):
    def __init__(self, batches_queue, placeholders, use_seqlen, learning_rate_fn, dataset):
        self.batches_queue = batches_queue
        self.placeholders = placeholders
        self.use_seqlen = use_seqlen
        self.learning_rate_fn = learning_rate_fn
        self.step = 0
        self.dataset = dataset

    def create_feed_dict(self):
        x, y, lr, is_train, seqlen = self.placeholders
        feed_dict = {}
        batch = self.batches_queue.get()
        if not batch:
            return None
        if self.use_seqlen:
            val_batch_x, val_batch_y, val_batch_seqlen = batch
            feed_dict[seqlen] = val_batch_seqlen
        else:
            val_batch_x, val_batch_y = batch

        feed_dict.update({x: val_batch_x,
                          y: val_batch_y,
                          lr: self.learning_rate_fn(self.step),
                          is_train: self.dataset == 'Train'})
        self.step += 1
        return feed_dict

    def set_queue(self, queue):
        self.batches_queue = queue
