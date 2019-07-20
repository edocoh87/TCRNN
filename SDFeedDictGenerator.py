class FeedDictGenerator(object):
    def __init__(self, batches_queue, placeholders, use_seqlen, learning_rate_fn, dataset):
        self.batches_queue = batches_queue
        self.placeholders = placeholders
        self.use_seqlen = use_seqlen
        self.learning_rate_fn = learning_rate_fn
        self.step = 0
        self.dataset = dataset

    def create_feed_dict(self):
        x, y, lr, seqlen = self.placeholders
        feed_dict = {}
        if self.use_seqlen:
            val_batch_x, val_batch_y, val_batch_seqlen = self.batches_queue.get()
            feed_dict[seqlen] = val_batch_seqlen
        else:
            val_batch_x, val_batch_y = self.batches_queue.get()

        feed_dict.update({x: val_batch_x,
                          y: val_batch_y,
                          lr: self.learning_rate_fn(self.step)})
        self.step += 1
        return feed_dict
