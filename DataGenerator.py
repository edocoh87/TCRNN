class DataGenerator(object):
    
    def next(self, batch_size):
        raise NotImplementedError
    
    def get_validation(self):
        return None
