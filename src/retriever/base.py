

class BaseRetriever:
    def __init__(self):
        pass

    def retrieve(self, query, k=1):
        raise NotImplementedError("Retrieve method is not implemented")
    
    def retrieve_batch(self, queries, k=1):
        return [self.retrieve(query, k) for query in queries]
    
    def save(self, path):
        raise NotImplementedError("Save method is not implemented")
    
    def load(self, path):
        raise NotImplementedError("Load method is not implemented")
