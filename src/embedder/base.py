class BaseEmbedder(object):
    def __init__(self):
        pass

    def embed(self, text):
        raise NotImplementedError("Embed method is not implemented")

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def save(self, path):
        raise NotImplementedError("Save method is not implemented")

    def load(self, path):
        raise NotImplementedError("Load method is not implemented")
