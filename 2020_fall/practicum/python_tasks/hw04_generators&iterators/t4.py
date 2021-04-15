# %%
class WordContextGenerator:
    def __init__(self, words, window_size):
        self.words = words
        self.ws = window_size

    def __iter__(self):
        for i in range(len(self.words)):
            for j in range(max(i-self.ws, 0),
                           min(i+self.ws+1, len(self.words))):
                if i != j:
                    yield self.words[i], self.words[j]
