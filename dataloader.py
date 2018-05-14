

class Dataloader:
    def __init__(self, dataf):
        self.dataf = dataf
        self.data = []
        self.load()

    def load(self):
        global words, tags
        words, tags = [], []
        with open(self.dataf, "r") as f:
            for line in f:
                if line.strip("\n") == "":
                    if len(words) != 0:
                        self.data.append((words, tags))
                    words, tags = [], []
                else:
                    words.append(line.split()[0]); tags.append(line.split()[1])
            if len(words) != 0:
                self.data.append((words, tags))



if __name__ == "__main__":
    dl = Dataloader('../data/ner/train_small.eval')

