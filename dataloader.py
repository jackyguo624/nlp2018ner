

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
                print(line)
                if line.strip("\n") == "":
                    self.data.append((words, tags))
                    words, tags = [], []
                else:
                    words.append(line.split()[0]); tags.append(line.split()[1])
            if words != ():
                self.data.append((words, tags))



if __name__ == "__main__":
    dl = Dataloader('../data/ner/train_small.eval')

