
class TokenAcc:
    def __init__(self):
        self.value = 0.
        self.n = 0
        self.tvalue = 0.
        self.tn = 0

    def get(self):
        if self.tn == 0: return None
        res = self.tvalue / self.tn
        self.tvalue = 0.
        self.tn = 0
        return res

    def update(self, x):
        self.tvalue += float(x)
        self.tn += 1
        self.value += float(x)
        self.n += 1

    def getAll(self):
        if self.n == 0: return None
        return self.value / self.n

    def reset(self):
        self.value = 0.
        self.n = 0
        self.tvalue = 0.
        self.tn = 0
