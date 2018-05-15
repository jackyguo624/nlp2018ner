
def F1(predict, label):
    assert len(predict) == len(label)
    nacc = 0.
    for i in range(len(predict)):
        if predict[i] == label[i]:
            nacc+=1
    return nacc / len(predict)



class TokenAcc:
    def __init__(self):
        self.value = 0.
        self.n = 0
        self.tvalue = 0.
        self.tn = 0
        self.f1value = 0.
        self.f1n = 0.

    def updatef1(self, pred, target):
        self.f1value += F1(pred, target)
        self.f1n += 1

    def getf1(self):
        if self.f1n == 0.: return None
        res = self.f1value / self.f1n
        self.f1value = 0.
        self.f1n =0.
        return res

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
        self.f1value = 0.
        self.f1n = 0.
