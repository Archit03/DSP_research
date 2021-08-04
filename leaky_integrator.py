class leaky:
    def __init__(self, lmb):
        self.lmb = lmb 
        self.y = 0 
        
    def compute(self, x):
        res = [] 
        for v in x:
            self.y = self.lmb * self.y + (1 - self.lmb) * v 
            res.append(self.y) 
        return res 
    
if __name__ == '__main__':
    L = leaky(0.95)
    print(L.compute([0,0,0,0,1,0,0,0,0,0]))
