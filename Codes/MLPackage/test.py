

class Parent():

    def __init__(self, A, B):
        self.A = A
        self.B = B
    
    def set_A(self, A):
        self.A = A


class Child(Parent):

    def __init__(self, A, B, C):
        super().__init__(A, B)
        self.C = C


# P = Parent(1, 2)

C = Child(2, 3, 4)
breakpoint()