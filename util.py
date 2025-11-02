class Threshold():
    def __init__(self, lower_bound: tuple[int], upper_bound: tuple[int], color: str = None, ):
        self.color: str = color
        self.lower_bound: tuple[int] = lower_bound
        self.upper_bound: tuple[int] = upper_bound

class Blob():
    def __init__(self, center_x: int, center_y: int, size: int):
        self.center_x: int = center_x
        self.center_y:int  = center_y
        self.size: int = size
    
    def centroid(self):
        return(self.center_x, self.center_y)
    
