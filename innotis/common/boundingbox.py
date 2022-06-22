class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = int(float(classID))
        self.confidence = float(confidence)
        self.x1 = int(float(x1))
        self.x2 = int(float(x2))
        self.y1 = int(float(y1))
        self.y2 = int(float(y2))
        
        self.u1 = float(x1 / image_width)
        self.u2 = float(x2 / image_width)
        self.v1 = float(y1 / image_height)
        self.v2 = float(y2 / image_height)
    
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
