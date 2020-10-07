import numpy as np
from PIL import Image
import cv2

class constSep(object):
    def __init__(self, hole, constraint):
        super(constSep, self).__init__()
        self.hole = np.array(Image.open(hole))
        self.hole[np.where(self.hole < 128)] = 0
        self.constraint = np.array(Image.open(constraint))
        self.constraint[np.where(self.constraint < 128)] = 0
        self.line = np.where(self.constraint == 0)
        self.coefficients = np.polyfit(self.line[0], self.line[1], 1)
        self.polynomial = np.poly1d(self.coefficients)
    def sep(self):
        h,w = self.hole.shape
        left = np.ones(self.hole.shape)
        right = np.ones(self.hole.shape)
        index = np.where(self.hole == 0)
        for i,j in zip(index[0], index[1]):
            value = self.polynomial(i)
            if value < j:
                left[i,j] = 0
            else:
                right[i,j] = 0
        left = left * 255
        right = right * 255
        return left, right


if __name__ == "__main__":
    hole = "./image/sea_mask2.png" # binary hole mask. h*w
    constraint = "./image/sea_const1.png" # binary constraint mask. h*w
    model = constSep(hole, constraint) # init model
    left, right = model.sep() # separate hole mask based on constraint mask. output separated left and right hole mask. h*w 
    
    # demo output
    ## left hole mask
    imgLeft = Image.fromarray(left.astype(np.uint8))
    imgLeft.save("./image/sea_mask_left.png")

    ## right hole mask
    imgRight = Image.fromarray(right.astype(np.uint8))
    imgRight.save("./image/sea_mask_right.png")