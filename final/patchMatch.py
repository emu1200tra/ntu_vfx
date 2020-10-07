import numpy as np
from PIL import Image
import time
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Pool

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./input.jpg", help='path to input image')
    parser.add_argument("--ref", type=str, default="./ref.jpg", help='path to reference image')
    parser.add_argument("--output", type=str, default="./output.jpg", help="path to output image")
    parser.add_argument("--hole", type=str, default="./hole.jpg", help="path to hole mask image")
    parser.add_argument("--constraint", type=str, default="./constraint.jpg", help="path to constraint mask image. None for no constraint.")
    parser.add_argument("--iteration", type=int, default=5, help="iteration num")
    parser.add_argument("--psize", type=int, default=3, help="patch size")
    parser.add_argument("--workers", type=int, default=8, help="num workers")
    parser.add_argument("--stride", type=int, default=2, help="num workers")
    opt = parser.parse_args()
    return opt

def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    temp = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :] - A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

def parallelInit(inputParam):
    i,j,random_B_r,random_B_c = inputParam
    a = np.array([i, j])
    b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
    return (i,j), a, b

def propagation(inputParam):
    f, a, dist, A_padding, B, p_size, is_odd = inputParam
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    x = a[0]
    y = a[1]
    newf = f[x,y]
    newdist = dist[x,y]
    if is_odd:
        d_left = dist[max(x-1, 0), y]
        d_up = dist[x, max(y-1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            newf = f[max(x - 1, 0), y]
            newdist = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2:
            newf = f[x, max(y - 1, 0)]
            newdist = cal_distance(a, f[x, y], A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y]
        d_down = dist[x, min(y + 1, A_w-1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            newf = f[min(x + 1, A_h-1), y]
            newdist = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2:
            newf = f[x, min(y + 1, A_w-1)]
            newdist = cal_distance(a, f[x, y], A_padding, B, p_size)
    return (x,y), newf, newdist


def random_search(inputParam):
    f, a, dist, A_padding, B, p_size, flag = inputParam
    alpha=0.5
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    newdist = dist[x, y]
    newf = f[x, y]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h-p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < newdist:
            newdist = d
            newf = b
        i += 1
    return (x,y), newf, newdist

class patchMatch(object):
    def __init__(self, reference, p_size):
        super(patchMatch, self).__init__()
        self.B = reference
        self.p_size = p_size

    def cal_distance(self, a, b, A_padding):
        p = self.p_size // 2
        temp = self.B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :] - A_padding[a[0]:a[0]+self.p_size, a[1]:a[1]+self.p_size, :]
        num = np.sum(1 - np.int32(np.isnan(temp)))
        dist = np.sum(np.square(np.nan_to_num(temp))) / num
        return dist

    def reconstruction(self, f, A):
        A_h = np.size(A, 0)
        A_w = np.size(A, 1)
        temp = np.zeros_like(A)
        for i in range(A_h):
            for j in range(A_w):
                temp[i, j, :] = self.B[f[i, j][0], f[i, j][1], :]
        return temp

    def initialization(self, A, workers=4):
        #print("initializing...")
        A_h = np.size(A, 0)
        A_w = np.size(A, 1)
        B_h = np.size(self.B, 0)
        B_w = np.size(self.B, 1)
        p = self.p_size // 2
        random_B_r = np.random.randint(p, B_h-p, [A_h, A_w])
        random_B_c = np.random.randint(p, B_w-p, [A_h, A_w])
        A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
        A_padding[p:A_h+p, p:A_w+p, :] = A
        f = np.zeros([A_h, A_w], dtype=object)
        dist = np.zeros([A_h, A_w])

        param = []
        for i in range(A_h):
            for j in range(A_w):
                param.append([i,j,random_B_r,random_B_c])
        #print("cpu num: {}".format(workers))
        p = Pool(processes = workers)
        data = p.map(parallelInit, param)
        p.close()

        for i in data:
            index, tmpa, tmpb = i  
            f[index] = tmpb
            dist[index] = self.cal_distance(tmpa, tmpb, A_padding)

        return f, dist, A_padding

    def pack(self, itr, A_h, A_w, f, dist, img_padding):
        param = []
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    param.append([f, a, dist, img_padding, self.B, self.p_size, False])
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    param.append([f, a, dist, img_padding, self.B, self.p_size, True])
        return param

    def NNS(self, img, iteration, workers=4):
        A_h = np.size(img, 0)
        A_w = np.size(img, 1)
        f, dist, img_padding = self.initialization(img, workers)
        for itr in (range(1, iteration+1)):
            #print("prepare propogation packing...")
            param = self.pack(itr, A_h, A_w, f, dist, img_padding)
            p = Pool(processes = workers)
            
            #print("Propogation start\r")
            start = time.time()
            data = p.map(propagation, param)
            p.close()
            end = time.time()
            #print("time cost propogation: {}".format(end - start))

            for i in data:
                index, newf, newdist = i  
                f[index] = newf
                dist[index] = newdist
            #print("Propogation finish\n")
            #print("prepare search packing...")
            param = self.pack(itr, A_h, A_w, f, dist, img_padding)
            p = Pool(processes = workers)
            #print("Search start\r")
            start = time.time()
            data = p.map(random_search, param)
            p.close()
            end = time.time()
            #print("time cost search: {}".format(end - start))

            for i in data:
                index, newf, newdist = i  
                f[index] = newf
                dist[index] = newdist
            #print("Search finish\n")
            #print("iteration: %d"%(itr))
        return f
        
if __name__ == "__main__":
    opt = config() # get parameters
    img = np.array(Image.open(opt.input)) # read in hole image example
    #img = img[:50, :50, :] # crop to serve as a patch
    #tmp = Image.fromarray(img)
    #tmp.save("crop.jpg")
    ref = np.array(Image.open(opt.ref)) # read in reference image example
    p_size = opt.psize
    ##### NNS usage part #####
    model = patchMatch(ref, p_size) # init patchMatch model
    itr = opt.iteration
    print("start NNS")
    start = time.time()
    f = model.NNS(img, itr, opt.workers) # start NNS and get index map f
    end = time.time()
    print("time cost NNS: {}".format(end - start))
    ##########################
    print("start reconstruct")
    start = time.time()
    result = model.reconstruction(f, img) # resconstruct back to complete image
    end = time.time()
    print("time cost reconstruct: {}".format(end - start))
    result = Image.fromarray(result.astype(np.uint8))
    result.save(opt.output)