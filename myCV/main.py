# CV


import argparse
# importing copy module
import math
import sys

import cv2 as cv
import copy
import numpy as np
from PIL import Image
from math import sqrt, acos


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='source image')
    parser.add_argument('-s', '--shape', help='input .txt file with shapes')
    return parser.parse_args()


def dist(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def dist2(x1, y1, x2, y2):
    d = float(sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)))
    return d

## repair
def is_tail(x1, y1, data):
    if neighbors3(x1, y1, data) == 1:
        return 1
    return 0


def draw_line(x1, y1, x2, y2, datanew, val):
    dx = x2 - x1
    dy = y2 - y1

    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    if dx < 0: dx = -dx
    if dy < 0: dy = -dy

    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy

    x, y = x1, y1

    error, t = el / 2, 0

    datanew[y][x] = val

    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        datanew[y][x] = val


# def fill(x0, y0, x1, y1, datanew):
#    deltax = abs(x1 - x0)
#    deltay = abs(y1 - y0)
#    error = 0
#    deltaerr = (deltay + 1) / (deltax + 1)
#    y = y0
#    diry = y1 - y0
#    if diry > 0:
#        diry = 1
#    if diry < 0:
#        diry = -1
#    for x in range(x0, x1):
#        datanew[y][x] = 255
#        error = error + deltaerr
#        if error >= 1.0:
#            y = y + diry
#            error = error - 1.0

### fillers
def check(dataold, i, j):
    cnt = 0
    for k in range(j, len(dataold)):
        if is_pixel(i, k, dataold) == 1 and not is_pixel(i, k - 1, dataold) == 1:
            cnt += 1
    return cnt


def refill(datanew, dataold):
    for i in range(1, len(dataold[0]) - 1):
        for j in range(1, len(dataold) - 1):
            if is_pixel(i, j, dataold) == 1 and is_pixel(i - 1, j, dataold) == 0 and is_pixel(i + 1, j, dataold) == 0 \
                    and is_pixel(i, j - 1, dataold) == 1 and is_pixel(i, j + 1, dataold) == 1:
                datanew[j][i] = 0
            if is_pixel(i, j, dataold) == 0 and is_pixel(i - 1, j, dataold) == 1 and is_pixel(i + 1, j, dataold) == 1:
                datanew[j][i] = 255


def fill(datanew, dataold):
    for i in range(len(dataold[0])):
        flag = False
        for j in range(len(dataold)):
            if is_pixel(i, j, dataold) == 1:
                a = check(dataold, i, j)
                if not a % 2 == 0 and not flag:
                    continue
            if is_pixel(i, j, dataold) == 1 and not is_pixel(i, j - 1, dataold) == 1:
                flag = not flag
            if flag:
                datanew[j][i] = 255

## repair
def fix(datanew, dataold):
    points = []
    flag = 0
    for i in range(len(dataold[0]) - 2):
        for j in range(len(dataold) - 2):
            if is_tail(i, j, dataold):
                points.append([i, j])
    if len(points) == 0:
        return 0
    for a in points:
        dista = 10000
        point = a
        for b in points:
            if not (a == b):
                if dist(a[0], a[1], b[0], b[1]) < dista:
                    point = b
                    dista = dist(a[0], a[1], b[0], b[1])
                    flag = 1
        if flag:
            ##!
            if dista < 10:
                draw_line(a[0], a[1], point[0], point[1], datanew, 255)
            else:
                flag = 0
    return flag

## base
def is_pixel(x1, y1, data):
    if data[y1][x1] == 0:
        return 0
    return 1


def neighbors3(x1, y1, data):
    cnt = 0
    if is_pixel(x1, y1, data):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if is_pixel(x1 + i, y1 + j, data):
                    cnt += 1
    return cnt


def neighbors5(x1, y1, data):
    cnt = 0
    if is_pixel(x1, y1, data):
        for i in [-2, -1, 0, 1, 2]:
            for j in [-2, -1, 0, 1, 2]:
                if i == 0 and j == 0:
                    continue
                if is_pixel(x1 + i, y1 + j, data):
                    cnt += 1
    return cnt


def is_noise3(x1, y1, data, tol):
    if neighbors3(x1, y1, data) < tol and is_pixel(x1, y1, data):
        return 1
    return 0


def is_noise5(x1, y1, data, tol):
    if neighbors5(x1, y1, data) < tol and is_pixel(x1, y1, data):
        return 1
    return 0


def denoise3(datanew, dataold, tol):
    flag = 0
    for i in range(len(dataold[0]) - 2):
        for j in range(len(dataold) - 2):
            if is_noise3(i + 1, j + 1, dataold, tol):
                flag = 1
                datanew[j + 1][i + 1] = 0
    return flag



def denoise5(datanew, dataold, tol):
    flag = 0
    for i in range(len(dataold[0]) - 4):
        for j in range(len(dataold) - 4):
            if is_noise5(i + 2, j + 2, dataold, tol):
                flag = 1
                datanew[j + 2][i + 2] = 0
    return flag

## merge
def merge(x1, y1, datanew, dataold):
    if dataold[y1][x1] != 127:
        return
    #print("in", x1, "--", y1)

    sum_i = 0
    sum_j  = 0
    cnt = 0
    for i in range(-10, 10, 1):
        for j in range(-10, 10, 1):
            var = 0
            if j + y1 > 199 or i + x1 > 299 or j + y1 < 0 or i + x1 < 0:
                var = 0
            else:
                var = dataold[j + y1][i + x1]
            if var == 127:
                cnt += 1
                sum_j += j + y1
                sum_i += i + x1
    if cnt == 0:
        return
    res_i = int(sum_i / cnt)
    res_j = int(sum_j / cnt)
    datanew[res_j][res_i] = 128
    #print("out", res_j, "--",  res_i)
    # clear
    for i in range(-10, 10, 1):
        for j in range(-10, 10, 1):
            if(dataold[j + y1][i + x1]) == 127:
                dataold[j + y1][i + x1] = 255


def mergeall(datanew, dataold):
    for i in range(1, len(dataold[0]) - 1):
        for j in range(1, len(dataold) - 1):
            merge(i, j, datanew, dataold)

def is_way (x1, y1, x2, y2, data):
    d = 1000
    for t in range(1000):
        t1 = t / d
        xt = int(x1 * t1 + x2 * (1 - t1))
        yt = int(y1 * t1 + y2 * (1 - t1))
        if data[yt][xt] == 0 and data[yt - 1][xt] == 0 and data[yt][xt - 1] == 0  and data[yt + 1][xt] == 0  and data[yt][xt + 1] == 0:
            return 0
    return 1
def is_way2 (x1, y1, x2, y2, data):
    d = 1000
    for t in range(1000):
        t1 = t / d
        xt = int(x1 * t1 + x2 * (1 - t1))
        yt = int(y1 * t1 + y2 * (1 - t1))
        if data[yt][xt] == 0 and data[yt - 1][xt] == 0 and data[yt][xt - 1] == 0  and data[yt + 1][xt] == 0  and data[yt][xt + 1] == 0:
            if data[yt + 1][xt + 1] == 0 and data[yt - 1][xt - 1] == 0 and data[yt - 1][xt + 1] == 0  and data[yt + 1][xt - 1] == 0:
                return 0
    return 1
def is_equal(s1, s2):
    if (s1[0] == s2[0] and s1[1] == s2[1]):
        return 1
    return 0
def is_connected(s1, s2):
    if is_equal(s1, s2):
        return 0
    if s1[0] == s2[0] or s1[0] == s2[1] or s1[1] == s2[0] or s1[1] == s2[1]:
        return 1


def makegroup(vert, gooddata, figures):
    sides = []
    for i in range(0, len(vert)):
        for j in range(i, len(vert)):
            v1 = vert[i]
            v2 = vert[j]
            if is_way(v1[0], v1[1], v2[0], v2[1], gooddata):
                sides.append((v1, v2))
    trash = []
    for s in sides:
        if s[0] == s[1]:
            trash.append(s)
    for s in trash:
        sides.remove(s)
    fig = [sides[0]]
    sides.remove(sides[0])
    cnt = 0
    while len(sides) > 0:
        if cnt > len(fig) - 1:
            figures.append(fig)
            fig = [sides[0]]
            sides.remove(sides[0])
            cnt = 0
        for s in sides:
            if is_connected(fig[cnt], s):
                fig.append(s)
                sides.remove(s)
        cnt += 1
    if len(fig) > 0:
        figures.append(fig)

def angle(ax,ay,bx,by):

    ma = sqrt(ax * ax + ay * ay)
    mb = sqrt(bx * bx + by * by)
    sc = ax * bx + ay * by
    res = acos(min(max(sc / ma / mb, -1), 1))
    sign = ax * by - ay * bx
    if sign > 0:
        return res
    else:
        return -res


def angle_of_vector(v1, v2):
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return acos(cos)


def rotatedeg(my_side, in_side):
    a = [(0, 0), (my_side[1][0] - my_side[0][0], my_side[1][1] - my_side[0][1])]
    b = [(0, 0), (in_side[1][0] - in_side[0][0], in_side[1][1] - in_side[0][1])]
    deg = angle(a[1][0], a[1][1], b[1][0], b[1][1])
    deg1 = angle(b[1][0], b[1][1], a[1][0], a[1][1])
    #deg3 = angle_of_vector((a[1][0], a[1][1]), (b[1][0], b[1][1]))
    #deg4 = angle_of_vector((b[1][0], b[1][1]), (a[1][0], a[1][1]))
    return deg / math.pi * 180


def shell(fig, m):
    n = int(1/2 * (1 + sqrt(8 * m + 1)))


    if m == n:
        return fig
    else:
        for s in fig:
            if len(fig) == n:
                return fig
            point = s[0]
            ways = []
            for s in fig:
                if s[0] == point:
                    ways.append(s)
                if s[1] == point:
                    stmp = s
                    tmp = stmp[0]
                    list(list(stmp)[0])[0] = stmp[1][0]
                    list(list(stmp)[0])[1] = stmp[1][1]
                    list(list(stmp)[1])[0] = tmp[0]
                    list(list(stmp)[1])[1] = tmp[1]
                    ways.append((stmp[1], stmp[0]))
            if len(ways) == 2:
                continue
            else:
                max = 0
                max1 = ways[0]
                max2 = ways[1]
                for w1 in ways:
                    for w2 in ways:
                        a = angle(w1[1][0] - point[0], w1[1][1] - point[1], w2[1][0] - point[0], w2[1][1] - point[1])
                        if a > max:
                            max = a
                            max1 = w1
                            max2 = w2
                for w in ways:
                    if not w == max1 and not w == max2:
                        w1 = w
                        tmp1 = w1[0]
                        list(list(w1)[0])[0] = w1[1][0]
                        list(list(w1)[0])[1] = w1[1][1]
                        list(list(w1)[1])[0] = tmp1[0]
                        list(list(w1)[1])[1] = tmp1[1]
                        if w in list(fig):
                            fig.remove(w)
                        if w1 in list(fig):
                            fig.remove(w1)
                        if w in list(ways):
                            ways.remove(w)
    return fig


def scale(fig, sc):
    fig2 = []
    for s in fig:
        s2 = []
        for p in s:
            p2 = []
            for c in p:
                p2.append(c * sc)
            s2.append(p2)
        fig2.append(s2)
    return fig2


def side_len(side):
    return dist2(side[0][0], side[0][1], side[1][0], side[1][1])


def find_scale(my_side, in_side):
    return side_len(my_side) / side_len(in_side)


def find_rotate(my_side, in_side):
    a = rotatedeg(in_side, my_side)
    while a < -180:
        a += 360
    while a > 180:
        a -= 360
    return a


def find_translate(my_side, in_side):
    return (my_side[0][0], my_side[0][1])


def is_sample(in_myfig, in_myperim, in_fig, in_perim):

    myfig = copy.deepcopy(in_myfig)
    myperim = copy.deepcopy(in_myperim)
    fig = copy.deepcopy(in_fig)
    perim = copy.deepcopy(in_perim)
    tol = 0.05
    c = myperim / perim
    fig2 = scale(fig, c)
    s = 0
    i = 0
    flag = 0
    trinity = []

    if len(myfig) != len(fig):
        return 0
    while s < len(myfig):
        flag = 0
        i = 0
        while i < len(fig2):
            s2 = fig2[i]

            if s < len(myfig) and i < len(fig2) and abs(side_len(myfig[s]) - side_len(s2)) < tol * side_len(s2):
                trinity.append((fig[i], fig2[i], myfig[s]))
                myfig.remove(myfig[s])
                fig2.remove(fig2[i])
                fig.remove(fig[i])
                flag = 1
            else:
                i += 1
        if flag == 0:
            s += 1
    if len(fig2) == 0:
        return trinity
    return 0


def find_params(trinity):
    out_scale = find_scale(trinity[2], trinity[0])
    out_rotate = find_rotate(trinity[2], trinity[0])
    out_translate = find_translate(trinity[2], trinity[0])
    return (out_scale, out_rotate, out_translate)

def cut_lines(datanew, dataold, tails):
    for l1 in tails:
        for l2 in tails:
            if is_way2(l1[0], l1[1], l2[0], l2[1], dataold):
                draw_line(l1[0], l1[1], l2[0], l2[1], datanew, 0)
                #draw_line(l1[0] - 1, l1[1], l2[0] - 1, l2[1], datanew, 0)
                draw_line(l1[0] + 1, l1[1], l2[0] + 1, l2[1], datanew, 0)
                #draw_line(l1[0], l1[1] - 1, l2[0], l2[1] - 1, datanew, 0)
                #draw_line(l1[0], l1[1] + 1, l2[0], l2[1] + 1, datanew, 0)



def print_hi():

    args = parse_arguments()
    noiseflag = 0
    #f = open("test.txt")
    f = open(args.shape)
    N = int(f.readline())
    list = []
    for i in range(N):
        obj = []
        s = f.readline()
        s1 = s.split(',')
        for i in range(int(len(s1) / 2)):
            obj.append([int(s1[i * 2]), int(s1[i * 2 + 1])])
        list.append(obj)
    im = Image.open(args.image)
    #im = Image.open("test2.png")

    pixelMap = im.load()
    img = Image.new(im.mode, im.size)
    pixelsNew = img.load()
    dataold = [[0] * im.size[0] for i in range(im.size[1])]
    img.save("out.png")
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixelMap[i, j][0] != 255 and pixelMap[i, j][0] != 0:
                noiseflag = 1
            if not (pixelMap[i, j][0] != 255 and pixelMap[i, j][1] != 255 and pixelMap[i, j][2] != 255):
                pixelsNew[i, j] = (255, 255, 255, 255)
                dataold[j][i] = 255
            else:
                pixelsNew[i, j] = (0, 0, 0, 255)
                dataold[j][i] = 0

    # denoise
    # flag = 1
    # while flag == 1:
    #    datanew = copy.deepcopy(dataold)
    #    flag = denoise5(datanew, dataold, 3)
    #    dataold = copy.deepcopy(datanew)
    copy.deepcopy(dataold)
    # denoise5(datanew, dataold, 4)
    # dataold = copy.deepcopy(datanew)
    # datanew = copy.deepcopy(dataold)
    # denoise3(datanew, dataold, 2)
    # dataold = copy.deepcopy(datanew)

    datanew = copy.deepcopy(dataold)
    denoise3(datanew, dataold, 2)
    dataold = copy.deepcopy(datanew)

    ## tail fix
    if not noiseflag:
        tails = []
        for i in range(img.size[0] - 1):
            for j in range(img.size[1] - 1):
                if datanew[j][i] == 255:
                    pixelsNew[i, j] = (0, 255, 255, 255)
                    if is_tail(i, j, datanew):
                        tails.append((i, j))

        datanew = copy.deepcopy(dataold)
        cut_lines(datanew, dataold, tails)
        dataold = copy.deepcopy(datanew)

        datanew = copy.deepcopy(dataold)
        denoise3(datanew, dataold, 2)
        dataold = copy.deepcopy(datanew)
    #####~~~~~
    flag = 1
    while flag:
        datanew = copy.deepcopy(dataold)
        flag = fix(datanew, dataold)
        dataold = copy.deepcopy(datanew)
    datanew = copy.deepcopy(dataold)
    dataslim = copy.deepcopy(dataold)
    fill(datanew, dataold)
    dataold = copy.deepcopy(datanew)
    datanew = copy.deepcopy(dataold)
    refill(datanew, dataold)
    dataold = copy.deepcopy(datanew)

    if not noiseflag:
        flag = 1
        while flag == 1:
            datanew = copy.deepcopy(dataold)
            flag = denoise3(datanew, dataold, 2)
            dataold = copy.deepcopy(datanew)



    # for i in range(img.size[0] - 2):
    #    for j in range(img.size[1] - 2):
    #        if is_noise(i + 1, j + 1, dataold):
    #            flag = 1
    #            #pixelsNew2[i + 1, j + 1] = (0, 0, 0, 255)
    #            datanew[j + 1][i + 1] = 0

    for i in range(img.size[0] - 1):
        for j in range(img.size[1] - 1):
            if datanew[j][i] == 255:
                pixelsNew[i, j] = (0, 255, 255, 255)
                if is_tail(i, j, datanew):
                    pixelsNew[i, j] = (0, 0, 255, 255)
            else:
                pixelsNew[i, j] = (0, 0, 0, 255)
    img.save("out.png")

    gooddata = copy.deepcopy(datanew)
    filename = 'out.png'
    img2 = cv.imread(filename)
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 5, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.

    for i in range(img.size[0] - 1):
        for j in range(img.size[1] - 1):
            if dst[j][i] > 0.01 * dst.max() and neighbors3(i, j, dataslim) == 2:
                pixelsNew[i, j] = (255, 0, 255, 255)
                dataold[j][i] = 127

    datanew = copy.deepcopy(dataold)
    mergeall(datanew, dataold)
    dataold = copy.deepcopy(datanew)

    vert = []
    for i in range(img.size[0] - 1):
        for j in range(img.size[1] - 1):
            if datanew[j][i] == 128:
                pixelsNew[i, j] = (255, 255, 0, 255)
                vert.append((i, j))
    img.save("outc.png")
    figgroup = []
    makegroup(vert, gooddata, figgroup)
    shell(figgroup[0], len(figgroup[0]))
    shell(figgroup[1], len(figgroup[1]))
    shell(figgroup[2], len(figgroup[2]))

    perims = []
    in_figs = []
    for f in list:
        sum = 0.0
        in_fig = []
        for j in range(0, len(f) - 1):
            sum += dist2(f[j][0], f[j][1], f[j + 1][0], f[j + 1][1])
            in_fig.append((f[j], f[j + 1]))
        sum += dist2(f[len(f) - 1][0], f[len(f) - 1][1], f[0][0], f[0][1])
        in_fig.append((f[len(f) - 1], f[0]))
        perims.append(sum)
        in_figs.append(in_fig)
    myperims = []
    for f in figgroup:
        sum = 0.0
        for j in range(0, len(f)):
            sum += dist2(f[j][0][0], f[j][0][1], f[j][1][0], f[j][1][1])
        myperims.append(sum)

    a = is_sample(figgroup[2], myperims[2], in_figs[1], perims[1])
    out = []
    for i in range(0, len(figgroup)):
        for j in range(0, len(in_figs)):
            a = is_sample(figgroup[i], myperims[i], in_figs[j], perims[j])
            if a == 0:
                continue
            else:
                for t in a:
                    if list[j][0] == t[0][0] and list[j][1] == t[0][1]:
                        answer = find_params(t)
                        num = j
                        dx = int(answer[2][0])
                        dy = int(answer[2][1])
                        sc = int(answer[0])
                        rot = int(answer[1])
                        out.append((j, dx, dy, sc, rot))
    print(len(out))
    for i in out:
        print(i[0], ", ", i[1], ", ", i[2], ", ", i[3], ", ", i[4], sep='')
    #img2[dst > 0.01 * dst.max()] = [0, 0, 255]
    #cv.imshow('dst', img2)

    #if cv.waitKey(0) & 0xff == 27:
    #    cv.destroyAllWindows()

    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
