#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import random
import pylab as pl
from matplotlib import collections  as mc

def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def is_close2(pa, pb, rel_tol=1e-09, abs_tol=0.0):
    return is_close(pa[0], pb[0]) and is_close(pa[1], pb[1], rel_tol, abs_tol)

def list_close(la, lb, rel_tol=1e-09, abs_tol=0.0):
    if len(la) != len(lb):
        return False
    for a in la:
        i = 0
        for b in lb:
            if is_close2(a, b, rel_tol, abs_tol):
                lb.pop(i)
                break
            i = i + 1
        else:
            return False
    return len(lb) == 0

def intersections(P1, P2, C):
    if P1 == P2 or C[1] == 0:
        return []
    c = np.array(C[0])
    r = C[1]
    p1, p2 = np.array(P1) - c, np.array(P2) - c
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dr = np.sqrt(dx * dx + dy * dy)
    D = p1[0] * p2[1] - p2[0] * p1[1]

    discriminant = r * r * dr * dr - D * D
    if discriminant < 0:
        return []
    else:
        sd = np.sqrt(discriminant)
        x1 = (D * dy + np.sign(dy) * dx * sd) / (dr * dr)
        y1 = ((-D) * dx + np.abs(dy) * sd) / (dr * dr)
        d1 = np.hypot(p1[0] - x1, p1[1] - y1)
        d2 = np.hypot(p2[0] - x1, p2[1] - y1)
        a = []
        if d1 < dr and d2 < dr:
            a.append((x1 + c[0], y1 + c[1]))

        x2 = (D * dy - np.sign(dy) * dx * sd) / (dr * dr)
        y2 = ((-D) * dx - np.abs(dy) * sd) / (dr * dr)
        if not (is_close(x1, x2) and is_close(y1, y2)):
            d1 = np.hypot(p1[0] - x2, p1[1] - y2)
            d2 = np.hypot(p2[0] - x2, p2[1] - y2)
            if d1 < dr and d2 < dr:
                a.append((x2 + c[0], y2 + c[1]))
        return a

def test_intersections():
    tests=[
    (((5, 5), (5, 5), [(1, 1), 10]), []), # undefined
    (((1, 2), (3, 4), [(2, 2), 0]), []), # undefined
    (((0, 0), (-1, -1), [(1, 1), 1]), []), # no intersect
    (((0, 0), (2, 0), [(1, 1), 1]), [(1.0, 0.0)]), # tangential y
    (((0, 0), (0, 2), [(1, 1), 1]), [(0.0, 1.0)]), # tangential x
    (((0, 0), (1, 1), [(1, 1), 1]), [(0.2928932188, 0.2928932188)]), # one intersect
    (((0, 0), (2, 4), [(1, 1), 1]), [(0.2, 0.4), (1.0, 2.0)]) # two intersect
    ]
    for t in tests:
        p1, p2, c = t[0]
        i = intersections(p1, p2, c)
        print(t, i, list_close(i,t[1]))

#test_intersections()

circles = [
  [(0, 0), 0.6]
]

N = 10000
L = []
C = []
CIRCLE=[(0, 0), 1.0]
G = 0
K = 0
MODE = 1
for i in range(N):
    p1, p2 = [], []
    if MODE == 0:
        p1 = (2.0 * random.random() - 1.0, 2.0 * random.random() - 1.0)
        p2 = (2.0 * random.random() - 1.0, 2.0 * random.random() - 1.0)
    elif MODE == 1: # random endpoints
        r = CIRCLE[1];
        a = random.random() * 4 * np.pi
        b = random.random() * 4 * np.pi
        p1 = (r * np.cos(a), r * np.sin(a))
        p2 = (r * np.cos(b), r * np.sin(b))
    elif MODE == 2: # random radial point
        r = CIRCLE[1] * random.random()
        a = random.random() * 4 * np.pi
        p1 = (np.cos(a), np.sin(a))
        p2 = (np.cos(b), np.sin(b))


    s = intersections(p1, p2, circles[0])
    S = circles[0][1] * np.sqrt(3.0)
    c = (0, 0, 0, 0.4)
    if len(s) > 0:
        c = (0, 0, 0, 0.7)
        if len(s) > 1:
            p1 = s[0]
            p2 = s[1]
            K = K + 1
            if np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) > S:
                G = G + 1
                c = (0, 1, 0, 1)
            else:
                c = (1, 0, 0, 1)

    L.append([p1, p2])
    C.append(c)


print(float(G) / float(K), G, K, N)
exit()
lc = mc.LineCollection(L, colors=C, linewidths=0.2)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.add_patch(pl.Circle(circles[0][0], circles[0][1], color='b', fill=False))
ax.autoscale()
#ax.margins(0.1)
pl.show()
