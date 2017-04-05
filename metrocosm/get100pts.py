# General libraries.
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos,sin,arccos,sqrt
from collections import namedtuple
from math import sqrt

def inv_parametric_circle(x,xc,R):
    t = arccos((x-xc)/R)
    return t

def parametric_circle(t,xc,yc,R):
    x = xc + R*cos(t)
    y = yc + R*sin(t)
    return np.array((x, y)).astype(int)

def find_center(p1, p2, angle):
    # End points of the chord
    x1, y1 = p1 
    x2, y2 = p2 

    # Slope of the line through the chord
    slope = (y1-y2)/(x1-x2)

    # Slope of a line perpendicular to the chord
    new_slope = -1/slope

    # Point on the line perpendicular to the chord
    # Note that this line also passes through the center of the circle
    xm, ym = (x1+x2)/2, (y1+y2)/2
#    plt.scatter([x1],[y1],color='y',s=100, label='p1')
#    plt.scatter([x2],[y2],color='y',s=100)
#    plt.scatter([xm],[ym],color='g',s=100)

    # Distance between p1 and p2
    d_chord = sqrt((x1-x2)**2 + (y1-y2)**2)

    # Distance between xm, ym and center of the circle (xc, yc)
    d_perp = d_chord/(2*np.tan(angle))

    # Equation of line perpendicular to the chord: y-ym = new_slope(x-xm)
    # Distance between xm,ym and xc, yc: (yc-ym)^2 + (xc-xm)^2 = d_perp^2
    # Substituting from 1st to 2nd equation for y,
    #   we get: (new_slope^2+1)(xc-xm)^2 = d^2

    # Solve for xc:
    xc = int((d_perp)/sqrt(new_slope**2+1))
    xc2 = int(xm + np.abs(xm - xc))
    # Solve for yc:
    yc = int((new_slope)*(xc-xm)+ym)
    yc2 = int((new_slope)*(xc2-xm)+ym)
    return [[xc,yc], [xc2, yc2]]

Pt = namedtuple('Pt', 'x, y')
Circle = Cir = namedtuple('Circle', 'x, y, r')

def circles_from_p1p2r(p1, p2, r):
    'Following explanation at http://mathforum.org/library/drmath/view/53027.html'
    if r == 0.0:
        raise ValueError('radius of zero')
    (x1, y1), (x2, y2) = p1, p2
    if p1 == p2:
        raise ValueError('coincident points gives infinite number of Circles')
    # delta x, delta y between points
    dx, dy = x2 - x1, y2 - y1
    # dist between points
    q = sqrt(dx**2 + dy**2)
    if q > 2.0*r:
        raise ValueError('separation of points > diameter')
    # halfway point
    x3, y3 = (x1+x2)/2, (y1+y2)/2
    # distance along the mirror line
    d = sqrt(r**2-(q/2)**2)
    # One answer
    c1 = Cir(x = x3 - d*dy/q,
             y = y3 + d*dx/q,
             r = abs(r))
    # The other answer
    c2 = Cir(x = x3 + d*dy/q,
             y = y3 - d*dx/q,
             r = abs(r))
    return c1, c2

def getPointsForCenter(x1, x2, C, R,N):
    start_t = inv_parametric_circle(x1, C[0], R)
    end_t   = inv_parametric_circle(x2, C[0], R)
    
    arc_T = np.linspace(start_t, end_t, N)
    
    X,Y = parametric_circle(arc_T, C[0], C[1], R) if C[0] < C[1] else parametric_circle(arc_T, C[1], C[0], R)
    # Now calculate N points between (x1,y1) and (x2,y2)
    #points = np.append(np.array([X]).T, np.array([Y]).T, 1)

    return X,Y
    
def getX(Y, C, R, min_x):
    # This can be: C[0] +- sqrt(R**2 - (C[1]-Y)**2)
#    print("getX:", Y, C, R,min_x)
    if (min_x >= C[0]):
        X = C[0] + np.sqrt(R**2 - (C[1]-Y)**2)
    else:
        X = C[0] - np.sqrt(R**2 - (C[1]-Y)**2)
    return X.astype(int)

def getY(X, C, R, min_y):
#    print("getY: ", min_y, C)
    if (min_y >= C[1]):
        Y = C[1] + np.sqrt(R**2 - (C[0]-X)**2)
    else:
        Y = C[1] - np.sqrt(R**2 - (C[0]-X)**2)
    return Y.astype(int)

def pick1Center(C1, C2):
#    print("pick1Center: ", C1, C2)
    if (C1[0] < 0) or (C1[1] < 0):
        return C2
    elif (C2[0] < 0) or (C2[1] < 0):
        return C1
    else:
        return C2

def getValues(y1, y2, N):
    step = np.abs(y1-y2)/N
#    print("step =",step)
    if (y1 < y2):
        Y = np.arange(y1,y2,step)
    else:
        Y = np.flipud(np.arange(y2,y1,step))
    return Y.astype(int)

#Given a point (Px, Py), that point's starting angle (Pa), and the radius (r), 
# you can calculate the center (Cx, Cy) like so:
def getPoints(x1, y1, x2, y2, N, pickCenter=1):
    # calculate radius
    R = int(round(sqrt(((x1-x2)**2 + (y1-y2)**2)/2)))

#    print("R =", R)
    
    # Assume angle = pi/6
    Cir1, Cir2 = circles_from_p1p2r([x1,y1], [x2,y2], R) #find_center([x1,y1], [x2,y2], angle)
    
    C1 = [int(Cir1.x), int(Cir1.y)]
    C2 = [int(Cir2.x), int(Cir2.y)]
    #X,Y = getPointsForCenter(x1, x2, C1, R,N) if x1 < x2 else getPointsForCenter(x2, x1, C1, R,N)
    #plt.scatter(X,Y)
    plt.scatter([x1],[y1],color='y')
    plt.scatter([x2],[y2],color='y')

    #X,Y = getPointsForCenter(x1, x2, C2, R, N) if x1 < x2 else getPointsForCenter(x2, x1, C2, R,N)
    #plt.scatter(X,Y)

    if (pickCenter ==1):
        C = C1
    else:
        C = C2
#    print(C1,C2,C)
    # Find N points
    if (abs(y1-y2) > abs(x1-x2)):
        Y = getValues(y1, y2, N)
        X = getX(Y, C, R, min(x1,x2))
    else:
        X = getValues(x1, x2, N)
        Y = getY(X, C, R, min(y1,y2))
        
#    print("Y:", Y)


    plt.scatter([C1[0]],[C1[1]],color='r',s=100)
    plt.scatter([C2[0]],[C2[1]],color='b',s=100)

    # Now get Xs
    #print(X,Y)
    points = np.append(np.array([X]).T, np.array([Y]).T, 1)
#    print(points)
       
    plt.scatter(X,Y)
    #plt.axis('equal')
    plt.show()
    
    
    return (points)
    
import argparse
parser = argparse.ArgumentParser(description='genenerate 100 points between 2 points')
parser.add_argument('input_2pts', help='2 points coordinates (x1,y1,x2,y2')
parser.add_argument('circle_1_2', help='cicle 1 or 2')
args = parser.parse_args()
arr = args.input_2pts.split(',')

#ARG[420,582] -> US [308,319]
points = getPoints(int(arr[0]),int(arr[1]),int(arr[2]),int(arr[3]),100,int(args.circle_1_2))
#points = getPoints(308,319, 420,582, 100)
print(len(points))
print(np.array2string(points,separator=','))

