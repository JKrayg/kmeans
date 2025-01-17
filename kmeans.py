import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import load_iris
import numpy as np
import math
import random

k = 3
numPoints = 1200
data = []
centroids = []
points = []
dimensions = 2
dataCount = 1

# iris = load_iris()
# data = np.array(iris["data"])

# create random test data
# while dataCount <= numPoints:
#     rand = []
#     for i in range(dimensions):
#         rand.append(np.random.randint(1, numPoints))

#     if not any(np.array_equal(rand, c) for c in data):
#         data.append(rand)
#         dataCount += 1


# tight cluster data (3D, k = 3)
# data = [[5.29, 6.25, 4.45], [5.18, 5.14, 4.90], [5.80, 3.55, 5.32],
#         [5.17, 5.38, 3.54], [5.64, 5.16, 6.03], [4.49, 5.58, 4.68],
#         [5.35, 4.41, 5.53], [4.81, 4.21, 5.82], [6.40, 6.25, 6.70],
#         [5.53, 4.83, 3.82], [6.24, 5.59, 4.30], [4.08, 5.71, 5.37],
#         [4.44, 5.47, 6.45], [6.14, 4.07, 4.37], [3.53, 4.95, 4.58],
#         [5.07, 5.67, 5.30], [24.64, 25.74, 25.66], [25.55, 25.14, 24.01],
#         [23.46, 23.91, 26.11], [26.08, 24.20, 25.92], [24.82, 23.97, 25.61],
#         [24.54, 25.88, 25.91], [26.53, 24.98, 26.00], [24.67, 24.22, 24.75],
#         [25.73, 25.71, 26.56], [27.48, 23.99, 26.12], [23.15, 25.30, 26.81],
#         [23.35, 26.11, 24.50], [25.61, 24.97, 25.27], [26.25, 24.94, 25.46],
#         [25.65, 25.12, 24.22], [24.53, 23.88, 25.68], [5.02, 25.43, 15.79],
#         [5.89, 25.58, 15.27], [4.19, 24.31, 15.40], [4.63, 27.80, 13.90],
#         [6.16, 24.90, 14.50], [4.39, 24.36, 16.39], [5.56, 24.71, 15.23],
#         [4.37, 24.73, 13.50], [5.47, 25.65, 15.85], [4.68, 26.07, 15.31],
#         [6.23, 25.94, 13.67], [4.20, 26.03, 16.29], [4.70, 24.44, 15.81],
#         [4.57, 25.75, 16.31], [4.16, 24.57, 15.22], [4.29, 24.17, 14.26]]

# tight cluster data (2D, k = 3)
# **** multiple centroids get stuck on a single cluster sometimes
data = [[5.29, 6.25], [5.18, 5.14], [5.80, 3.55],
        [5.17, 5.38], [5.64, 5.16], [4.49, 5.58],
        [5.35, 4.41], [4.81, 4.21], [6.40, 6.25],
        [5.53, 4.83], [6.24, 5.59], [4.08, 5.71],
        [4.44, 5.47], [6.14, 4.07], [3.53, 4.95],
        [5.07, 5.67], [24.64, 25.74], [25.55, 25.14],
        [23.46, 23.91], [26.08, 24.20], [24.82, 23.97],
        [24.54, 25.88], [26.53, 24.98], [24.67, 24.22],
        [25.73, 25.71], [27.48, 23.99], [23.15, 25.30],
        [23.35, 26.11], [25.61, 24.97], [26.25, 24.94],
        [25.65, 25.12], [24.53, 23.88], [5.02, 25.43],
        [5.89, 25.58], [4.19, 24.31], [4.63, 27.80],
        [6.16, 24.90], [4.39, 24.36], [5.56, 24.71],
        [4.37, 24.73], [5.47, 25.65], [4.68, 26.07],
        [6.23, 25.94], [4.20, 26.03], [4.70, 24.44],
        [4.57, 25.75], [4.16, 24.57], [4.29, 24.17]]


def scale(d):
    return np.array([i / max(d.flatten()) for i in [j for j in d]])

data = scale(np.array(data))


# euclidean distance
def euc(centroid, point):
    dis = 0
    for i in range(len(centroid)):
        dis += (point[i] - centroid[i])**2
    return math.sqrt(dis)

# initialize centroids
def initCentroids(k, d):
    centroids.append(d[random.randint(0, len(d) - 1)])

    for i in range(1, k):
        dists = np.array([min(euc(point, centroid)**2 for centroid in centroids) for point in d])
        probs = dists / dists.sum()
        cumulProbs = np.cumsum(probs)

        rand = random.random()
        for id, c in enumerate(cumulProbs):
            if rand < c:
                centroids.append(d[id])
                break
    return centroids

# assign points to nearest centroid
def assignPoints(d, c):
    pnts = []
    for i in d:
        temp = []
        for j in c:
            d = euc(j, i)
            temp.append(((j, i), d if d != 0.0 else 1.0))
        
        m = min(list(enumerate(temp)), key = lambda x: x[1][1])
        pnts.append((m[1][0] , m[0]))
    return pnts

# get clusters
def groupClusters(c, p):
    clst = []
    for center in c:
        grp = []
        for pnt in p:
            if tuple(pnt[0][1]) not in [tuple(ctr) for ctr in c]:
                if tuple(center) == tuple(pnt[0][0]):
                    grp.append(pnt[0][1])
        clst.append((center, grp))
    return clst


def updateCentroid(c):
    cents = []
    for i in c:
        currC = []
        for k in range(len(i[0])):
            vals = [x[k] for x in i[1]]
            vals.append(i[0][0])
            currC.append(sum(vals) / len(vals))
        cents.append(currC)
    
    
    return cents

# check distance change between previous centroids and current centroids
def change(p, c):
    d = 0
    for p, c in zip(p, c):
        d += sum(abs(p[i] - c[i]) for i in range(len(c)))
    
    if d > 10**-4:
        return True
    
    return False

def plot():
    colors = []
    for b in points:
        colors.append(b[1])

    if dimensions == 3:
        data3d = np.multiply(data, 125)
        centroids3d = np.multiply(centroids, 125)
        plt.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2], c=colors, alpha = 0.5, edgecolors='black')
        for p in centroids3d:
            plt.scatter(p[0], p[1], p[2], c='red', marker='x')

        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.title('update ' + str(count))

        fig = go.Figure()

        # Add data points to the scatter plot
        fig.add_trace(go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,  # Use colors array to color the points
                opacity=0.5,
                line=dict(color='black', width=0.5)
            ),
            name='Data points'
        ))

        # Add centroids to the scatter plot
        fig.add_trace(go.Scatter3d(
            x=[c[0] for c in centroids],
            y=[c[1] for c in centroids],
            z=[c[2] for c in centroids],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='x'
            ),
            name='Centroids'
        ))

        # Update layout to add axis labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature1',
                yaxis_title='Feature2',
                zaxis_title='Feature3'
            ),
            title='3D Scatter Plot with Centroids',
            showlegend=True
        )

        # Show the plots
        fig.show()
        plt.show()

    else:
        plt.scatter(data[:, 0], data[:, 1], c=colors, alpha = 0.5, edgecolors='black')
        for p in centroids:
            plt.scatter(p[0], p[1], c='red', marker='x')

        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.title('update ' + str(count))

        plt.show()




print("Starting centroids", initCentroids(k, data), "\n\n")
points = assignPoints(data, centroids)
count = 0
ce = np.add(centroids, 1)
plot()

while change(ce, centroids):
    count += 1
    ce = centroids
    centroids = updateCentroid(groupClusters(centroids, points))
    points = assignPoints(data, centroids)

    # plot()
    

plot()

print("Ending centroids", centroids)
