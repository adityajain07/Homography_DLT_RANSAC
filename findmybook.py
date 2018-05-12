
# coding: utf-8

# In[1]:


### Author: Aditya Jain #####
### Topic: Find My Book: Homography Calculation ###
### Start Date: 22nd March, 2018 ###

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from __future__ import division

test = cv2.imread('Q1_data/test3.jpg')
collage = cv2.imread('Q1_data/collage_image.jpg')

# Converting to gray scale
test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
collage = cv2.cvtColor(collage,cv2.COLOR_BGR2GRAY)

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Applying SIFT feature detector
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(test,None)   # kp are the keypoints, des are the descriptors
kp2, des2 = sift.detectAndCompute(collage,None)

# drawing the keypoints with the drawing the size of keypoints
draw1 = cv2.drawKeypoints(test,kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
draw2 = cv2.drawKeypoints(collage,kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints1.jpg',draw1)
cv2.imwrite('sift_keypoints2.jpg',draw2)


# In[2]:


# Plotting the matches

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together    
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
#         
        cv2.circle(out, (int(x1),int(y1)),4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    
#         cv2.imwrite('Matched_Keypoints_test3.jpg',out)


# Draw first 200 matches.
out = drawMatches(test, kp1, collage, kp2, matches[:100])


# In[3]:


######## Applying RANSAC here i.e. in the subsequent sections #########
m = 200   # Taking m top matches
matches = matches[:m]


# Returns 4 random pairs from the matches, in pixels 
def fourRandomPairs(matches, m):
    index = np.random.randint(0,m,4)
#     print index
    
    # If there are any duplicate entries, then again call the function
    if len(set(index)) !=4:
        fourRandomPairs(matches, m) 
     
    list1 = []
    list2 = []
    
    for i in index:
#         print i        
        mat = matches[i:i+1][0]       
        
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        
        # x - columns
        # y - rows
        p = kp1[img1_idx].pt
        q = kp2[img2_idx].pt
        
        list1.append(p)
        list2.append(q)
        
    return list1, list2       
   

TestList, CollageList = fourRandomPairs(matches, m)
# Converting to numpy arrays for further calculation
TestList = np.array(TestList)
CollageList = np.array(CollageList)


# In[4]:


## Normalisation of the pixel coordinates
# Below returns a similarity transform to normalise the data; supply a numpy array
def normaliseT(listn):   
    
    xall = listn[:,0]
    yall = listn[:,1]
    
    # Average of the points
    n = len(xall)  # no of datapoints
    xavg = np.sum(xall)/n
    yavg = np.sum(yall)/n
    
    summ = 0
    for i in range(n):
        temp = sqrt((xall[i] - xavg)**2 + (yall[i] - yavg)**2)        
        summ += temp    
    
    avgsum = summ/n   
    
    # The transformation matrix parameters
    s = sqrt(2)/avgsum  # scaling factor
    tx = -s*xavg        # translation in x direction
    ty = -s*yavg        # translation in y direction
    
    return np.array([[s,0,tx], [0,s,ty], [0,0,1]])   
     

# Get the scaled coordinates for DLT
Ttest = normaliseT(TestList)           # scaling matrix for test image points
Tcollage = normaliseT(CollageList)     # scaling matrix for collage image points

def normalisePoints(T, points):
    # converting to points to homogenous form and stacking them together
    ones = np.ones(len(points))
    points = np.transpose(points)  
    # Normalising them
    pointsHomN = np.matmul(T, np.vstack((points, ones)))
#     print pointsHomN
    return pointsHomN
    
TestNormal = normalisePoints(Ttest, TestList)           # Normalised coordinates of test image in homogenous form
CollageNormal = normalisePoints(Tcollage, CollageList)  # Normalised coordinates of collage image in homogenous form


# In[5]:


## DLT: Taking SVD to get H matrix

# This function stacks the matrix A, for which SVD will be taken later
def stackA(test, collage):
    size = np.shape(collage)
    lensize = size[1]    # the no of correspondences
    
    matrixA = np.array([])
    for i in range(lensize):
        tempTest = test[:,i]
        tempCollage = collage[:,i]
        x = tempTest[0]
        y = tempTest[1]
        u = tempCollage[0]
        v = tempCollage[1]
        
        tempA = np.array([[-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, y]])
        
        if i==0:
            matrixA = tempA
        else:
            matrixA = np.vstack((matrixA, tempA))        
    return matrixA
    
matrixforSVD = stackA(TestNormal, CollageNormal)  # matrix for which SVD has to be taken

# Taking SVD now
U, S, V = np.linalg.svd(matrixforSVD)
Vtranspose = np.transpose(V)    # because in SVD, it is USV'

h = Vtranspose[:,-1]     # homograph matrix in 1x9 form
HMatrix = np.array([h[:3], h[3:6], h[6:9]])
HmatrixFinal = np.matmul(np.linalg.inv(Tcollage), np.matmul(HMatrix, Ttest))   # Removing the scaling from the H matrix
print h, HMatrix, HmatrixFinal


# In[6]:


## Applying RANSAC to calculate and return inliers

def RANSAC(HmatrixF, matches):
    
    list1 = [] # Test Image Points
    list2 = [] # Collage Image Points
    # Getting all the points in (x,y) format
    for mat in matches:        
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        
        # x - columns
        # y - rows
        p = kp1[img1_idx].pt
        q = kp2[img2_idx].pt
        
        list1.append(p)
        list2.append(q)
        
    list1 = np.array(list1)    
    list2 = np.array(list2)
    ones = np.ones(len(list1))    
    TestPoints = np.transpose(list1)  
    # Homogenous form
    TestPointsHom = np.vstack((TestPoints, ones))    
    
    projectedPointsH = np.matmul(HmatrixF, TestPointsHom)  # projecting the points in test image to collage image using homography matrix
    
    projectedPointsNH = np.transpose(np.array([np.true_divide(projectedPointsH[0,:], projectedPointsH[2,:]), np.true_divide(projectedPointsH[1,:], projectedPointsH[2,:])]))
#     print projectedPointsNH
#     print list2
    
    errorList = []
    inlierMatchList = []  # This list contains the list of points with the reprojection error less than the threshold
    outlierMatchList = []  # This list contains the list of points with the reprojection error less than the threshold
    count = 0
    for i in range(len(list2)):
        error = np.linalg.norm(projectedPointsNH[i]-list2[i])
        if error < 2:
            count += 1
            inlierMatchList.append([list1[i], list2[i]])
        else:
            outlierMatchList.append([list1[i], list2[i]])
            
        errorList.append(error)        
    
#     print count
    return count, inlierMatchList, outlierMatchList
        

# RANSAC(HmatrixFinal, matches)
 


# In[7]:


# This function runs RANSAC for N interations and returns the max no of inliers
def maxInliers(N):
    
    maxCount = -1  # count for keeping the max number of inliers
    inlierCorrespondences = []  # this contain the pair of the points which are considered inlier match
    outlierCorrespondences = []  # this contain the pair of the points which are considered outlier match
    for i in range(N):
        TestList, CollageList = fourRandomPairs(matches, m)    # generating 4 random pairs to calculate homography
        # Converting to numpy arrays for further calculation
        TestList = np.array(TestList)
        CollageList = np.array(CollageList) 
        
        # Get the scaled coordinates for DLT
        Ttest = normaliseT(TestList)           # scaling matrix for test image points
        Tcollage = normaliseT(CollageList)     # scaling matrix for collage image points        
        
        TestNormal = normalisePoints(Ttest, TestList)           # Normalised coordinates of test image in homogenous form
        CollageNormal = normalisePoints(Tcollage, CollageList)  # Normalised coordinates of collage image in homogenous form
        
        matrixforSVD = stackA(TestNormal, CollageNormal)  # matrix for which SVD has to be taken

        # Taking SVD now
        U, S, V = np.linalg.svd(matrixforSVD)
        Vtranspose = np.transpose(V)    # because in SVD, it is USV'

        h = Vtranspose[:,-1]     # homograph matrix in 1x9 form
        HMatrix = np.array([h[:3], h[3:6], h[6:9]])
        HmatrixFinal = np.matmul(np.linalg.inv(Tcollage), np.matmul(HMatrix, Ttest))   # Removing the scaling from the H matrix
        
        count, matchInlier, matchOutlier = RANSAC(HmatrixFinal, matches)
#         print count
        if count > maxCount:
            maxCount = count
            inlierCorrespondences = matchInlier
            outlierCorrespondences = matchOutlier
            
            
    print maxCount, np.shape(inlierCorrespondences), np.shape(outlierCorrespondences)
    return inlierCorrespondences, outlierCorrespondences
        
        
inliers, outliers = maxInliers(2000)


# In[8]:


## Plotting inliers and outliers
# print outliers[0]

def plotInlierOutlier(img1, img2, inliers, outliers):
    
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])   
    
    for i in outliers:
        
        x1 = i[0][0]        
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        
#         cv2.circle(out, (int(x1),int(y1)),4, (255, 0, 0), 1)   
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)
    
#         cv2.imwrite('Inliers_Outliers_test3.jpg',out)
        
    for i in inliers:        
    
        x1 = i[0][0]        
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        
#         cv2.circle(out, (int(x1),int(y1)),4, (255, 0, 0), 1)   
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
    
#         cv2.imwrite('Inliers_Outliers_test3.jpg',out)
    
plotInlierOutlier(test, collage, inliers, outliers)   


# In[9]:


## Recalculating homography using the inliers now

# This function returns the homography matrix based on the inliers using least squares method
def homographInliersLS(inliers):
    
    test = []
    collage = []
    for i in inliers:
        test.append(i[0])
        collage.append(i[1])   
    
    # Converting to numpy arrays
    test = np.array(test)
    collage = np.array(collage)    
    
    # Get the scaled coordinates for DLT
    Ttest = normaliseT(test)           # scaling matrix for test image points
    Tcollage = normaliseT(collage)     # scaling matrix for collage image points
    
    # Normalised coordinates
    TestNormal = normalisePoints(Ttest, test)           # Normalised coordinates of test image in homogenous form
    CollageNormal = normalisePoints(Tcollage, collage)  # Normalised coordinates of collage image in homogenous form
    
    matrixforSVD = stackA(TestNormal, CollageNormal)  # matrix for which SVD has to be taken
    print np.shape(matrixforSVD)

    # Taking SVD now
    U, S, V = np.linalg.svd(matrixforSVD)
    Vtranspose = np.transpose(V)    # because in SVD, it is USV'

    h = Vtranspose[:,-1]     # homograph matrix in 1x9 form
    HMatrix = np.array([h[:3], h[3:6], h[6:9]])
    HmatrixFinal = np.matmul(np.linalg.inv(Tcollage), np.matmul(HMatrix, Ttest))   # Removing the scaling from the H matrix
    return HmatrixFinal
    

HomographyLS = homographInliersLS(inliers)  # Homography matrix obtained by least-squares


# In[10]:


## Calculating the corners of the image in the collage using the final homography matrix

rows = test.shape[0]
cols = test.shape[1]

CollageImageRGB = cv2.imread('Q1_data/collage_image.jpg')

def boundingBox(TestImage, CollageRGB, Homog):
    
    rows = TestImage.shape[0]
    cols = TestImage.shape[1]
    print rows, cols
    
    # test image corners in homogenous form
    box = np.array([[0,0,1], [cols-1,0,1], [0,rows-1,1], [cols-1,rows-1,1]])
    
    # corner points in collage after applying homogrpahy
    collageCornerHom = np.matmul(Homog, np.transpose(box))
    print collageCornerHom
    
    # corner points in collage in pixel form
    collageCornerNH = np.transpose(np.array([np.true_divide(collageCornerHom[0,:], collageCornerHom[2,:]), np.true_divide(collageCornerHom[1,:], collageCornerHom[2,:])]))
    points = collageCornerNH
    print points
    
    cv2.line(CollageRGB, (int(points[0][0]),int(points[0][1])), (int(points[1][0]),int(points[1][1])), (0, 0, 255), 10)
    cv2.line(CollageRGB, (int(points[0][0]),int(points[0][1])), (int(points[2][0]),int(points[2][1])), (0, 0, 255), 10)
    cv2.line(CollageRGB, (int(points[1][0]),int(points[1][1])), (int(points[3][0]),int(points[3][1])), (0, 0, 255), 10)
    cv2.line(CollageRGB, (int(points[2][0]),int(points[2][1])), (int(points[3][0]),int(points[3][1])), (0, 0, 255), 10)
        
    cv2.imwrite('BoundingBoxBook.jpg',CollageRGB)   


boundingBox(test, CollageImageRGB, HomographyLS)


# In[11]:


print HomographyLS


# In[12]:


print np.shape(inliers)

