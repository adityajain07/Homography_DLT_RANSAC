# About
The aim of the project is to find homography between two image scenes using DLT along with RANSAC. Everything has been coded from scratch. The 'Collage Image' folder contains a collage of book images (skewed, rotated, translated, distorted) and some of the raw book images are present in 'Test Images' folder. The aim is to find homography between the test images and the collage image and given a test image, find its bounding box in the collage image.  <br/>

## Collage Image
![Collage Image](https://github.com/adityajain07/Homography_DLT_RANSAC/blob/master/Collage%20Image/collage_image.jpg)


# Technique
Following is the process implemented to find the bounding box of a test image in the collage image:
1. Given a test image and the collage image
1. Find SIFT features in both test and collage image
1. Find the feature matches using any matcher (used BF matcher in this project)
1. Pick any four random pairs of matches and normalise them (to have their centroid at (0,0) and average distance from the origin is sqrt(2))
1. Find homography using the above points using DLT
1. Find the reprojection error using the homography calculated and store all the inliers (inliers are those points whose reprojection error is below the threshold decided)
1. Repeat steps 4-6 for N iterations (used 4000 in the results, though 1000 is good enough too)
1. Store the maximum no. of inliers in the N iterations
1. Again normalise the inlier points in the above step
1. Find homography in the above using DLT in least-squared sense
1. Apply the homography obtained above on the corner points of the test image to get their corresponding positions in the collage image
1. Draw lines between the mapped corners to get the bounding box of the test image in the collage
