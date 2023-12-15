#!/usr/bin/env python3

from skimage import measure
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Script Name: curvature.py
Author: Stefan Herdy
Date: 14.11.2023
Description: 
This script performs a curvature analysis on binary masks
Usage: 
- Change the the path to the path to your binary mask and execute the script.
- Additionally you can set the minimum contour length, the window size ratio and the min/max values of the colormap according to your specific requirements
"""

def compute_curvature(point, i, contour, window_size):
    # Compute the curvature using polynomial fitting in a local and rotated coordinate system
    # Extract neighboring edge oints
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the oints to be able to compute the curvature independend of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)

    # We compute the 2nd derivative in order to determine wether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

def compute_curvature_profile(mask, min_contour_length, window_size_ratio):
    # Compute the contours of the mask to be able to analyze each part individually
    contours = measure.find_contours(mask, 0.5)

    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    edge_pixels = []

    # Iterate over each contour
    for contour in contours:
        # Iterate over each point in the contour
        for i, point in enumerate(contour):
            if contour.shape[0] > min_contour_length:
                # Compute the curvature for the point
                # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
                window_size = int(contour.shape[0]/window_size_ratio)
                curvature = compute_curvature(point, i, contour, window_size)
                # We compute, whether a point is convex or concave.
                # If you want to have the 2nd derivative shown you can comment this part
                if curvature > 0:
                    curvature = 1
                if curvature <= 0:
                    curvature = -1
                # Store curvature information and corresponding edge pixel
                curvature_values.append(curvature)
                edge_pixels.append(point)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    return edge_pixels, curvature_values

def plot_edges_with_curvature(mask, min_contour_length, window_size_ratio):
    # Compute edge properties
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)

    # Plot the mask
    plt.imshow(mask, cmap='gray')
    # We set the min and max of the colorbar, so that 90% of the curvature values are shown.
    # This is to have a nice visualization. You can change this threshold according to your specific task.
    threshold = np.percentile(np.abs(curvature_values), 90)
    plt.scatter(edge_pixels[:, 1], edge_pixels[:, 0], c=curvature_values, cmap='cool', s=5, vmin=-threshold, vmax=threshold)

    plt.colorbar(label='Curvature')
    plt.title("Curvature of Binary Mask")
    plt.show()

# Set the path to the mask you want to analyze and you are ready to start
mask = cv2.imread('path_to_your_mask', cv2.IMREAD_GRAYSCALE)

# Set minimum length of the contours that should be analyzed
min_contour_length = 20
# Set the ratio of the window size (contour length / window_size_ratio) for local polynomial approximation
window_size_ratio = 5

# Visualize curvature on the binary mask
plot_edges_with_curvature(mask, min_contour_length, window_size_ratio)