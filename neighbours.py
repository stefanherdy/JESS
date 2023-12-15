#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

'''
Script Name: neighbours.py  
Author: Stefan Herdy  
Date: 15.06.2023  
Description:   
This script performs a neighborhood analysis to get a deeper understanding of 
neighbourhood interactions between the investigated classes. 
'''

def main():
    if args.set == 'usa':
        class_labels = {1: r"$\it{Toninia}$", 
                        2: r"$\it{Psora}$", 
                        3: r"$\it{Fulgensia}$", 
                        4: r"$\it{Placidium}$", 
                        5: r"$\it{Syntrichia}$", 
                        6: r"$\it{Collema}$", 
                        7: 'Cyano'}
        
        # Define the path to your image annotations folder
        annotations_folder = ['input_data/mask_usa']

    if args.set == 'john':
        class_labels = {0: r"$\it{Tortella}$" + " " + r"$\it{t.}$",
                        1: r"$\it{Fissidens}$",
                        2: r"$\it{Ctenidium}$", 
                        3: r"$\it{Dicranaceae}$", 
                        4: r"$\it{Schistidium}$", 
                        5: r"$\it{Rhytidiadelphus}$", 
                        6: r"$\it{Tortella}$" + " " + r"$\it{d.}$"}
        
        # Define the path to your image annotations folder
        annotations_folder = ['input_data/mask_john_handy', 'input_data/mask_john_cam']

    # Define categorical class values
    #class_values = np.arange(len(class_labels))
    class_values = class_labels.keys()
    # Create a dictionary to store class neighbor counts
    neighbor_counts = {i: {j: 0 for j in class_values} for i in class_values}

    # Find neighbors for a given pixel with pixel indices x, y
    def find_neighbors(image, x, y):
        neighbors = []
        height, width, _ = image.shape
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < height and 0 <= new_y < width:
                    if image[x, y, 0] != image[new_x, new_y, 0]:
                        neighbors.append(image[new_x, new_y, 0])
        return neighbors

    # Process each image in the folder
    full_paths = [os.path.join(folder, filename) for folder in annotations_folder for filename in os.listdir(folder)]
    for filename in full_paths:
        print(filename)
        if filename.endswith('.png'):
            # Load the image
            image = cv2.imread(filename)
            image = cv2.resize(image, (int(image.shape[0]/2), int(image.shape[1]/2)), interpolation = cv2.INTER_NEAREST)
            # Iterate through all pixels in the image
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    pixel_value = image[x, y, 0]

                    # Find neighbors
                    neighbors = find_neighbors(image, x, y)

                    # Update neighbor counts
                    for neighbor_value in neighbors:
                        for key in class_labels:
                            if neighbor_value == key:
                                for subkey in class_labels:
                                    if pixel_value == subkey:
                                        neighbor_counts[pixel_value][neighbor_value] += 1

    # Calculate the percentage of neighbors for each class pair
    percentage_matrix = np.zeros((len(class_values), len(class_values)))

    for i, class_i in enumerate(class_values):
        for j, class_j in enumerate(class_values):
            total_neighbors = sum(neighbor_counts[class_i].values())
            if total_neighbors != 0:
                percentage_matrix[i, j] = (neighbor_counts[class_i][class_j] / total_neighbors) * 100

    # Create a matrix plot
    plt.imshow(percentage_matrix, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(class_values)), [class_labels[i] for i in class_labels.keys()], rotation=45)
    plt.yticks(np.arange(len(class_values)), [class_labels[i] for i in class_labels.keys()], )
    plt.xlabel(f'Neighbor Class of data set {args.set}')
    plt.ylabel('Class')
    plt.title('Percentage of Neighbors between Taxa')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("JESS Neighbors")
    parser.add_argument("--set", choices=['usa', 'john'], default='john', help="Dataset for neighbor calculation")
    args = parser.parse_args()
    print(args.set)
    main()
