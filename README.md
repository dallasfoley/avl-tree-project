# Project 6 README
## Overview
Project 6, as part of CSE 331 FS23, is an extensive implementation of Binary Search Trees (BST) and AVL Trees in Python. The project primarily involves building and manipulating these tree data structures, with additional functionalities such as visualization and nearest-neighbor classification.

## Features
Binary Search Tree (BST): Implementation of basic BST operations including insertion, deletion, search, and traversal methods. <br />
AVL Tree: An extension of BST that maintains a balanced tree through rotations to ensure O(log n) time complexity in operations.<br />
Visualization: Functions to generate SVG images of the BST and AVL trees, aiding in understanding their structure and balance.<br />
Nearest-Neighbor Classifier: A unique implementation utilizing AVL trees for efficient lookup and classification in a one-dimensional space.
## Structure
Node: A basic node class used in both BST and AVL Tree implementations.<br />
BinarySearchTree: Contains methods for BST operations.<br />
AVLTree: Inherits from BinarySearchTree, adding AVL-specific rotations and rebalancing.<br />
AVLWrappedDictionary: A helper class for the NearestNeighborClassifier, holding a key and a dictionary for data storage.<br />
NearestNeighborClassifier: Implements a one-dimensional nearest-neighbor classifier using AVL Tree.<br />
## Usage
Initialization: Create an instance of BinarySearchTree or AVLTree.<br />
Operations: Use methods like insert, remove, search, and various traversal methods to manipulate the tree.<br />
Visualization: Call visualize method on an instance of a tree to generate an SVG image file.<br />
Classification: Use NearestNeighborClassifier for nearest-neighbor classification tasks.<br />
Installation
Ensure you have Python 3.7 or above. Clone the repository and install dependencies (if any).


# Example: Using NearestNeighborClassifier
from starter import NearestNeighborClassifier<br />

classifier = NearestNeighborClassifier(resolution=2)<br />
classifier.fit([(0.1, "Class A"), (0.2, "Class B")])<br />
print(classifier.predict(0.15, 0.05))<br />
# Dependencies
Python 3.7+
Additional libraries for visualization (optional): matplotlib
Notes
The NearestNeighborClassifier is a unique feature of this project, demonstrating an innovative use of AVL trees.
Visualization requires matplotlib if using the provided plot_time_comparison function.
#Author
Dallas Foley
