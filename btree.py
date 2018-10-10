# -*- coding: utf-8 -*-
import typing
from typing import *

"""
This is a definition of the tree structure we require to fully
parse and then compile the recipes.
"""

import collections
import functools
import json
import pprint

# Credits
__author__ = 'George Flanagin'
__copyright__ = 'Copyright 2018, University of Richmond'
__credits__ = None
__version__ = '0.5'
__license__ = 'https://www.gnu.org/licenses/gpl.html'
__maintainer__ = 'George Flanagin'
__email__ = 'gflanagin@richmond.edu'
__status__ = 'Prototype'

class BinaryTree:
    class Node:
        pass
    pass

class BinaryTree:
    """
    This binary tree is based on the example given in Chapter 6 of 
    /Data Structures and Algorithms with Python/, with the original
    code adapted from:
        http://knuth.luther.edu/~leekent/CS2Plus/_downloads/binarysearchtree.py
    collected on 9 April 2018.
    """

    class Node:
        """
        This is a Node class that is internal to the BinaryTree class. 
        """
        def __init__(self,
                val:Any,
                left:BinaryTree.Node=None,
                right:BinaryTree.Node=None,
                parent:BinaryTree.Node=None):

            self.parent = parent
            self.datum = val
            self.left = left
            self.right = right

            
        @property
        def _value(self) -> Any:
            return self.datum

        
        @property
        def _left(self) -> BinaryTree.Node:
            return self.left

        
        @property
        def _right(self) -> BinaryTree.Node:
            return self.right


        @property
        def _parent(self) -> BinaryTree.Node:
            return self.parent
        

        def set_value(self, newval:Any) -> Any:
            """
            Set the value of this node. 

            returns -- the new value of this node.
            """
            self.datum = newval

            
        def set_left(self, newleft:BinaryTree.Node) -> BinaryTree.Node:
            """
            Join this node to another node on the left.

            returns -- newleft for convenience.
            """

            self.left = newleft
            newleft.parent = self
            return self.left

            
        def set_right(self, newright:BinaryTree.Node) -> BinaryTree.Node:
            """
            Join this node to another node on the right.    
            """

            self.right = newright
            newright.parent = self
            return self.right

            
        def __dig(self) -> tuple:
            it = self
            i = 1
            while it.parent is not None:
                i += 1
                it = it.parent
            return it, i        


        def __myparents(self) -> list:

            it = self.parent
            nodes = [id(it)]
            while it != self._root:
                it = it.parent
                nodes.append(id(it))
    
            return nodes


        @property
        def _root(self) -> BinaryTree.Node:
            return self.__dig()[0]


        @property
        def _height(self) -> int:
            return self.__dig()[1]


        def __iter__(self) -> Any:
            """
            This method deserves a little explanation. It does an inorder traversal
            of the nodes of the tree yielding all the values. In this way, we get
            the values in ascending order.
            """
            if self.left is not None:
                for elem in self.left:
                    yield elem
                    
            yield self.datum
            
            if self.right is not None:
                for elem in self.right:
                    yield elem


        def __str__(self) -> str:
            return "L:{}, {}, R:{}".format(self.left, self.datum, self.right)


        def __repr__(self) -> str:
            return "BinaryTree.Node({}, {}, {})".format(
                repr(self.datum), repr(self.left), repr(self.right)
                )            
            

        def __sub__(self, other:BinaryTree.Node) -> int:
            """
            Find the distance between two nodes.

            returns -- a non-negative integer

            raises -- Exception if the nodes are in different trees.
            """
            if self == other: return 0

            my_parents = self.__parents()
            other_parents = other.__parents()
            common_node = set(my_parents) - set(other_parents)
            if not common_node: 
                raise Exception("{} and {} are in different trees.".format(self, other))
            return my_parents.index(common_node) + other_parents.index(common_node) 
                

    def __init__(self, 
            root:BinaryTree.Node=None):
        """
        Make a new tree, and establish the value of its node.
        Optionally, establish its parent.
        """
        self.root = root
        

    def insert(self, datum) -> BinaryTree:
        """
        Insert a new datum somewhere in the tree.
        """
        self.root = __insert(self.root, datum)
        return self.root

        
    def __insert(root, val) -> BinaryTree:
        """
        This /private/ method performs the comparison, and inserts 
        at the appropriate point.
        """
        if root == None:
            return BinaryTree.Node(val)
        
        if val < root.value():
            root.set_left(BinaryTree.__insert(root._left, val))
        else:
            root.set_right(BinaryTree.__insert(root._right, val))
            
        return root


    def __eq__(self, other:BinaryTree) -> bool:
        """
        If you have been traversing a tree, it is possible to get
        lost. This function compares the IDs of this sub-tree
        with the other sub-tree.
        """

        return id(self) == id(other)


    def __len__(self) -> int:
        """
        Quite simply, this is the number of nodes.
        """
        i = 1
        while (next(self)): i += 1
        return i

        
    def __iter__(self) -> BinaryTree.Node:
        """
        The public iterator invoked by calling next().
        """
        return iter(self.root) if self.root is not None else iter([])


    def __str__(self):
        return "BinarySearchTree({})".format(self.root)
 

def main():
    s = input("Enter a list of numbers: ")
    lst = s.split()
    
    tree = BinaryTree()
    
    for x in lst:
        tree.insert(float(x))
        
    for x in tree:
        print(x)

if __name__ == "__main__":
    main()


