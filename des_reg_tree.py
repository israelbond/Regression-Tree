#des_reg_BSTree.py
#DEVELOPER: Israel Bond
#implementation of classes needed for assignment

class BSTNode:
    def __init__(self,number):
        self.left = None
        self.right = None
        self.num = number

class BSTree:
    def __init__(self):
        self.root = None

    def insert(self,number):
        if(self.root == None):
            self.root = BSTNode(number)
        else:
            self._insert(number, self.root)

    def _insert(self, number, BSTNode):
        if(number < BSTNode.num):
            if(BSTNode.left != None):
                self._insert(number,BSTNode.left)
            else:
                BSTNode.left = BSTNode(number)
        else:
            if(BSTNode.right != None):
                self._insert(number,BSTNode.right)
            else:
                BSTNode.right = BSTNode(number)

    def deleteBSTree(self):
        self.root = None

    def displayBSTree(self):
        if(self.root != None):
            self._displayBSTree(self.root)

    def _displayBSTree(self, node):
        if(node != None):
            self._displayBSTree(node.left)

            print(str(node.num) + ' ')
            self._displayBSTree(node.right)


