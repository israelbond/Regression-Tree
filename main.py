# main.py
# DEVELOPER:  Israel Bond
# project: Regression Tree for advanced ML
# adv. ML with Anthony Rhodes

# this is the main.py which runs my regression tree algorithm. I have gotten to a point where pruning
# would be the next step however i have spent as much time as i could and this is what if produces:
    # split data set to a 75% training, 25% testing sets
    # built full regression tree with test set to a particular depth
    # feed testing set into regression tree for evaluation

import pandas as pd
from sklearn.model_selection import train_test_split
from dict_bst import BSTree

def main():
    print("here we go!...it's Mario?!?")
    #grab csv file with pandas objects
    df = pd.read_csv('Carseats.csv')
    #split the data
    training, testing = train_test_split(df,test_size=0.25)
    tree = BSTree(0)
    tree.train(training)
    # uncomment this line to see the trained regression tree
#    tree.displayBSTree()
    # establishes test sets for each level of the trained tree
    tree.test(testing)

    #split file with numpy objects
        # XXX this method adds some variance to each set
#    data = np.random.rand(len(df)) <= 0.75
#    train = df[data]
#    test = df[~data]
#    print(train)
#    print(test)


#    tree = BSTree()
#    print(tree)
#    print(tree.keys[6])
#    print(tree.values[6][2])

if __name__ == "__main__" :
    main()
