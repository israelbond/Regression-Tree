# dict_bst.py
# REGRESSION TREE
# Israel Bond
# adv. ML with Anthony Rhodes

# i attempted to implement a regression tree where each BSTree object contains enough variables to
# establish a good splitting point based on the RSS value of the left & right sets.
# the fact that i derived my class from a dict I thought I would get some really interesting functionality
# but i was unable to see how that would work...
# I have gotten to a point where my regression tree keeps creating nodes(BSTree) objects until I reach a MAX
# for calling recursive functions. Additionally I am unable to set an appropriate stopping condition so set up
# a max depth so that i could evaluate the trained tree
import numpy as np
import pandas as pd
import statistics as stats

maxDepth = 5 # value for stopping conditional in recursive training process
threshold = .2 # used to evaluate when i reach a reduced RSS value
class BSTree(dict):

    def __init__(self, depth, keys = ['Sales',
                                    'CompPrice',
                                    'Income',
                                    'Advertising',
                                    'Population',
                                    'Price',
                                    'ShelveLoc',
                                    'Age',
                                    'Education',
                                    'Urban',
                                    'US'],
                 values = [0.0, # Sales
                           0, # CompPrice
                           0, # Income
                           0, # Advertising
                           0, # Population
                           0, # Price
                           {'Bad': 0,'Good' : 0, 'Medium': 0}, # ShelveLoc
                           0, # Age
                           0, # Education
                           0, # {'No': 0,'Yes': 0}, #Urban
                           0]): # {'No':0,'Yes':0}]): #US

        self.keys = keys
#        self.predictorI = 0  #establishes index for the predictor used at a particular lvl in the tree
        self.values = values # List of values and types
        self.myDepth = depth + 1 # used as stopping condition

        self.initSet = None  # for parent set of data
        self.salesAvg = 0   # for calculated mean squared error
        self.initRSS = 2000.0  # to hold the lowest RSS value discovered
        self.feature = None     # used for the test set and display
        self.featureValue = 0   # used for the test sst and display

        self.leftSet = pd.DataFrame(columns=keys)  # for the left sub set
        self.leftRSS = 0
        self.leftTree = None     # sub child

        self.rightSet = pd.DataFrame(columns=keys)  # for the right sub set
        self.rightRSS = 0
        self.rightTree = None    # sub child

#        self.rssRate = None      #for mean values array

        np.set_printoptions(threshold= np.nan) # set display options for numpy objects

    def displayBSTree(self):
        if(self.leftTree != None):
            self.leftTree.displayBSTree()

        print(str(self.feature) + ' ', self.featureValue)
        if(self.rightTree != None):
            self.rightTree.displayBSTree()

   #attempting to take in the test set for evaluation but didnt complete
    def test(self, set):
        self.initSet = set
    # determine evaluation process
    # produce the sets to travers the tree
    # if their are children nodes pass each set respectively
        print(self.feature, "    ", self.featureValue)
        if self.feature == "Urban" or self.feature == "US":
            leftsplit = self.initSet.loc[self.initSet[self.feature] == "Yes"]
            rightsplit = self.initSet.loc[self.initSet[self.feature] == "No"]
            if self.leftTree != None and self.rightTree != None:
                self.leftTree.test(leftsplit)
                self.rightTree.test(rightsplit)
            else:
                print(leftsplit.shape, "   ", rightsplit.shape)
        elif self.feature == "ShelveLoc":
            leftsplit = self.initSet.loc[self.initSet[self.feature] == self.featureValue]
            rightsplit = self.initSet.loc[self.initSet[self.feature] != self.featureValue]
            if self.leftTree != None and self.rightTree != None:
                self.leftTree.test(leftsplit)
                self.rightTree.test(rightsplit)
            else:
                print(leftsplit.shape, "   ", rightsplit.shape)
        else:
            leftsplit = self.initSet.loc[self.initSet[self.feature] < self.featureValue]
            rightsplit = self.initSet.loc[self.initSet[self.feature] >= self.featureValue]
            if self.leftTree != None and self.rightTree != None:
                self.leftTree.test(leftsplit)
                self.rightTree.test(rightsplit)
            else:
                print(leftsplit.shape, "   ", rightsplit.shape)

    def calcRSS(self, yValues):
#        rss = 0.0
        sumRSS = 0.0
        for y in yValues['Sales']:
            rss = (y - self.salesAvg)**2
            sumRSS += rss
        return sumRSS


    def evaluateSets(self, key, arr):
#        print(type(arr))
#        print(key)
        if key == "ShelveLoc":
    # get three sets for 'Bad' 'Good' 'Medium'
            setA = self.initSet.loc[self.initSet[key] == "Bad"]
            setB = self.initSet.loc[self.initSet[key] == "Good"]
            setC = self.initSet.loc[self.initSet[key] == "Medium"]
    # evaluate EACH compination of sets
            setAB = setA.append(setB)
            setAC = setA.append(setC)
            setBC = setB.append(setC)
    # calculate RSS for each split set
            #A vs BC
            aRss = self.calcRSS(setA)
            bcRss = self.calcRSS(setBC)
            #B vs AC
            bRss = self.calcRSS(setB)
            acRss = self.calcRSS(setAC)
            #C vs AB
            cRss = self.calcRSS(setC)
            abRss = self.calcRSS(setAB)
    # determine if any is best split
        # the conditional for splitting will be if equal to the featureValue go left!
            if self.initRSS > (aRss + bcRss):
                self.initRSS = (aRss + bcRss)
                self.leftSet = setA
                self.rightSet = setBC
                self.feature = key
                self.featureValue = "Bad"
            if self.initRSS > (bRss + acRss):
                self.initRSS = (bRss + acRss)
                self.leftSet = setB
                self.rightSet = setAC
                self.feature = key
                self.featureValue = "Good"
            if self.initRSS > (cRss + abRss):
                self.initRSS = (cRss + abRss)
                self.leftSet = setC
                self.rightSet = setAB
                self.feature = key
                self.featureValue = "Medium"
#            print("ShelveLoc not handled")
            return
        elif key == "Urban":
            leftsplit = self.initSet.loc[self.initSet[key] == "Yes"]
            rightsplit = self.initSet.loc[self.initSet[key] == "No"]
            self.leftRSS = self.calcRSS(leftsplit)
            self.rightRSS = self.calcRSS(rightsplit)
            if self.initRSS > (self.leftRSS + self.rightRSS):
                self.initRSS = (self.leftRSS + self.rightRSS)
                self.leftSet = leftsplit
                self.rightSet = rightsplit
                self.feature = key
                self.featureValue = "Yes"

#            print("Urban not handled yet")
            return
        elif key == "US":
            leftsplit = self.initSet.loc[self.initSet[key] == "Yes"]
            rightsplit = self.initSet.loc[self.initSet[key] == "No"]
            self.leftRSS = self.calcRSS(leftsplit)
            self.rightRSS = self.calcRSS(rightsplit)
            if self.initRSS > (self.leftRSS + self.rightRSS):
                self.initRSS = (self.leftRSS + self.rightRSS)
                self.leftSet = leftsplit
                self.rightSet = rightsplit
                self.feature = key
                self.featureValue = "Yes"

#            print("US not handled yet")
            return
        elif len(arr) != 0:
            for i in np.nditer(arr):
                leftsplit = self.initSet.loc[self.initSet[key] < i]
                rightsplit = self.initSet.loc[self.initSet[key] >= i]
    #calculate RSS's for each set
                self.leftRSS = self.calcRSS(leftsplit)
                self.rightRSS = self.calcRSS(rightsplit)
    #evaluate if the current sets provide us with the minimal RSS
    #1) store the feature & feature value AND set up sets for each Sub Tree
                if self.initRSS > (self.leftRSS + self.rightRSS):
        #            print("split is less than RSS!!! ")
                    self.initRSS = (self.leftRSS + self.rightRSS)
                    self.leftSet = leftsplit
                    self.rightSet = rightsplit
#                    print(leftsplit.shape)
#                    print(rightsplit.shape)
                    self.feature = key
                    self.featureValue = i

                elif self.initRSS == 2000.0:
#                    print("SETTING DEFAULT INITRSS ")
                    self.initRSS = (self.leftRSS + self.rightRSS)
                    self.leftSet = leftsplit
                    self.rightSet = rightsplit
                    self.feature = key
                    self.featureValue = i

            return

    def train(self, set):
        self.initSet = set
    # calculate the mean of the sales average for the set
        self.salesAvg = self.initSet['Sales'].mean()
    # evaluate each feature for possible RSS
        for key in self.keys:
    # get the distinct values from the feature
            arr = self.initSet[key].unique()
    # evaluate each feature & set of distict values
            self.evaluateSets(key, arr)
    # evaluate each mean of each feature for possible RSS
    # evaluate if stopping condition is met
        #had issues getting this to give me a finite stopping contitioin without setting it to a greater value
        if self.initRSS < threshold:
            return
    # Using a maxDepth argument to stop traversing
        elif self.myDepth == maxDepth:
            return
    #GO CHILDREN!!
        if self.leftTree == None:
            self.leftTree = BSTree(self.myDepth)
            self.leftTree.train(self.leftSet)

        if self.rightTree == None:
            self.rightTree = BSTree(self.myDepth)
            self.rightTree.train(self.rightSet)
    #done with training!!!
        return

#Couldn't make this work and mostly just experiments... implemented the top portions of code to resolve issues i was having here
    def setMse(self, set):
        self.initSet = set
        print(set)
    #get count for evaluating mean values
        count = len(set)
        mean = None
    #calculate the means of the dataframe
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['Sales'])))
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['CompPrice'])))
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['Income'])))
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['Advertising'])))
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['Population'])))
        mean = np.append(mean, stats.mean(pd.Series(self.initSet['Price'])))
#        print(mean)
        #calculate mean for Bad,Good,Medium **ShelveLoc**
        bad, good, medium = 0

        for i in self.initSet['ShelveLoc']:
            if i == 'Bad':
                bad += 1
            elif i == 'Good':
                good += 1
            else:
                medium += 1
        #calculate their cumulative values


        # gets rid of the "None" value
        mean = np.delete(mean,0, axis=0)
        #calculate mean for YES:NO answers **URBAN**
        for i in self.initSet['Urban']:
            if i == 'Yes':
                self.values[9] += 1
#            else:
#                self.values[i] += 1
#        print(self.values[9])
        self.values[9] = self.values[9] / count
#        print(self.values[9])
        mean = np.append(mean,self.values[9])

#        for column in self.initSet:
#            mean = np.append(mean, self.initSet[column]).sum()
        print(type(mean))
        print(self.values)


    def __missing__(self, key):
        return 0

