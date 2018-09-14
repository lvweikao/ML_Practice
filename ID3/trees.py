from math import log
import operator
# 输入数据集之后，获取数据集香农熵（根据公式）
# 集所有类别的总和
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # 由于每个熵值的取值范围在负无穷到0，所以-=得到的是就是正总和
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 创建测试数据集测试效果
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return  dataSet, labels

myDat,labels = createDataSet()

# 划分数据集
# 此函数做的是，选取符合某种特征的数据，除开那个特征的其他值
# axis是维数，value是判断是否符合
# 最后返还，axis维与value值相等的行，除开axis外的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
# myDat,labels = createDataSet()
# print(myDat)
# print(splitDataSet(myDat, 0, 0))

# 选择当前最好的一列来进行划分的方式
# 取该列计算香农熵
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# myDat,labels = createDataSet()
# print(chooseBestFeatureToSplit(myDat))

# 返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),
                              reversed = True)
    return sortedClassCount[0][0]

# 创建树的函数代码,不断消耗labels信息，最终构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree
# myDat,labels = createDataSet()
# print(createTree(myDat, labels))

# 使用决策树的分类函数
# 因为在任何一步，测试数据都可以停止判断，所以重要的是知道每步决策树对应类别的索引
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:   classLabel = secondDict[key]
    return classLabel
myDat,labels = createDataSet()
#  因为创建树的函数里面对类别信息进行了删除，所以要另外赋值
#  应当注意列表不能直接赋值，会报错
TestLabels = labels.copy()
myTree = createTree(myDat, TestLabels)
myTree = createTree(myDat, TestLabels)
print(classify(myTree, labels, [0, 1]))