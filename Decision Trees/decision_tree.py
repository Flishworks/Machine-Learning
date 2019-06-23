import numpy as np

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.tree = None
        return
    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )
    def get_best_feature(self, features, targets, feature_names):
        ig = []
        max_ig = 0
        max_ig_index = 0
        for feature in range(features.shape[1]):
            ig.append(information_gain(features, feature, targets))
            if ig[feature] > max_ig:
                max_ig = ig[feature]
                max_ig_index = feature
        feature_name = feature_names[max_ig_index]
        #print(max_ig_index, " ", max_ig, " ", attribute_names[max_ig_index])
        return feature_name, max_ig_index

    def branch(self, features, targets, feature_names, value):
        if len(feature_names) == 0: #no more features to split on
            print("prediction = ",1 if targets.sum()/len(targets)>.5 else 0)
            branch = Tree(
                attribute_name="Prediction",
                attribute_index = 1 if targets.sum()/len(targets)>.5 else 0, #store prediction here, because where else?
                value=value, #np.round(np.sum(targets)/len(targets),0), #choose max class
                branches = None
            )
            return branch
        if (np.sum(targets)/len(targets))%1 == 0: #all values of target are the same
            print("prediction = ",1 if targets.sum()/len(targets)>.5 else 0)
            branch = Tree(
                attribute_name="Prediction",
                attribute_index = 1 if targets.sum()/len(targets)>.5 else 0, #store prediction here, because where else?
                value=targets[0],
                branches=None
            )
            return branch
        best_feature_name, best_feature_index = self.get_best_feature(features, targets, feature_names)
        if np.sum(features[:, best_feature_index]/len(targets))%1 == 0: #all values of features are the same
            print("prediction = ",1 if targets.sum()/len(targets)>.5 else 0)
            branch = Tree(
                attribute_name="Prediction",
                attribute_index = 1 if targets.sum()/len(targets)>.5 else 0, #store prediction here, because where else?
                value=targets[0],
                branches=None
            )
            return branch
        print(best_feature_name, " ", value)
        features_temp = features[features[:, best_feature_index] == value] #filter data where best attribute = specified value
        targets = targets[features[:, best_feature_index] == value] #repeat for targets
        features = features_temp
        features = np.delete(features, best_feature_index, 1) #remove the best feature column from data
        feature_names = np.delete(feature_names, best_feature_index)
        branch = Tree(
                    attribute_name=best_feature_name,
                    attribute_index=best_feature_index,
                    value=value,
                    branches=[self.branch(features, targets, feature_names, 0),self.branch(features, targets, feature_names, 1)]
                )
        return branch

    def fit(self, features, targets):
        self._check_input(features)
        feature_names = self.attribute_names
        self.tree = Tree(
                attribute_name="root",
                attribute_index=None,
                value=None,
                branches=[self.branch(features, targets, feature_names, 0), self.branch(features, targets, feature_names, 1)]
            )


    def get_branch(self, branch, observation):
        next_attribute_index = branch.branches[0].attribute_index #next feature to split on, or the prediction
        next_attribute_name = branch.branches[0].attribute_name #next feature to split on
        if next_attribute_name == "Prediction":
            return next_attribute_index
        else:
            choice = int(observation[next_attribute_index]) #observation's value of this feature
            observation = np.delete(observation, next_attribute_index)
        return self.get_branch(branch.branches[choice], observation)

    def predict(self, features):
        self._check_input(features)
        #import pdb; pdb.set_trace()
        root = self.tree
        predictions = []
        for observation in features:
            next_attribute_index = root.branches[0].attribute_index #next feature to split on
            choice = int(observation[next_attribute_index]) #observation's value of this feature
            observation = np.delete(observation, next_attribute_index)
            predictions.append(self.get_branch(root.branches[choice], observation))
        return np.array(predictions)


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        if tree.attribute_name == "Prediction":
            prediction = tree.attribute_index
            print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, prediction))
        else:
            print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def entropy(vect):
    '''
    entropy = −sum(P*log(P))
    '''
    if len(vect) == 0:
        return 0
    vect = vect.astype(float)
    p1 = vect.sum()/len(vect)
    if p1 == 0:
        return 0
    p2 = 1-p1
    if p2 == 0:
        return 0
    ent = -(p1*np.log2(p1)+p2*np.log2(p2))
    if  np.isnan(ent):
        ent = 0
    return ent

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    if len(targets) > 0:
        parent_entropy = entropy(targets)
        filtArr1 = targets[features[:,attribute_index] >= 0.5] #extract data where specified atribute == 1
        filtArr0 = targets[features[:,attribute_index] < 0.5] #extract data where specified atribute == 0
        child_entropy = (entropy(filtArr1)*len(filtArr1)+entropy(filtArr0)*len(filtArr0))/len(targets)#calc entropy for targets after the data split by this attribute
        return float(parent_entropy - child_entropy)
    else:
        return 0

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
