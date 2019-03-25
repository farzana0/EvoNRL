import scipy.io as sio
import numpy 
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

global returnprecsion

def build_word_vector_matrix(vector_file, n_words):
  '''Return the vectors and labels for the first n_words in vector file'''
  numpy_arrays = []
  labels_array = []
  with open(vector_file, 'r') as f:
    for c, r in enumerate(f):
      sr = r.split()

      labels_array.append(sr[0])
      numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )
      if c == n_words :
        return numpy.array( numpy_arrays[1:] )
  return numpy.array( numpy_arrays )

def averageprec(x, Y):
  random_state = numpy.random.RandomState(0)
  #Y = build_labels_matrix_0('Homo_sapiens.mat')
  #print Y
  n_classes = Y.shape[1]
  #X = build_word_vector_matrix('PIPemb.emd', 3890)
  X = x

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=random_state)
  # We use OneVsRestClassifier for multi-label prediction
  from sklearn.multiclass import OneVsRestClassifier

  # Run classifier
  classifier = OneVsRestClassifier(sklearn.linear_model.LogisticRegression(random_state=random_state))

  classifier.fit(X_train, Y_train)
  y_score = classifier.decision_function(X_test)

  from sklearn.metrics import precision_recall_curve
  from sklearn.metrics import average_precision_score

  # For each class
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                          y_score[:, i])
      average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

  # A "micro-average": quantifying score on all classes jointly
  precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
      y_score.ravel())
  average_precision["macro"] = average_precision_score(Y_test, y_score,
                                                       average="macro")
  returnprecision = average_precision["macro"]
  #print('Average precision score, macro-averaged over all classes: {0:0.4f}'
  #      .format(average_precision["macro"]))

  return returnprecision
