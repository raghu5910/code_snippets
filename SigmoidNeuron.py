class SigmoidNeuron:

  def __init__(self):
    self.w = None
    self.b = None

  def sigmoid(self, z):
    return 1/(1 + np.exp(-z))

  def initialise_parameters(self, dim, intialise = True):
    """
    This function creates a vector of random numbers of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    if intialise:
      self.w = np.random.rand(dim,1)
      self.b = np.random.rand()
    else:
      self.w = np.zeros([dim,1], dtype = float)
      self.b = 0

    assert(self.w.shape == (dim, 1))
    assert(isinstance(self.b, float) or isinstance(self.b, int))

    return self.w, self.b

  def propagate(self, X, Y):

    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[0]
    # FORWARD PROPAGATION (FROM X TO COST)
    A = self.sigmoid((X@(self.w)) + self.b)                      # compute activation
    cost =-(((np.log(A).T)@Y) +((np.log(1-A)).T)@(1 - Y))/m    # compute cost


    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = ((X.T)@(A - Y))/m
    db = np.sum(A - Y, axis = 0, keepdims = True)/m # axis = 0 for column sum


    assert(dw.shape == self.w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db, }

    return grads, cost


  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True):

    costs = {}
    dim = X.shape[1]
    m = X.shape[0]
    Y = Y.reshape((m, 1))

    self.w, self.b = self.initialise_parameters(dim, initialise)
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      grads, costs[i] = self.propagate(X, Y)


      dw = grads["dw"]
      db = grads["db"]

      self.w = self.w - learning_rate*dw
      self.b = self.b - learning_rate*db

    plt.plot(costs.values())
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy error')
    plt.show()

  def predict(self, X):
    Y_pred = self.sigmoid((X@(self.w)) + self.b)
    return Y_pred
