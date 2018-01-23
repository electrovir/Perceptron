const Perceptron = (function() {
  /**
   * Inserts bias input into each pattern.
   * @param  Array patterns List of input patterns. Each array element corresponds to one row. Each array element should be an array of elements representing each column in the row.
   * @return Array          The patterns with a bias inserted.
   */
  function insertBias(patterns, biasValue = 1) {
    return patterns.map((pattern) => {
      return pattern.concat(biasValue);
    });
  }

  /**
   * Run a single epoch of the perceptron, running through all the patterns.
   * @param  {Array} weights       Array of weights. Each element should be a number.
   * @param  {Array} patterns      Matrix of input patterns. Each row is a pattern, each column is an input in the pattern.
   * @param  {Array} targets       Array of target values for each pattern. The index of this array corresponds to the row number in the patterns matrix.
   * @param  {Number} learningRate Scalar representing the learning rate.
   * @param  {Number} threshold    Neuron threshold. output > threshold grants an output of 1. Defaults to 0.
   * @return {Object}              An object containing both the accruacy of the epoch and the newly learned weights.
   */
  function singleTargetsEpoch(weights, patterns, targets, learningRate, threshold = 0) {  
    return patterns.reduce((returnObject, pattern, patternIndex) => {
      const output = getPatternOutput(pattern, returnObject.weights, threshold);
      returnObject.outputs.push(output);
      let accurate = 0;
      
      if (output == targets[patternIndex]) {
        accurate = 1;
      }
      else {
        returnObject.weights = returnObject.weights.map((weight, weightIndex) => {
          // targets[i] is the current target for the entire pattern
          // pattern[weightIndex] is the input corresponding to the current weight, weight
          return weight + learningRate * pattern[weightIndex] * (targets[patternIndex] - output);
        });
      }
      returnObject.accuracy.push(accurate);
      
      return returnObject;
    }, {
      accuracy: [],
      weights: weights,
      outputs: []
    });
  }

  function getPatternOutput(pattern, weights, threshold) {
    return Number(pattern.reduce((a, b, index) => a + b * weights[index], 0) > threshold);
  }

  function runOnData(weights, patterns, threshold = 0) {
    return insertBias(patterns).map((pattern) => {
      return getPatternOutput(pattern, weights, threshold);
    });
  }

  /**
   * Train a perceptron based on the given inputs.
   * @param  {Array} weights              Array of initial weights.
   * @param  {Array} patterns             Matrix of inputs.
   * @param  {Array} targets              Array of neuron target values for each row in patterns
   * @param  {Number} learningRate        The rate at which learning takes place.
   * @param  {Number} [maxIterations=999] Maximum number of epochs allowed. Failsafe to prevent infinite looping. Defaults to 999;
   * @param  {Number} [biasValue=1]           Value of bias input. Defaults to 1.
   * @param  {Number} [threshold=0]           Threshold for each neuron. If neuron net > threshold then neuron output is a 1. Defaults to 0.
   * @return {Array}                      Array of results for each epoch.
   */
  function train(weights, patterns, targets, learningRate, maxIterations = 999, biasValue = 1, threshold = 0, weightRepeatMax = 5) {
    
    let iteration = 0;
    let trainResults = {
      epochs: [],
      patterns: patterns,
      targets: targets,
      biasValue: biasValue,
      maxReached: {
        reached: false,
        type: 'none'
      }
    };
    let latestWeights = weights;
    let perfection = false;
    let looping = false;
    let weightsHistory = {};
    
    const inputs = insertBias(patterns, biasValue);
    
    if (weights.length !== inputs[0].length) {
      throw new Error('Invalid number of weights. # weights: ' + weights.length + ' # patterns: ' + inputs[0].length);
    }
    
    // TODO: create way of determining when the training is finished
    while(iteration < maxIterations && !perfection && !looping) {
      const epochResult = singleTargetsEpoch(latestWeights, inputs, targets, learningRate, threshold);
      
      latestWeights = epochResult.weights;
      
      trainResults.epochs.push(epochResult);
      
      perfection = epochResult.accuracy.length === epochResult.accuracy.reduce((a, b) => a + b, 0);
      
      const weightChecker = checkWeights(epochResult.weights, weightsHistory, weightRepeatMax);
      weightsHistory = weightChecker.history;
      
      looping = weightChecker.repeatMaxxed;
      
      iteration++;
    }
    
    if (iteration >= maxIterations) {
      trainResults.maxReached.reached = true;
      trainResults.maxReached.type = 'maxIterations';
    }
    else if (looping) {
      trainResults.maxReached.reached = true;
      trainResults.maxReached.type = 'looping';
      
    }
    
    console.log('End training.');
    
    return trainResults;
  }

  function checkWeights(weights, history, max) {
    
    const weightsKey = weights.reduce((key, weight) => {
      return key.concat(weight, '_');
    }, '');
    
    if (!history.hasOwnProperty(weightsKey)) {
      history[weightsKey] = 0;
    }
    
    history[weightsKey]++;
    
    return {
      history: history,
      repeatMaxxed: history[weightsKey] > max
    };
  }
  
  return {
    train: train,
    run: runOnData
  };
})();