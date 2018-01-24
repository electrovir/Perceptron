const Perceptron = (() => {
  
  // it might be considered unneccessary to include this since node-arff already includes a randomize function, but this file is meant to be node independent and able to run in the browser.
  // used with permission from: https://github.com/Prendus/functions/blob/master/functions.js
  // 
  // This MIGHT be a pretty big performance hit
  function shuffleItems(array) {
    let resultArray = [];
    
    for (let i = 0; i < array.length; i++) {
      let randomIndex = Math.floor(Math.random() * array.length);
      
      resultArray.push(array[randomIndex]);
      
      array.splice(randomIndex, 1);
    }
    resultArray = resultArray.concat(array);
    return resultArray;
	}
  
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
          return Number((weight + learningRate * pattern[weightIndex] * (targets[patternIndex] - output)).toFixed(5));
        });
      }
      returnObject.accuracy.push(accurate);
      return returnObject;
    }, {
      accuracy: [],
      startWeights: weights,
      weights: weights, // this one changes
      outputs: []
    });
  }

  /**
   * Get the output for a single pattern
   * @param  {Array} weights       Array of weights. Each element should be a number.
   * @param  {Array} pattern       An array of inputs. This is a single row in the complete list of patterns.
   * @param  {Number} threshold    Neuron threshold. output > threshold grants an output of 1. Defaults to 0.
   * @return {Number}              The output of the neuron. Either a 1 or a 0.
   */
  function getPatternOutput(pattern, weights, threshold) {
    return Number(pattern.reduce((a, b, index) => a + b * weights[index], 0) > threshold);
  }
  

  /**
   * Generates the perceptron output for patterns and weights without doing any learning.
   * @param  {Array} weights       Array of weights. Each element should be a number.
   * @param  {Array} patterns      Matrix of input patterns. Each row is a pattern, each column is an input in the pattern.
   * @param  {Number} threshold    Neuron threshold. output > threshold grants an output of 1. Defaults to 0.
   * @return {Array}               An Array with each element corresponding to the neuron output of the row of the same index in the patterns matrix.
   */
  function runOnData(weights, patterns, threshold = 0) {
    weights = weights.concat(0);
    
    // returns an array of outputs from the given patterns
    return insertBias(patterns).map((pattern) => {
      return getPatternOutput(pattern, weights, threshold);
    });
  }
  
  function testData(weights, patterns, targets, threshold = 0) {
    return runOnData(weights, patterns, threshold).map((output, index) => {
      return output === targets[index];
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
  function train(weights, patterns, targets, learningRate, shuffleBetweenEpochs = false, maxIterations = 999, biasValue = 1, threshold = 0, weightRepeatMax = 2) {
    
    let iteration = 0;
    let trainResults = {
      epochsTillFinish: 0,
      epochs: [],
      inputs: patterns,
      epcohWeights: [],
      initWeights: weights,
      targets: targets,
      finalWeights: [],
      maxReached: {
        reached: false,
        type: 'none'
      }
    };
    
    let latestWeights = weights.concat(0);
    let perfection = false;
    let looping = false;
    let weightsHistory = {};
    
    let inputs = insertBias(patterns, biasValue);
    
    if (latestWeights.length !== inputs[0].length) {
      throw new Error('Invalid number of weights. # weights: ' + weights.length + ' # inputs: ' + inputs[0].length);
    }
    
    // TODO: create way of determining when the training is finished
    while(iteration < maxIterations && !perfection && !looping) {
      const epochResult = singleTargetsEpoch(latestWeights, inputs, targets, learningRate, threshold);
      
      trainResults.epcohWeights.push(latestWeights);
      latestWeights = epochResult.weights;
      
      trainResults.epochs.push(epochResult);
      
      perfection = epochResult.accuracy.length === epochResult.accuracy.reduce((a, b) => a + b, 0);
      
      const weightChecker = checkWeights(epochResult.weights, weightsHistory, weightRepeatMax);
      weightsHistory = weightChecker.history;
      
      looping = weightChecker.repeatMaxxed;
      
      if (shuffleBetweenEpochs) {
        inputs = shuffleItems(inputs);
      }
      
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
    
    trainResults.finalWeights = trainResults.epochs[trainResults.epochs.length - 1].weights;
    trainResults.epochsTillFinish = trainResults.epochs.length - 1;
    
    return trainResults;
  }
  
  /**
   * Count how many times the given weights have been seen and determine if they have been seen too many times or not.
   * @param  {Array} weights       Array of weights. Each element should be a number.
   * @param  {Object} history      An object containing the history of seen weights so far.
   * @param  {Number} max          The max number of a times a list of weights can be repeated.
   * @return {Object}              An object containing both the updated history and a boolean determining if the max weight repition has been passed.
   */
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
  
  // only the train and runOnData functions are publically available.
  return {
    train: train,
    run: runOnData,
    test: testData
  };
})();

// set module exports if in node
if (typeof module !== 'undefined' && typeof module === 'object') {
  Object.assign(module.exports, Perceptron);
}