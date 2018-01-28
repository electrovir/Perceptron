const Perceptron = (() => {
  
  // it might be considered unneccessary to include this since node-arff already includes a randomize function, but this file is meant to be node independent and able to run in the browser.
  // 
  // This MIGHT be a pretty big performance hit
  function shuffleDoubleArray(array1, array2) {
    let resultArray1 = [];
    let resultArray2 = [];
    
    for (let i = 0; i < array1.length; i++) {
      let randomIndex = Math.floor(Math.random() * array1.length);
      
      resultArray1.push(array1[randomIndex]);
      resultArray2.push(array2[randomIndex]);
      
      array1.splice(randomIndex, 1);
      array2.splice(randomIndex, 1);
    }
    resultArray1 = resultArray1.concat(array1);
    resultArray2 = resultArray2.concat(array2);
    
    return [resultArray1, resultArray2];
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
    let epochResults = patterns.reduce((returnObject, pattern, patternIndex) => {
      const output = getPatternOutput(pattern, returnObject.weights, threshold);
      returnObject.outputs.push(output);
      
      if (output != targets[patternIndex]) {
        returnObject.weights = returnObject.weights.map((weight, weightIndex) => {
          // targets[i] is the current target for the entire pattern
          // pattern[weightIndex] is the input corresponding to the current weight, weight
          return Number((weight + learningRate * pattern[weightIndex] * (targets[patternIndex] - output)).toFixed(5));
        });
      }
      
      return returnObject;
    }, {
      startWeights: weights,
      weights: weights, // this one changes
      outputs: []
    });
    
    epochResults.accuracy = testData(epochResults.weights, patterns, targets, threshold);
    epochResults.accuracyCount = epochResults.accuracy.reduce((a, b) => a + b, 0);
    
    return epochResults;
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
  function runOnData(weights, inputPatterns, threshold = 0) {
    let patterns = inputPatterns;
    if (patterns[0].length === weights.length - 1) {
      patterns = insertBias(patterns);
    }
    
    // returns an array of outputs from the given patterns
    return patterns.map((pattern) => {
      return getPatternOutput(pattern, weights, threshold);
    });
  }
  
  function testData(weights, patterns, targets, threshold = 0) {
    return runOnData(weights, patterns, threshold).map((output, index) => {
      return Number(output == targets[index]);
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
  function train(weights, patterns, inputTargets, learningRate, shuffleBetweenEpochs = false, maxIterations = 999, biasValue = 1, threshold = 0, maxAccuracyDecrease = 10) {
    
    // this is really hacky looking but this creates a deep copy in one simple line of code
    let targets = JSON.parse(JSON.stringify(inputTargets));
    let inputs = insertBias(patterns, biasValue);
    
    let iteration = 0;
    let trainResults = {
      epochsTillFinish: 0,
      epochs: [],
      inputs: patterns,
      epcohWeights: [],
      initWeights: weights,
      targets: targets,
      finalWeights: undefined,
      maxReached: {
        reached: false,
        type: 'none'
      },
      finalAccuracy: 0,
      firstEpoch: singleTargetsEpoch(weights.slice(), inputs, targets, learningRate, threshold)
    };
    
    let latestWeights = weights.concat(0);
    let perfection = false;
    let worsening = false;
    let accuracies = [{
      accuracy: -1,
      epochIndex: -1,
      countSinceLastReset: 0
    }];
    
    if (latestWeights.length !== inputs[0].length) {
      throw new Error('Invalid number of weights. # weights: ' + weights.length + ' # inputs: ' + inputs[0].length);
    }
    
    
    // TODO: create way of determining when the training is finished
    while(iteration < maxIterations && !perfection && !worsening) {
      const epochResult = singleTargetsEpoch(latestWeights, inputs, targets, learningRate, threshold);
      
      trainResults.epcohWeights.push(latestWeights);
      latestWeights = epochResult.weights;
      
      trainResults.epochs.push(epochResult);
    
      perfection = epochResult.accuracy.length === epochResult.accuracyCount;
      
      if (epochResult.accuracyCount <= accuracies[0].accuracy) {
        accuracies.push({
          accuracy: epochResult.accuracyCount,
          epochIndex: iteration
        });
      }
      else {
        accuracies = [{
          accuracy: epochResult.accuracyCount,
          epochIndex: iteration,
          countSinceLastReset: iteration - accuracies[0].countSinceLastReset
        }];
      }
      
      if (accuracies.length > 10) {
        worsening = true;
      }
      
      if (shuffleBetweenEpochs) {
        [inputs, targets] = shuffleDoubleArray(inputs, targets);
      }
      
      iteration++;
    }
    
    if (iteration >= maxIterations) {
      trainResults.maxReached.reached = true;
      trainResults.maxReached.type = 'maxIterations';
    }
    else if (worsening) {
      trainResults.maxReached.reached = true;
      trainResults.maxReached.type = 'accuracyDecrease';
    }
    
    if (perfection) {
      trainResults.finalWeights = trainResults.epochs[trainResults.epochs.length - 1].weights;
      trainResults.epochsTillFinish = trainResults.epochs.length;
    }
    else {
      trainResults.finalWeights = trainResults.epochs[accuracies[0].epochIndex].weights;
      trainResults.epochsTillFinish = accuracies[0].epochIndex + 1;
    }
    
    trainResults.finalAccuracy = trainResults.epochs[trainResults.epochsTillFinish - 1].accuracy.reduce((a, b) => a + b, 0);
    
    return trainResults;
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