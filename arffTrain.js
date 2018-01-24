if (!module.parent && process.argv.length > 1) {
  trainOnArff(process.argv[2], (data) => console.log(data.map((item) => item.finalWeights)), ...process.argv.slice(3));
}

function loadArff(fileName, callback) {
  require('node-arff').load(fileName, (error, data) => {
    if (error) {
      throw new Error(error);
    }
    
    if ((typeof callback).toLowerCase() === 'function') {
      callback(data);
    }
  });
}

function defaultWeights(targets, inputs) {
  return targets.map(() => {
    // one weight for each attribute
    return inputs.map(() => {
      // default weight of 0
      return 0;
    });
  });
}

function getArffInputs(fileName, callback, targetAttributes = [], shuffle = false) {
  function getInputs(data) {
    if (shuffle) {
      data.randomize();
    }
    // set the target to the last attribute if no target attributes are given
    if (targetAttributes.length === 0) {
      targetAttributes = [data.attributes[data.attributes.length - 1]];
    }
    else if (!targetAttributes) {
      targetAttributes = [];
    }
    
    // get all the columsn that represent the targets and place them in one place
    let targetColumns = targetAttributes.map((attribute) => {
      return data.data.map((dataRow) => {
        return dataRow[attribute];
      });
    });
    
    // filter out the target attributes to get the input attributes
    const inputAttributes = data.attributes.filter((attribute) => {
      return targetAttributes.indexOf(attribute) === -1;
    });
    
    // grab all the input attributes out of each data row to form the set of inputs/patterns
    let inputs = data.data.map((dataRow) => {
      return inputAttributes.map((attribute) => {
        return dataRow[attribute];
      });
    });
    
    
    return {
      targetColumns: targetColumns,
      patterns: inputs,
      inputAttributes: inputAttributes,
      targetAttributes: targetAttributes
    };
  }
  
  loadArff(fileName, (data) => {
    let inputs = getInputs(data);
    
    if ((typeof callback).toLowerCase() === 'function') {
      callback(inputs);
    }
    else {
      console.log('arff inputs: ', inputs);
    }
  });
}


// TODO: update javadoc
// FIXME: generalize this so it can be used in later labs more
/**
 * Loads an arff file and trains a perceptron on the data. This is asynchronous and returns nothing!
 * @param  {String}   fileName      The name of the arff file to load.
 * @param  {Function} callback      The function to call with the perceptron's results. If no callback is passed, the results are merely loggged.
 * @param  {Array}   targetAttributes  An array of strings. This tells the arrf reader which attributes to treat as targets rather than inputs. If this is left blank, the last attribute is automatically picked. If multiple columns are picked, a perceptron will be created for each in the same order as the passed target attributes.
 */
function trainOnArff(fileName, callback, shuffle, learningRate, initWeights, targetAttributes = []) {  
  getArffInputs(fileName, (inputs) => {
    trainPerceptron(inputs.patterns, inputs.targetColumns, callback, shuffle, learningRate, initWeights);
  }, targetAttributes, shuffle);
}

function trainPerceptron(patterns, targetColumns, callback, shuffle = false, learningRate = 0.1, initWeights = []) {
  const startTime = Number(new Date());
  const Perceptron = require('./perceptron.js');
  
  // set all default weights to 0 if none are given
  if (!initWeights || initWeights.length === 0) {
    // form a set of weights for each target
    initWeights = defaultWeights(targetColumns, patterns[0]);
  }
  
  let trainResults = targetColumns.map((targets, targetIndex) => {
    return Perceptron.train(initWeights[targetIndex], patterns, targets, learningRate, shuffle);
  });
  
  console.log('time:', Number(new Date()) - startTime);
  
  if (callback && typeof callback === 'function') {
    callback(trainResults);
  }
  else {
    console.log(trainResults);
  }
}

/**
 * Asynchronously run a Perceptron with an arff file's inputs and given weights. No training is done.
 * @param  {String}   fileName   Path to the arff file to load.
 * @param  {Array}   weightsSet  A matrix. Each entry in the matrix is a set of weights for each output. This allows support of multiple outputs.
 * @param  {Function} callback   Passed the resulting data beacuse this is asynchronous
 */
function runOnArff(fileName, weightsSet, callback) {
  getArffInputs(fileName, (arffInputs) => {
    const Perceptron = require('./perceptron.js');
    const results = weightsSet.map((weights) => {
      return Perceptron.run(weights, arffInputs.patterns);
    });
  
    if ((typeof callback).toLowerCase === 'function') {
      callback(results, arffInputs);
    }
    else {
      console.log(results);
    }
  });
}

function testOnArff(fileName, weightsSet, callback) {
  getArffInputs(fileName, (arffInputs) => {
    callback(testPerceptron(arffInputs.patterns, weightsSet, arffInputs.targetColumns));
  });
}

function testPerceptron(patterns, weightsSet, targetColumns, callback) {
  const Perceptron = require('./perceptron.js');
  
  const results = weightsSet.map((weights, index) => {
    return Perceptron.test(weights, patterns, targetColumns[index]);
  });
  
  return results;
}

function trainTestArff(fileName, trainSplit, callback, shuffle = true, learningRate = 0.1, targetAttributes = [], initWeights = []) {
  if (trainSplit > 1 || trainSplit <= 0) {
    throw new Error('Invalid training split: ' + trainSplit);
  }
  getArffInputs(fileName, (arffInputs) => {
    let trainInputs = [];
    let trainTargetColumns = [];
    
    let testInputs = [];
    let testTargetColumns = [];
    
    arffInputs.patterns.forEach((pattern, index) => {
      // no need for randomness here, the patterns have already been randomized
      if (index < Math.floor(trainSplit * arffInputs.patterns.length)) {
        trainInputs.push(pattern);
        arffInputs.targetColumns.forEach((targetColumn, columnIndex) => {
          if (trainTargetColumns[columnIndex] === undefined) {
            trainTargetColumns.push([]);
          }
          trainTargetColumns[columnIndex].push(targetColumn[index]);
        });
      }
      else {
        testInputs.push(pattern);
        arffInputs.targetColumns.forEach((targetColumn, columnIndex) => {
          if (testTargetColumns[columnIndex] === undefined) {
            testTargetColumns.push([]);
          }
          testTargetColumns[columnIndex].push(targetColumn[index]);
        });
      }
    });
    
    console.log('trainInputs', trainInputs, 'trainTargetColumns', trainTargetColumns, 'testInputs', testInputs, 'testTargetColumns', testTargetColumns);
    
  }, targetAttributes, true);
  
  // getArffInputs(fileName, trainPerceptron.bind(null, targetAttributes, callback), targetAttributes, shuffle);
}

module.exports = {
  train: trainOnArff,
  run: runOnArff,
  test: testOnArff,
  trainTest: trainTestArff
};