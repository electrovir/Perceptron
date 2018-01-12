setupOr();

function setupOr() {

  const targets = [[0], [1], [1], [1]];

  const inputs = [[0,0],[0,1],[1,0],[1,1]];

  // random number between 0.1 and .433
  const rate = Math.random()/4+0.1;
  
  let weights = [];
  for (let i = 0; i < inputs[0].length; i++) {
    // random number between -0.1 and +0.1
    weights.push([Math.random()/5-1/10]);
  }
  const bias = [Math.random()/5-1/10];
  
  console.log('Learn Rate: ', rate);
  console.log('Start Weights: ', weights);
  console.log('Start Bias Weights: ', bias);
  
  const finalWeights = train(1, rate, 999, [0], inputs, targets, weights, bias);
  run(inputs, finalWeights.biasWeights, finalWeights.weights, [0], 1);
}


// targetMatrix [input index][neuron index]
// weightsMatrix [input index][neuron index]
// biasWeights [neuron index]
function train(neuronCount, learnRate, iterationMax = 999, thresholds = [], inputVectors = [[]], targetMatrix = [[]], weightsMatrix = [[]], biasWeights = []) {

  function getNewWeight(oldWeight, learnRate, output, target, input) {
    return oldWeight - learnRate * (output - target) * input;
  }
  
  let calculatedWeights = weightsMatrix.map((innerArray) => {
    return innerArray.slice();
  });
  let calculatedBiasWeights = biasWeights.slice();
  
  let iterationCount = 0;
  let correctAnswers = false;
  
  while (iterationCount < iterationMax && !correctAnswers) {
    // reset each iteration
    correctAnswers = true;
    
    for (let inputVectorIndex = 0; inputVectorIndex < inputVectors.length; inputVectorIndex++) {
      for (let neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
        
        const result = getNeuronResult(calculatedBiasWeights[neuronIndex], inputVectors[inputVectorIndex], calculatedWeights, neuronIndex, thresholds);
        
        const target = targetMatrix[inputVectorIndex][neuronIndex];
        // check output vs target values
        if (result !== target) {
          // output was not correct
          correctAnswers = false;
          
          calculatedBiasWeights[neuronIndex] = getNewWeight(calculatedBiasWeights[neuronIndex], learnRate, result, target, -1);
          
          for (let inputIndex = 0; inputIndex < inputVectors[inputVectorIndex].length; inputIndex++) {
            calculatedWeights[inputIndex][neuronIndex] = getNewWeight(calculatedWeights[inputIndex][neuronIndex], learnRate, result, target, inputVectors[inputVectorIndex][inputIndex]);
          }
        }
        
      }
    }
    iterationCount++;
  }
  console.log('>>>>>>>> TRAINING FINISHED <<<<<<<<');
  console.log('Iterations: ', iterationCount);
  console.log('Final Weights: ', calculatedWeights);
  console.log('Final Bias Weights: ', calculatedBiasWeights);
  
  return {
    weights: calculatedWeights,
    biasWeights: calculatedBiasWeights
  };
}

function run(inputVectors, biasWeights, weights, thresholds, neuronCount) {
  let results = [];
  for (let inputVectorIndex = 0; inputVectorIndex < inputVectors.length; inputVectorIndex++) {
    let inputVectorResults = [];
    for (let neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
      
      const result = getNeuronResult(biasWeights[neuronIndex], inputVectors[inputVectorIndex], weights, neuronIndex, thresholds);
      inputVectorResults.push(result);
    }
    results.push(inputVectorResults);
  }
  console.log('>>>>>>>> RUN FINISHED <<<<<<<<');
  console.log('Results: ', results);
  return results;
}

function getNeuronResult(neuronBiasWeight, inputVector, weights, neuronIndex, thresholds) {
  // sum up inputs
  let summation = -1 * neuronBiasWeight;
  
  for (let inputIndex = 0; inputIndex < inputVector.length; inputIndex++) {
    summation += weights[inputIndex][neuronIndex] * inputVector[inputIndex];
  }

  // determine neuron output
  if (summation > thresholds[neuronIndex]) {
    return 1;
  }
  else {
    return 0;
  }
}