

const DATA_SETS = {
  // for simply testing that the perceptron works
  orOperation: {
    patterns: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ],
    targets: [
      0,
      1,
      1,
      1
    ],
    name: 'OR',
    maxReached: false
  },

  // for simply testing that the perceptron works
  xorOperationLinear: {
    patterns: [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 1]
    ],
    targets: [
      0,
      1,
      1,
      0
    ],
    name: 'XOR_linear',
    maxReached: false
  },

  // for simply testing that the perceptron works
  xorOperation: {
    patterns: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ],
    targets: [
      0,
      1,
      1,
      0
    ],
    name: 'XOR',
    maxReached: true
  },

  myData1: {
    patterns: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
    targets: [ 1, 1, 1, 1, 1, 0, 1, 0, 0 ],
    name: 'myData1',
    maxReached: false
  },

  myData2: {
    patterns: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
    targets: [0, 0, 1, 1, 1, 1, 0, 0, 1],
    name: 'myData2',
    maxReached: true
  }
};

function runTests(testOperations, learningRate = 0.4) {
  // console.log('Starting tests...');
  
  let testResults = {};
  let outputString = '';
  
  testOperations.forEach((testOperation) => {
    
    console.log('testing ' + testOperation.name);
    const weights = testOperation.patterns[0].map(() => 0);
    let testResult = {
      results: Perceptron.train(weights, testOperation.patterns, testOperation.targets, learningRate),
    };
    
    testResult.pass = testResult.results.maxReached.reached === testOperation.maxReached;
    
    testResults[testOperation.name] = testResult;
    outputString += testOperation.name + ': epoch count: ' + testResult.epochs + ' passed: ' + testResult.pass + '\n';
  });
  
  // console.log('Testing complete.');
  
  console.log('learningRate: ' + learningRate, outputString, 'results: ', testResults);
  
  return testResults;
}

//
// CODE EXECUTION
// 

// runTests(
//   [
//     DATA_SETS.myData1
//   ],
//   0.1
// );
// 
// runTests(
//   [
//     DATA_SETS.myData2
//   ],
//   0.1
// );
// 
// runTests(
//   [
//     DATA_SETS.myData1
//   ],
//   -0.1
// );
// 
// runTests(
//   [
//     DATA_SETS.myData2
//   ],
//   -0.1
// );
