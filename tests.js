function runTests() {
  console.log('Starting tests...');
  
  let testResults = {};
  
  const testOperations = [
    orOperation(),
    xorOperationLinear(),
    myData1(),
    myData2(),
    xorOperation()
  ];
  
  testOperations.forEach((testOperation) => {
    const RATE = 0.4;
    
    console.log('testing ' + testOperation.name);
    const weights = testOperation.patterns[0].map(() => 0).concat(0);
    let testResult = {
      results: Perceptron.train(weights, testOperation.patterns, testOperation.targets, RATE),
    };
    
    testResult.pass = testResult.results.maxReached.reached === testOperation.maxReached;
    testResult.inputs = testResult.results.patterns;
    testResult.finalWeights = testResult.results.epochs[testResult.results.epochs.length - 1].weights;
    
    testResults[testOperation.name] = testResult;
  });
  
  console.log('Testing complete.');
  
  return testResults;
  
  //
  // FUNCTIONS FOR TEST CASES
  //
  
  function orOperation() {
    return {
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
    };
  }


  function xorOperationLinear() {
    return {
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
    };
  }


  function xorOperation() {
    return {
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
    };
  }

  function myData1() {
    return {
      patterns: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
      targets: [1,1,1,1,1,0,1,0,0],
      name: 'myData1',
      maxReached: false
    };  
  }
  
  function myData2() {
    return {
      patterns: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
      targets: [0, 0, 1, 1, 1, 1, 0, 0, 1],
      name: 'myData2',
      maxReached: true
    };  
  }
}