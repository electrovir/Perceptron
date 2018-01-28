const arff = require('../arffTools.js');

let weights;

const testFile = './project/my_data_1.arff';

arff.train(testFile, (output) => {
  weights = output.results[0].finalWeights;

  // arff.run(testFile, [weights], () => {});
  arff.test(testFile, [weights], console.log);
}, {shuffle: true});

