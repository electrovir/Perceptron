const arff = require('../arffTools.js');

function test(callback) {
  const testFileBig = './project/voting.arff';

  arff.trainTest(testFileBig, 0.7, (output) => {
    const accuracy = output.testing[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0)/output.testing[0].length;
    console.log('Voting test: ', accuracy > 0.9, accuracy);
    
    if (callback) {
      callback(output);
    }
  }, {shuffle: true});
}

test();

module.exports = test;


// I need:
// training accuracy
// test accuracy
// # epochs
// 
// weights