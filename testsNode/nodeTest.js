const arff = require('./arffTrain.js');

let weights;

const testFile = './project/my_data_1.arff';

arff.train(testFile, (results) => {
  weights = results[0].finalWeights;

  arff.run(testFile, [weights], console.log);
  arff.test(testFile, [weights], console.log);
});

// const votingFile = './project/voting.arff';
// 
// arff.train(votingFile, (results) => {
//   let accuracy = results[0].epochs.map((epoch) => {return epoch.accuracy.reduce((a,b) => a+b, 0);});
//   console.log('accuracy', String(accuracy));
//   console.log('epochs', results[0].epochs.map((epoch, index) => String(epoch.weights) + ' : ' + accuracy[index]));
// }, true);

