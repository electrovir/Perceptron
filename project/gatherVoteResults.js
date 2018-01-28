const arff = require('../arffTools.js');

function test(callback) {
  const testFileBig = './project/voting.arff';

  arff.trainTest(testFileBig, 0.7, (output) => {  
    if (callback) {
      callback(output);
    }
  }, {shuffle: true});
}

function gatherResultsForReport() {
  let results = [];
  
  const MAX = 5;
  
  for (let i = 0; i < MAX; i++) {
    test((output) => {
      let organizedResults = {};
      
      organizedResults.trainingAccuracy = output.training.results[0].epochs.map((epoch) => {return Number((epoch.accuracy.reduce((a,b) => a+b, 0) / epoch.accuracy.length * 100).toFixed(2));});
      organizedResults.testingAccuracy = output.testing[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / output.testing[0].length * 100;
      organizedResults.epochCount = output.training.results[0].epochsTillFinish;
      organizedResults.finalWeights = output.training.results[0].finalWeights;
      organizedResults.finalTrainingAccuracy = output.training.results[0].finalAccuracy / output.training.results[0].inputs.length * 100;
      organizedResults.firstEpoch = Number((output.training.results[0].firstEpoch.accuracy.reduce((a,b) => a+b, 0) / output.training.results[0].firstEpoch.accuracy.length * 100).toFixed(2));
      
      results.push(organizedResults);
      if (results.length === MAX) {
        createOutput(results);
      }
    });
  }
  
}

function createOutput(data) {
  let table1 = data.map((item) => [Number(item.finalTrainingAccuracy.toFixed(1)), Number(item.testingAccuracy.toFixed(1)), item.epochCount]);
  console.log(table1);
  
  let averageWeights = data.map((a) => a.finalWeights).reduce((sum, current) => {
    if (!sum) {
      return current;
    }
    else {
      return sum.map((b, i) => b + current[i]);
    }
  }).map(c => Number((c / 4).toFixed(4)));
  console.log(JSON.stringify(averageWeights).replace(/,/g, ', '));
  
  data.forEach((item, i) => {
    console.log('accuracies' + i + ' = ' + JSON.stringify([item.firstEpoch].concat(item.trainingAccuracy)).replace(/,/g, ', ') + ';');
  });
  
  let epochCount = data.map((item) => item.epochCount);
  console.log(epochCount);
}

gatherResultsForReport();