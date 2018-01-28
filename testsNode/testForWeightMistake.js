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
  
  for (let i = 0; i < 5; i++) {
    test((output) => {
      let organizedResults = {};
      
      organizedResults.epochAccuracy = output.training.results[0].epochs.map((epoch) => {return epoch.accuracy.reduce((a,b) => a+b, 0) / epoch.accuracy.length;});
      
      organizedResults.testingAccuracy = output.testing[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / output.testing[0].length;
      
      organizedResults.testingAccuracy1 = output.testing1[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / output.testing1[0].length + '';
      
      organizedResults.testingAccuracy2 = output.testing2[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / output.testing2[0].length + ' vs. ' + output.training.results[0].epochs[output.training.results[0].epochsTillFinish - 1].accuracy.reduce((a,b) => a+b, 0) / output.training.results[0].epochs[output.training.results[0].epochsTillFinish - 1].accuracy.length;
      
      organizedResults.testingAccuracy3 = output.testing3[0].reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / output.testing3[0].length;
      
      // organizedResults.epochCount = output.training.results[0].epochsTillFinish;
      // organizedResults.finalWeights = output.training.results[0].finalWeights;
      
      results.push(organizedResults);
      console.log(organizedResults);
    });
  }
  
}

let data = gatherResultsForReport();