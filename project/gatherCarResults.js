const arff = require('../arffTools.js');

function test(callback) {
  const testfile = './project/cars.arff';

  arff.trainTest(testfile, 0.9, (output) => {  
    if (callback) {
      callback(output);
    }
  }, {shuffle: true, targetAttributes: ['safety', 'rating']});
}

function gatherResultsForReport() {
  let allResults = [];
  
  const MAX = 5;
  
  for (let i = 0; i < MAX; i++) {
    test((output) => {
      let organizedResults = [];
      
      output.training.results.forEach((result) => {
        let organized = {};
        
        organized.trainingAccuracy = result.epochs.map((epoch) => {return Number((epoch.accuracy.reduce((a,b) => a+b, 0) / epoch.accuracy.length * 100).toFixed(2));});
        
        organized.epochCount = result.epochsTillFinish;
        organized.finalWeights = result.finalWeights;
        organized.finalTrainingAccuracy = result.finalAccuracy / result.inputs.length * 100;
        
        organized.firstEpoch = Number((result.firstEpoch.accuracy.reduce((a,b) => a+b, 0) / result.firstEpoch.accuracy.length * 100).toFixed(2));
        
        organizedResults.push(organized);
      });
    
      output.testing.forEach((result, index) => {
        organizedResults[index].testingAccuracy = result.reduce((a, b) => {if(b) {return a + 1;}else {return a;}}, 0) / result.length * 100;
      });
      
      console.log(organizedResults[0].finalWeights);
      
      // console.log(organizedResults);
      
      allResults.push(organizedResults);
      
      if (allResults.length === MAX) {
        // createOutput(allResults);
      }
    });
  }
  
}

function createOutput(results) {
  let setup = results.map((trial) => {
    return trial.map((result) => {
      return {
        test: Number(result.finalTrainingAccuracy.toFixed(2)),
        train: Number(result.testingAccuracy.toFixed(2)),
        epochCount: result.epochCount
      };
    });
  });
  
  let print = [
    {
      test: 0,
      train: 0,
      epochCount: 0
    },
    {
      test: 0,
      train: 0,
      epochCount: 0
    }
  ];
  
  setup.forEach((trial) => {
    trial.forEach((output, index) => {
      print[index].test += output.test;
      print[index].train += output.train;
      print[index].epochCount += output.epochCount;
    });
  });
  
  print = print.map((output) => {
    return {
      test: Number((output.test / results.length).toFixed(2)),
      train: Number((output.train / results.length).toFixed(2)),
      epochCount: Number((output.epochCount / results.length).toFixed(2))
    };
  });
  
  console.log(print);
}

gatherResultsForReport();