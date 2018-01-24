const arff = require('../arffTools.js');

const votingFile = './project/votingSmall.arff';

arff.load(votingFile, (data) => {
  console.log(arff.split(data, 0.7).testArffData.data.length === 7);
});