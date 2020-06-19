// hyperparameters
let numOfClasses = 3;
let numOfEpochs = 15;
let learningRate = 0.001;
let batchSize = 3; //ceil [2,5,10,20,25,33,50,100]%
let numOfUnits = 50;

// optimiser
let adamOp;

// confidence threshold
let cThreshold = 0.8;

// default zoom rate is 100% (1.0 = 100%)
let zoomValue = 1.0;

// stores CNN model, classifier model and webcam
let cam;
let model;
let classifier;

// counters for the amount of training data supplied for each function
// *maybe create dictionary in place of the variables
let classes = ["No action", "Jump", "Duck", "Go back", "Go forward", "Zoom in", "Zoom out"]; 
let classesCounter = Array(numOfClasses).fill(0);

// dataset for training that will contain both images and labels
let trainingDataset = {data: null, labels: []};

// boolean variables that stops certain functions from running
let isRunning;
let isTesting = false;
let isTrained;

// variables for the error filter function
let streakTarget = 2;
let streak = 0;
let prevPredict = 0;

// Accuracy and loss chart
let lossChart;
let accChart;

// applied and visual crop height and width
let cropWidth = 300;
let cropHeight = 300;
let tempCropWidth = 300;
let tempCropHeight = 300;

// optional preprocessing techniques
let flipImage = false;
let segmentImage = false;
let isTestCamOn = false;

// stores crop visualiser canvas object
const cropOverlay = document.getElementById("cropOverlay").getContext("2d");

// stores value returned by setInterval()
let captureInterval;

//let mean;
let duckEvent = new Event('downAction');
let jumpEvent = new Event('upAction');
let noEvent = new Event('noAction');


// loads CNN model, prepares webcam and sets HTML values and elements
async function initialise(){
  //tf.enableDebugMode()
  try{
    cam = await tf.data.webcam(document.getElementById("cam"), {resizeWidth: cropWidth, resizeHeight: cropHeight, centerCrop: true});
    //model =  await tf.loadLayersModel("https://storage.googleapis.com/co600_models/model99/model.json");
    model = await mobilenet.load();
    //console.log(model.summary());
    htmlValueInit();
  
    await createModel();
    //testTime()
 }
  catch(error){
    document.getElementById("errorDiv").style.display = "block";
    document.getElementById("loading").style.display = "none";
  }

  //preprocess();
    
}


// builds and compiles classifier model to classify data from pretrained model
async function createModel(){
  //global patterns

  // stores output tensor of CNN model
  const modelOutput = await modelInfer();

  // input shape set to shape of output tensor from CNN model
  const input = tf.input({shape: [modelOutput.shape[1],]});
  //const input = tf.input({shape: [modelOutput.shape[1], modelOutput.shape[2], modelOutput.shape[3],]});
  //const flatten = tf.layers.flatten();
  const layer1 = tf.layers.dense({units: numOfUnits, activation: "relu"});
  // output layer that has units that are the number of classes. Uses softmax for multiclass classification
  const layer2 = tf.layers.dense({units: numOfClasses, activation: "softmax", useBias: false});
  // connects all layers together in order
  //const output = await layer2.apply(layer1.apply(flatten.apply(input)));
  const output = await layer2.apply(layer1.apply(input));

  classifier = tf.model({inputs: input, outputs: output}); 

  // creates optimiser 'adam' along with learning rate
  adamOp = tf.train.adam(learningRate);

  // compiles model with optimiser and loss being categorical crossentropy 
  // categorical crossentropy calculates the loss of the prediction for the correct class
  classifier.compile({optimizer: adamOp, loss: "categoricalCrossentropy", metrics: "accuracy"});

  // releases tensor from memory to prevent memory leak
  modelOutput.dispose();

  // resets trained and running to default values as a new model represents a new state
  isTrained = false;
  isRunning = false;

  //gradient
  //console.log(classifier.summary());
  //rmsprop
  //Kernel initialiser
}


// contains loop for creating predictions and browser functions based on predictions
async function run(){
  // only begins loop if model is trained
  if(!isTrained){
    // if classifier model is not trained then the user is alerted
    alert("Trained model required");
    return;
  }

  // turns on webcam to start getting input
  if(!cam.stream.active){
    await toggleCam();
  }

  // if prediction loop is already running when this function is called then it is stopped
  // inner text of button is changed accordingly to what it will do
  if(!isRunning){
    document.getElementById("run").innerText = "Stop";
    isRunning = true;
  }
  else{
    stopRun();
  }

  while(isRunning){ 

    // stores CNN output of current image captured by webcam
    let modelOutput = await modelInfer();
    
    // tensor storing prediction values for each output produced by clasifier model using CNN output as input
    let prediction = await classifier.predict(modelOutput);
    
    //class with the highest prediction stored
    let prediction1 = await prediction.as1D();
    let maxPrediction = prediction1.max();
    if(!isTesting && maxPrediction.arraySync() >= cThreshold){
      let prediction2 = await prediction1.argMax();
      browserFunctions(prediction2.toString().slice(-1));
      prediction2.dispose();
    }
    else if(isTesting){
      testPredictions(prediction1.dataSync());
    }
    
    // tensors related to the prediction made by classifier and CNN output tensor are all disposed
    prediction.dispose();
    prediction1.dispose();
    modelOutput.dispose();
    maxPrediction.dispose();
    
    // tf.memory checks how many tensors and bytes are occupying space in the GPU
    //console.log(tf.memory());

    // waits for next webcam frame before repeating loop
    await tf.nextFrame();
  }
}


async function browserFunctions(predictedClass){
  // function called according to predicted class

  // error filter
  // if(isConsecutive(predictedClass)){

      // functions for changing vertical window position
      //let content = document.getElementById("zoom");
      if(predictedClass == '1'){
        //window.scrollBy(0, -scrollSpeed);
       
    //console.log(upE);
    document.dispatchEvent(jumpEvent);
      }
      else if(predictedClass == '2'){
        
        //console.log(upE);
        document.dispatchEvent(duckEvent);
        //window.scrollBy(0, scrollSpeed);
      }
      else{
        document.dispatchEvent(noEvent);
      }
      /*
      else if(predictedClass == 3){ 
        window.history.forward();
      }
      else if(predictedClass == 4){
        window.history.back();
      }
      //zoom goes out of viewport
      else if(predictedClass == 5){
        zoomValue +=  0.01;
        document.body.style.transform = "scale("+zoomValue+")";
        console.log(document.body.style.transform);
      }
      else if(predictedClass == 6){
        zoomValue -= 0.01;
        document.body.style.transform = "scale("+zoomValue+")";
        console.log(document.body.style.transform);
      }
      else if(predictedClass == 7){
        window.open("https://www.example.com");
      }
    }
    */
}


async function testPredictions(tensorValues){
  // prediction values displayed in container
  for(let i = 0; i < numOfClasses; i++){
    let testElement = document.getElementById("class" + (i+1));
    let value = Math.round(tensorValues[i]*100) + "%";
    testElement.innerText = value; //classes[i] + ": " + 
    testElement.style.width = value;
 }
}

function toggleTest(buttonElement){
  if(isTesting){
    isTesting = false;
    buttonElement.innerText = "Test mode: Off"
  }
  else{
    isTesting = true;
    buttonElement.innerText = "Test mode: On";
  }
}

async function train(){
  // prevents the classifier from being trained until data for all classes are supplied
  for(let i = 0; i < numOfClasses; i++){
    if(classesCounter[i] == 0){
      alert("Need to add training data to all functions");
      return;
    }
  }

  // array of numbers representing labels are transformed to a 1 dimensional tensor of type 32 bit integer
  // one hot encoding. Changes nominal values to binary up to length of 3 e.g. 1 will be 001
  let labelTensors = tf.tensor1d(trainingDataset.labels, 'int32');
  let trainingLabels = await tf.oneHot(labelTensors, 3);
  
  let trainingProgress = document.getElementById("trainingProgress");

  // resets loss chart and accuracy for new post processing
  resetChart(lossChart);
  resetChart(accChart);

  // trains the classifier model and updates chart per epoch, displaying the loss and accuracy
  const results = await classifier.fit(trainingDataset.data, trainingLabels, {batchSize: batchSize, epochs: numOfEpochs, verbose: 0, shuffle: true, 
    callbacks: {onEpochEnd: async(epoch, logs) => { trainingProgress.value = Math.round(((epoch+1)/numOfEpochs)*100);
    updateChart(epoch, logs.loss, lossChart); updateChart(epoch, logs.acc, accChart);} 
  }}); 

  isTrained = true;

  trainingLabels.dispose();
  labelTensors.dispose();
}

async function postprocess(metrics){

}

// resets chart by creating an empty array for both labels and data
function resetChart(metricsChart){
  metricsChart.data.labels = [];
  metricsChart.data.datasets[0].data = [];
  metricsChart.update();
}

// updates chart using values returned from training
function updateChart(label, dataset, metricsChart){
  metricsChart.data.labels.push((label+1));
  metricsChart.data.datasets[0].data.push(dataset);
  metricsChart.update();
}

// a chart is created with the use of chartjs
function createMetricsChart(name){
  let metricsChart = new Chart(document.getElementById(name + "Chart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: name + " per epoch",
        data: []

      }]
    },
    // allows chart to be resized with a different ratio
    maintainAspectRatio: false
  });

  return metricsChart;
}


// hides or shows container that contains statistics related to predictions made and training results
async function showDataAnalytics(button){
  const results = document.getElementById("trainingResults");

  if(results.style.display != "block") {
    button.innerText = "Hide training results";
    results.style.display =  "block";
  }
  else{
    button.innerText = "Show training results"
    results.style.display = "none";
  }
}


// adds tensors and labels to dataset
async function addData(data, label){
  trainingDataset.labels.push(label);   
  
  if(trainingDataset.data == null){
    trainingDataset.data = data;
  }
  else{
    trainingDataset.data = trainingDataset.data.concat(data);
    data.dispose();
  }

  //trainingDataset.data.push({data});
}


// gets the output of CNN for the image captured once function is called
async function modelInfer(){
  let image = await cam.capture(); //tf.expandDims(await preprocess()); //
  //image.print();
  let output = await model.infer(image);
  //let output = await model.predict(image);
  //console.log("test");
  image.dispose();
  return output;
}


// displays image captured on canvas for the class it was captured for
// adds CNN output to training dataset along with label for the class it was captured for 
async function captureFrame (label, frame){
    
  // prevents attempt on capturing images if webcam is not streaming
 if(!cam.stream.active){
	 alert("Turn on cam to capture images");
	 return;
 }
 
 let captureFrames = async () => {
  classesCounter[label]++;
  frame.parentNode.childNodes[0].innerText = classes[label] + ": " + classesCounter[label];
 
  let image = await tf.image.resizeBilinear(await cam.capture(), [112, 112], true);//preprocess();
  let resizedImage = image.toInt()
  addData(await modelInfer(), label);
  
  await tf.browser.toPixels(resizedImage, frame);
  image.dispose();
  resizedImage.dispose();
  };

  captureFrames();
  captureInterval = setInterval( captureFrames , 200);
  
}


// all dynamic HTML values are initialised
async function htmlValueInit(){ 
  // Draws square that is used to overlay webcam footage and represent the crop height and width
  cropOverlay.strokeRect(150 - (150*(cropWidth/300)), 150 - (150*(cropHeight/300)), cropWidth, cropHeight);

  // canvas text
  const canvases = document.querySelectorAll("#classCanvases > div > canvas");
  let length = canvases.length;
  for(let i = 0; i < length; i++){
    let canvas =canvases[i].getContext("2d");
    canvas.textAlign = "center";
    canvas.font = "20px Arial";
    canvas.fillText("Click/Hold", 56, 62);

  }

  // chart initialisation
  lossChart = createMetricsChart("loss"); 
  accChart = createMetricsChart("accuracy");

  // settings initialisation
  document.getElementById("settings").lastElementChild.innerText = cThreshold;

  // hyperparameter range value initialisation
  document.getElementById("scrollSpeed").value = cThreshold;
  document.getElementById("numberOfEpochs").value = numOfEpochs;
  document.getElementById("learningRate").value = learningRate;
  document.getElementById("batchSize").value = batchSize;
  document.getElementById("numberOfUnits").value = numOfUnits;


  document.getElementById("modelHyperparameters").lastElementChild.innerText = numOfUnits;

  // displays the value in text next for each range value for each training hyperparameter
  const trainingParameters = document.getElementById("trainingHyperparameters");
  trainingParameters.children[0].lastElementChild.innerText = batchSize;
  trainingParameters.children[1].lastElementChild.innerText = numOfEpochs;
  trainingParameters.children[2].lastElementChild.innerText = learningRate;
 
  // displays main HTML content once finished with initialisation
  document.getElementById("loading").style.display = "none";
  document.getElementById("content").style.display = "block";
}


async function updateHTMLValues(node){
  node.nextElementSibling.innerText = node.value;
}


/* User options */
// updates scroll speed value based on input on HTML
async function setScrollSpeed(scrollRange){    
  if(scrollRange.value == 1){
    cThreshold = 0.99;
  }
  else{
    cThreshold = scrollRange.value; 
  } 
  
  document.getElementById("settings").lastElementChild.innerText = cThreshold;
}

// once called, all hyperparameters regarding training a changed according to values input in HTML
function setTrainingParameters(){
  numOfEpochs = parseInt(elementEpoch = document.getElementById("numberOfEpochs").value);
  learningRate = parseFloat(document.getElementById("learningRate").value);
  batchSize = parseInt(document.getElementById("batchSize").value);

  adamOp.learningRate = learningRate;

  alert("Training hyperparameters set");
} 


// sets new value for number of units that is specified in HTML and creates new model with this new hyperparameter value
function setModelParameters(){
  numOfUnits = parseInt(document.getElementById("numberOfUnits").value);
  disposeModel();
  createModel();
  document.getElementById("run").innerText = "Run";
  alert("new classifier created");
}

 // preprocesses captured image
async function preprocess(){
  let image = await cam.capture();
  let preprocessedImage = image;

  // disposes all tensors created within function unless specified to be kept
  tf.tidy(() => {
    //preprocessedImage = tf.sub(tf.div(preprocessedImage,127.5),1);
    preprocessedImage = tf.div(preprocessedImage, 255);

    // flips image if true
    if(flipImage){
      preprocessedImage = dataAugmentation(preprocessedImage).toInt();
    }

    // creates high contrast
    if(segmentImage){
      // grayscale image
      // fractional problem
      // grayscale. citation: https://stackoverflow.com/questions/50334200/how-to-transform-a-webcam-image-to-grayscale-using-tensorflowjs 
      // author: Yuri Malheiros date: May 22nd 2018
      //preprocessedImage = preprocessedImage.mean(2).expandDims(2);
      preprocessedImage = imageSegment(preprocessedImage);
    }

    // resizes image
    //preprocessedImage = tf.image.resizeBilinear(preprocessedImage, [32, 32], true);

    if(segmentImage){
      preprocessedImage = tf.keep(preprocessedImage.toFloat());
    }
    else{
      preprocessedImage = tf.keep(preprocessedImage); //.toInt());
    }
    /*
      // centering
      mean = preprocessedImage.mean();
      preprocessedImage = preprocessedImage.sub(mean);
    */  
  });

  image.dispose();

  return preprocessedImage;
} 

// normalisation techniques (appears imagenet does preprocessing already)
/*
function rescale(preprocessedImage){
  //downscale pixels to 0, 1
  return preprocessedImage.div(255);
}


function minmaxNorm(image1){
  image = image1.toFloat();
  const max = image.max();
  const min = image.min();

  return tf.div(tf.sub(image, min), tf.sub(max, min));
}

function zscoreNorm(preprocessedImage){
  //z-score normalisation for faster convergence
  const standardDeviation = tf.sqrt(tf.square(tf.sub(preprocessedImage, mean)).mean());
  return tf.div(tf.sub(preprocessedImage, mean), standardDeviation);
  
}
*/


// error filter
// checks if previous prediction was the same, stops it from being executed unless it is
// to deduct the amount of times a function is called due to a misprediction
async function isConsecutive(prediction){

  if(prevPredict == prediction){
    streak++;
  }
  else{
    streak = 0;
    /*
    error++;
    console.log(error);
    */

    //console.log("0");
  }

  prevPredict = prediction;
    
  if(streak == streakTarget){
    return true;
  }

  return false;

  }


// saves both weights and topology locally
async function exportModel(){
  // only allows the model to be exported if trained
  if(isTrained){
    console.log(await classifier.save('downloads://my-model-1'));
  }
  else{
    alert("train model before exporting");
  }
}


// changes current classifier model with a user chosen model stored that is stored locally
async function importModel(){
  // stores files that have been chosen as input
  const modelFiles = document.getElementById("modelFiles").files;

  // only attempts to import classifier model only if exactly two files are supplied
  if(modelFiles.length != 2){
    alert("Input one file for topology and a file for weights");
  }
  // if one file type is not json and the other is not of type 'bin' than the files provided are invalid for model import
  else if(modelFiles[0].name.slice(-5) != ".json" || modelFiles[1].name.slice(-4) != ".bin"){
    alert("Files need to be of type .json for topology and type .bin for weights");
  }
  else{
    // model is imported and compiled with new optimiser
    console.log(modelFiles[0], modelFiles[1]);
    isTrained = true;
    disposeModel();
    classifier = await tf.loadLayersModel(tf.io.browserFiles([modelFiles[0], modelFiles[1]]));
    adamOp = tf.train.adam(learningRate);
    classifier.compile({optimizer: adamOp, loss: "categoricalCrossentropy", metrics: "accuracy"});
  }

}


// releases model and optimiser memory once user has chosen to create or import a new model
// this prevents memory leak 
function disposeModel(){
  classifier.dispose();
  adamOp.dispose();
  // variables are reinitialised
  classifier;
  adamOp;
}


// deletes all training data and sets counters to 0 to state that there is 0 training data for each class
function clearData(){
  
  console.log(trainingDataset.data);
  console.log(tf.memory());
  trainingDataset.data.dispose();
  trainingDataset.data = null;
  trainingDataset.labels = [];
  console.log(tf.memory());
  console.log(trainingDataset);

  classesCounter = Array(numOfClasses).fill(0);

  //html values set to 0 // 
  const counterElements = document.querySelectorAll("#classCanvases > div > p");
  const length = counterElements.length;
  for(let i = 0; i < length; i++){
    counterElements[i].innerText = classes[i] + ": 0";
  }

}


//maybe lock this function after first capture or only allow on intial setup
// applies crop height and width to webcam input
async function setCrop(){
  stopRun();
  cam.stop();
  if(isTestCamOn){
    testPreprocess();
  }
  cropHeight = tempCropHeight;
  cropWidth = tempCropWidth;
  cam = await tf.data.webcam(document.getElementById("cam"), {resizeWidth: cropWidth, resizeHeight: cropHeight, centerCrop: true});
}


// changes height and width of the drawn rectangle for the crop canvas
function setCropWidth(widthInput){
  cropOverlay.clearRect(0,0,300,300);
  tempCropWidth = widthInput.value;
  cropOverlay.strokeRect(150 - (150*(tempCropWidth/300)), 150 - (150*(tempCropHeight/300)), tempCropWidth, tempCropHeight);
}


function setCropHeight(heightInput){
  cropOverlay.clearRect(0,0,300,300);
  tempCropHeight = heightInput.value;
  cropOverlay.strokeRect(150 - (150*(tempCropWidth/300)), 150 - (150*(tempCropHeight/300)), tempCropWidth, tempCropHeight);
}


// opens file called tips in a new tab to give advice on how to get the best experience out of the extension
function tips(){
  window.open("tips.txt");
}


// opens file called underTheHood in a new tab to inform user how the extension works
function underTheHood(){
  window.open("underTheHood.txt");
}


// toggles webcam stream on/off
async function toggleCam(){
  if(cam.stream.active){
    document.getElementById("camButton").innerText = "Start webcam stream";
    stopRun();
    if(isTestCamOn){
      testPreprocess();
    }
    cam.stop();
  }
  else{
    try{
      document.getElementById("camButton").innerText = "Stop webcam stream";
      cam = await tf.data.webcam(document.getElementById("cam"), {resizeWidth: cropWidth, resizeHeight: cropHeight, centerCrop: true});
    }
    catch(error){
      alert("Access to webcam reqiured");
    }
  }
}


// stops the run function
function stopRun(){
  document.getElementById("run").innerText = "Run";
  isRunning = false;
}


initialise();