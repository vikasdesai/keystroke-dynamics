var kevents = [];
var keyevents = [];
var keyupevents = [];
var keydownevents = [];
var UD = [];
var DU = [];
var DD = [];
var lastdown = -1;
var lastup = -1;
var UDseq = [];
var DUseq = [];
var DDseq = [];
var UDtest = [];
var DUtest = [];
var DDtest = [];
var testText = "";
var currText = "";

function createDefaultChart( ctx, title, maxy) {
  return new Chart(ctx, {
    type: 'line',
    data: {
        labels: [0,1,2,3,4,5,6,7,8,9,10,11,12],
        datasets: []
    },
    options: {
        title: {
            display: true,
            text: title
        },
        legend: {
            display: false
        },
        scales: {
            yAxes: [{
                ticks: {
                    //beginAtZero:true
                    min: 0,
                    max: maxy
                }
            }]
        }
    }
  });
}

function createScatterChart( ctx, title) {
  return new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: []
    },
    options: {
      title: {
          display: true,
          text: title
      },
      legend: {
          display: false
      },
      scales: {
          xAxes: [{
              ticks: {
                  beginAtZero:true
              }
          }],
          yAxes: [{
              ticks: {
                  beginAtZero:true
              }
          }]
      }
    }
  });
}

function createNN(hiddenLayers, activation) {
  var options = {};
  options.hiddenLayers = hiddenLayers;
  options.activation = activation;
  nn = new ML.FNN(options);
  return nn;
}

function meanPath(data) {
  var columns = math.transpose(data);
  var meanp = columns.map(function(row) {
    return math.mean(row);
  });
  return meanp;
}

function avgAbsDevPath(data) {
  var columns = math.transpose(data);

  return columns.map(function(row) {
    return math.mad(row);
  });
}

function smd_nn(data, path, nnpred) {
  var avgAbsDev = avgAbsDevPath(data);

  return nnpred.reduce(function(sum, val, i) {
    return sum + Math.abs(path[i] - val) / avgAbsDev[i];
  }, 0);
}

function smd(data, path) {
  var avgAbsDev = avgAbsDevPath(data);
  var meanp = meanPath(data);

  return meanp.reduce(function(sum, val, i) {
    return sum + Math.abs(path[i] - val) / avgAbsDev[i];
  }, 0);
}

function smd_for_path(path, index, allpaths) {
  var prepath = allpaths.slice(0,index);
  var postpath = allpaths.slice(index + 1,allpaths.length);

  return smd(prepath.concat(postpath), path);
}

function maxsmd(duseq, udseq, ddseq) {
  var maxdu = math.max(duseq.map(smd_for_path));
  var maxud = math.max(udseq.map(smd_for_path));
  var maxdd = math.max(ddseq.map(smd_for_path));

  return [maxdu,maxud,maxdd];
}

function minsmd(duseq, udseq, ddseq) {
  var mindu = math.min(duseq.map(smd_for_path));
  var minud = math.min(udseq.map(smd_for_path));
  var mindd = math.min(ddseq.map(smd_for_path));

  return [mindu,minud,mindd];
}

//Same as smd above - duplicated because of refactoring can be replace with smd
function scaledManhattanDist(x,y) {
  var columns = math.transpose(x);
  var avgAbsDev = columns.map(function(row){
    return math.mad(row);
  });

  var meanp = meanPath(x);
  return meanp.reduce(function(sum, val, i){
    return sum + Math.abs(y[i] - val) / avgAbsDev[i];
  }, 0);
}

function removeLastFromChart(myChart) {
  myChart.data.datasets.pop();
  myChart.update();
}

function addDataToChart(myChart, newdata, color, showline) {
  myChart.data.datasets.push({
    label: "",
    data: newdata,
    backgroundColor: color,
    borderColor: color,
    fill: false,
    pointRadius: 3,
    pointHoverRadius: 9,
    showLine: showline
  });
  myChart.update();
}

async function startTest() {
  if (document.getElementById("testCheck").checked) {
    var l = DUseq[0].length;
    if (l > 0) {
      var l1 = Math.ceil(l / 2);
      var l2 = Math.ceil(l1 / 2);
      //var options = { hiddenLayers : [l1,l2,l1] };
      var options = { hiddenLayers : [l1] };

      dunn = new ML.FNN(options);
      dunn.activation = "exponential-elu"; //rectified linear unit (ReLU)
      dunn.train(DUseq, DUseq);

      udnn = new ML.FNN(options);
      udnn.activation = "exponential-elu";
      udnn.train(UDseq, UDseq);

      ddnn = new ML.FNN(options);
      ddnn.activation = "exponential-elu";
      ddnn.train(DDseq, DDseq);

      //Neuro-Evolutionary Network - neataptic.js
      dunedata = DUseq.map(function(data) { return {input:data, output:data};});
      udnedata = UDseq.map(function(data) { return {input:data, output:data};});
      ddnedata = DDseq.map(function(data) { return {input:data, output:data};});

      dunenn = neataptic.architect.Perceptron( l, l1, l);
      udnenn = neataptic.architect.Perceptron( l-1, l1, l-1);
      ddnenn = neataptic.architect.Perceptron( l-1, l1, l-1);

      neopts = {
        mutation: neataptic.methods.mutation.FFW,
        equal: true,
        popsize: 100,
        elitism: 10,
        log: 100,
        error: 100, //0.03,
        iterations: 1000,
        mutationRate: 0.5
      };

      neDone = await Promise.all([
        dunenn.evolve(dunedata, neopts),
        udnenn.evolve(udnedata, neopts),
        ddnenn.evolve(ddnedata, neopts)
      ]);
      console.log(neDone);
    }
  }
}

function initCharts() {
  ductx = document.getElementById("DUChart").getContext('2d');
  DUChart = createDefaultChart(ductx, "DOWN-UP (Dwell Time)", 350);

  udctx = document.getElementById("UDChart").getContext('2d');
  UDChart = createDefaultChart(udctx, "UP-DOWN (Flight Time)", 350);

  ddctx = document.getElementById("DDChart").getContext('2d');
  DDChart = createDefaultChart(ddctx, "DOWN-DOWN (Speed)", 350);

  txtCanvas = document.getElementById("textEntry");
  txtctx = txtCanvas.getContext("2d");
  txtctx.font = "150px verdana";

  dunnctx = document.getElementById("DUNNChart").getContext('2d');
  DUNNChart = createDefaultChart(dunnctx, "DOWN-UP (Dwell Time)", 350);

  udnnctx = document.getElementById("UDNNChart").getContext('2d');
  UDNNChart = createDefaultChart(udnnctx, "UP-DOWN (Flight Time)", 350);

  ddnnctx = document.getElementById("DDNNChart").getContext('2d');
  DDNNChart = createDefaultChart(ddnnctx, "DOWN-DOWN (Speed)", 350);

  dunectx = document.getElementById("DUNEChart").getContext('2d');
  DUNEChart = createDefaultChart(dunectx, "DOWN-UP (Dwell Time)", 350);

  udnectx = document.getElementById("UDNEChart").getContext('2d');
  UDNEChart = createDefaultChart(udnectx, "UP-DOWN (Flight Time)", 350);

  ddnectx = document.getElementById("DDNEChart").getContext('2d');
  DDNEChart = createDefaultChart(ddnectx, "DOWN-DOWN (Speed)", 350);

  duudctx = document.getElementById("DU-UD").getContext('2d');
  DUUDChart = createScatterChart(duudctx, "DU-UD");

  udddctx = document.getElementById("UD-DD").getContext('2d');
  UDDDChart = createScatterChart(udddctx, "UD-DD");

  ddductx = document.getElementById("DD-DU").getContext('2d');
  DDDUChart = createScatterChart(ddductx, "DD-DU");
}

function showNN() {
  document.getElementById("nnheader").style.visibility = "visible";
  document.getElementById("nntable").style.visibility = "visible";

  if (document.getElementById("testCheck").checked == false) {
    document.getElementById("nnpredict").checked = false;
    alert("Please enable Test for this to work");
  }
}

function showNE() {
  if (document.getElementById("testCheck").checked == false) {
    document.getElementById("nepredict").checked = false;
    alert("Please enable Test for this to work");
    return;
  }
  if (typeof neDone == "undefined") {
    alert("Sorry Still Evolving!!!");
    document.getElementById("nepredict").checked = false;
  } else {
  document.getElementById("neheader").style.visibility = "visible";
  document.getElementById("netable").style.visibility = "visible";
  }
}

function updateMinMaxTrain() {
  var maxTrainError = maxsmd(DUseq, UDseq, DDseq);
  var minTrainError = minsmd(DUseq, UDseq, DDseq);

  document.getElementById("dutrain").innerHTML = Math.round(minTrainError[0]) + "-" + Math.round(maxTrainError[0]);
  document.getElementById("udtrain").innerHTML = Math.round(minTrainError[1]) + "-" + Math.round(maxTrainError[1]);
  document.getElementById("ddtrain").innerHTML = Math.round(minTrainError[2]) + "-" + Math.round(maxTrainError[2]);
}

function captureKeyEvent(e) {
	kevents.push(e);
	//newevent = {event :  e.type, code : e.code, keycode :  e.keyCode, keytime : e.timeStamp };
	//keyevents.push(newevent);
	if (e.key.length > 1) {
    if (e.key === "Backspace") {
      txtctx.clearRect(0,0,txtCanvas.width,txtCanvas.height);
      txtctx.strokeText(testText,0,txtCanvas.height/2);
      currText = "";

      DU = [];
      UD = [];
      DD = [];
      return
    }
		if (e.key === "Enter" && e.type === "keyup") {
      if (UD.length != (DU.length - 1) || DD.length != (DU.length - 1)) {
        console.log(DU,UD,DD);
        alert("Please type slowly!!!");
        console.log(currText);
        txtctx.clearRect(0,0,txtCanvas.width,txtCanvas.height);
        txtctx.strokeText(testText,0,txtCanvas.height/2);
        currText = "";

        DU = [];
        UD = [];
        DD = [];
  	    lastdown = lastup = -1;
        return
      }

      if (testText === "") {
        testText = currText;
      }

      txtctx.clearRect(0,0,txtCanvas.width,txtCanvas.height);
      txtctx.strokeText(testText,0,txtCanvas.height/2);

      if (currText !== testText) {
        alert("You entered [" + currText + "], Please Enter ["+testText+"]");
  	    DU = [];
  	    UD = [];
  	    DD = [];

        currText = "";
  	    lastdown = lastup = -1;
        return;
      }

      currText = "";

      //console.log("DU :", scaledManhattanDist(DUseq,DU));
      //console.log("UD :", scaledManhattanDist(UDseq,UD));
      //console.log("DD :", scaledManhattanDist(DDseq,DD));

      duavg = math.mean(DU);
      udavg = math.mean(UD);
      ddavg = math.mean(DD);

      var test = document.getElementById("testCheck").checked;

      if (test) {
        document.getElementById("dutest").innerHTML = Math.round(scaledManhattanDist(DUseq,DU));
        document.getElementById("udtest").innerHTML = Math.round(scaledManhattanDist(UDseq,UD));
        document.getElementById("ddtest").innerHTML = Math.round(scaledManhattanDist(DDseq,DD));

        if (DUChart.data.datasets.length > DUseq.length + 1) {
          removeLastFromChart(DUChart);
          removeLastFromChart(UDChart);
          removeLastFromChart(DDChart);

          removeLastFromChart(DUUDChart);
          removeLastFromChart(UDDDChart);
          removeLastFromChart(DDDUChart);
        }
        addDataToChart(DUChart, DU, 'rgba(0,0,255,0.5)', true);
        addDataToChart(UDChart, UD, 'rgba(0,0,255,0.5)', true);
        addDataToChart(DDChart, DD, 'rgba(0,0,255,0.5)', true);

        addDataToChart(DUUDChart, [{ x: Math.round(duavg), y: Math.round(udavg) }], 'rgba(0,0,255,1)', false);
        addDataToChart(UDDDChart, [{ x: Math.round(udavg), y: Math.round(ddavg) }], 'rgba(0,0,255,1)', false);
        addDataToChart(DDDUChart, [{ x: Math.round(ddavg), y: Math.round(duavg) }], 'rgba(0,0,255,1)', false);

			  DUtest.push(DU);
			  UDtest.push(UD);
			  DDtest.push(DD);

        if (typeof dunn == "undefined") {
          console.log("No Neural Network create - Failure!!!");
        }

        //Calculate and Display Neural Network predictions
        if (document.getElementById("nnpredict").checked) {
          DUNNChart.data.datasets = [];
          UDNNChart.data.datasets = [];
          DDNNChart.data.datasets = [];

          dupred = dunn.predict([DU]);
          udpred = udnn.predict([UD]);
          ddpred = ddnn.predict([DD]);

          document.getElementById("dunnpred").innerHTML = Math.round(smd_nn(DUseq, DU, dupred[0]));
          document.getElementById("udnnpred").innerHTML = Math.round(smd_nn(UDseq, UD, udpred[0]));
          document.getElementById("ddnnpred").innerHTML = Math.round(smd_nn(DDseq, DD, ddpred[0]));

          document.getElementById("dunnsmd").innerHTML = document.getElementById("dutest").innerHTML;
          document.getElementById("udnnsmd").innerHTML = document.getElementById("udtest").innerHTML;
          document.getElementById("ddnnsmd").innerHTML = document.getElementById("ddtest").innerHTML;

          addDataToChart(DUNNChart, dupred[0], 'rgba(0,255,0,0.5)', true);
          addDataToChart(UDNNChart, udpred[0], 'rgba(0,255,0,0.5)', true);
          addDataToChart(DDNNChart, ddpred[0], 'rgba(0,255,0,0.5)', true);

          addDataToChart(DUNNChart, DU, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UDNNChart, UD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(DDNNChart, DD, 'rgba(0,0,255,0.5)', true);

          //addDataToChart(DUNNChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(UDNNChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(DDNNChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
        }

        //Calculate and Display Neuro-Evolutionary Network predictions
        if (document.getElementById("nepredict").checked) {
          DUNEChart.data.datasets = [];
          UDNEChart.data.datasets = [];
          DDNEChart.data.datasets = [];

          dunepred = dunenn.activate(DU);
          udnepred = udnenn.activate(UD);
          ddnepred = ddnenn.activate(DD);

          document.getElementById("dunepred").innerHTML = Math.round(smd_nn(DUseq, DU, dunepred));
          document.getElementById("udnepred").innerHTML = Math.round(smd_nn(UDseq, UD, udnepred));
          document.getElementById("ddnepred").innerHTML = Math.round(smd_nn(DDseq, DD, ddnepred));

          document.getElementById("dunnpred_").innerHTML = document.getElementById("dunnpred").innerHTML;
          document.getElementById("udnnpred_").innerHTML = document.getElementById("udnnpred").innerHTML;
          document.getElementById("ddnnpred_").innerHTML = document.getElementById("ddnnpred").innerHTML;

          document.getElementById("dunnsmd_").innerHTML = document.getElementById("dutest").innerHTML;
          document.getElementById("udnnsmd_").innerHTML = document.getElementById("udtest").innerHTML;
          document.getElementById("ddnnsmd_").innerHTML = document.getElementById("ddtest").innerHTML;

          addDataToChart(DUNEChart, dunepred, 'rgba(0,0,0,0.5)', true);
          addDataToChart(UDNEChart, udnepred, 'rgba(0,0,0,0.5)', true);
          addDataToChart(DDNEChart, ddnepred, 'rgba(0,0,0,0.5)', true);

          addDataToChart(DUNEChart, DU, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UDNEChart, UD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(DDNEChart, DD, 'rgba(0,0,255,0.5)', true);

          //addDataToChart(DUNEChart, dupred[0], 'rgba(0,255,0,0.5)', true);
          //addDataToChart(UDNEChart, udpred[0], 'rgba(0,255,0,0.5)', true);
          //addDataToChart(DDNEChart, ddpred[0], 'rgba(0,255,0,0.5)', true);

          //addDataToChart(DUNEChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(UDNEChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(DDNEChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
        }

			  DU = [];
			  UD = [];
			  DD = [];

			  lastdown = lastup = -1;
        return;
      }

      document.getElementById("dutrain").innerHTML = Math.round(scaledManhattanDist(DUseq,DU));
      document.getElementById("udtrain").innerHTML = Math.round(scaledManhattanDist(UDseq,UD));
      document.getElementById("ddtrain").innerHTML = Math.round(scaledManhattanDist(DDseq,DD));
      document.getElementById("count").innerHTML = DUseq.length;

			DUseq.push(DU);
			UDseq.push(UD);
			DDseq.push(DD);

      if (DUChart.data.datasets.length > 1) {
        removeLastFromChart(DUChart);
        removeLastFromChart(UDChart);
        removeLastFromChart(DDChart);
      }

      addDataToChart(DUChart, DU, 'rgba(255,0,0,1)', false);
      addDataToChart(UDChart, UD, 'rgba(255,0,0,1)', false);
      addDataToChart(DDChart, DD, 'rgba(255,0,0,1)', false);

      addDataToChart(DUChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
      addDataToChart(UDChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
      addDataToChart(DDChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);

      addDataToChart(DUUDChart, [{ x: duavg, y: udavg }], 'rgba(255,0,0,1)', false);
      addDataToChart(UDDDChart, [{ x: udavg, y: ddavg }], 'rgba(255,0,0,1)', false);
      addDataToChart(DDDUChart, [{ x: ddavg, y: duavg }], 'rgba(255,0,0,1)', false);

			DU = [];
			UD = [];
			DD = [];

			lastdown = lastup = -1;
		}
		return;
	}

	if ( e.type ===  "keydown" ) {
    currText += e.key;
		if (lastdown >= 0) {
			DD.push(e.timeStamp - lastdown);
		}
		if (lastup >= 0) {
			UD.push(e.timeStamp - lastup);
		}
		lastdown = e.timeStamp;
    txtctx.fillText(currText,0,txtCanvas.height/2);
		//keydownevents.push(newevent);
	} else {
		if (lastdown >= 0) {
			DU.push(e.timeStamp - lastdown);
		}
		lastup = e.timeStamp;
		//keyupevents.push(newevent);
	};
}

window.onload = initCharts;
document.onkeydown = captureKeyEvent;
document.onkeyup = captureKeyEvent;
