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

var ductx = document.getElementById("DUChart").getContext('2d');
var DUChart = createDefaultChart(ductx, "DOWN-UP (Dwell Time)", 350);

var udctx = document.getElementById("UDChart").getContext('2d');
var UDChart = createDefaultChart(udctx, "UP-DOWN (Flight Time)", 350);

var ddctx = document.getElementById("DDChart").getContext('2d');
var DDChart = createDefaultChart(ddctx, "DOWN-DOWN (Speed)", 350);

var txtCanvas = document.getElementById("textEntry");
var txtctx = txtCanvas.getContext("2d");
txtctx.font = "170px verdana";

var dunnctx = document.getElementById("DUNNChart").getContext('2d');
var DUNNChart = createDefaultChart(dunnctx, "DOWN-UP (Dwell Time)", 350);

var udnnctx = document.getElementById("UDNNChart").getContext('2d');
var UDNNChart = createDefaultChart(udnnctx, "UP-DOWN (Flight Time)", 350);

var ddnnctx = document.getElementById("DDNNChart").getContext('2d');
var DDNNChart = createDefaultChart(ddnnctx, "DOWN-DOWN (Speed)", 350);

function updateMinMaxTrain() {
  var maxTrainError = maxsmd(DUseq, UDseq, DDseq);
  var minTrainError = minsmd(DUseq, UDseq, DDseq);

  document.getElementById("dutrain").innerHTML = Math.round(minTrainError[0]) + "-" + Math.round(maxTrainError[0]);
  document.getElementById("udtrain").innerHTML = Math.round(minTrainError[1]) + "-" + Math.round(maxTrainError[1]);
  document.getElementById("ddtrain").innerHTML = Math.round(minTrainError[2]) + "-" + Math.round(maxTrainError[2]);
}

function captureKeyEvent(e) {
	kevents.push(e);
	newevent = {event :  e.type, code : e.code, keycode :  e.keyCode, keytime : e.timeStamp };
	keyevents.push(newevent);
	if (e.key.length > 1) {
		if (e.key === "Enter" && e.type === "keyup") {
      if (testText === "") {
        testText = currText;
      }
      if (currText !== testText) {
        alert("You entered [" + currText + "], Please Enter ["+testText+"]");
  	    DU = [];
  	    UD = [];
  	    DD = [];

  	    lastdown = lastup = -1;
        txtctx.clearRect(0,0,txtCanvas.width,txtCanvas.height);
        txtctx.strokeText(testText,0,txtCanvas.height/2);
        currText = "";
        return;
      }
      txtctx.clearRect(0,0,txtCanvas.width,txtCanvas.height);
      txtctx.strokeText(testText,0,txtCanvas.height/2);
      currText = "";

      //console.log("DU :", scaledManhattanDist(DUseq,DU));
      //console.log("UD :", scaledManhattanDist(UDseq,UD));
      //console.log("DD :", scaledManhattanDist(DDseq,DD));

      var test = document.getElementById("testCheck").checked;

      if (test) {
        document.getElementById("dutest").innerHTML = Math.round(scaledManhattanDist(DUseq,DU));
        document.getElementById("udtest").innerHTML = Math.round(scaledManhattanDist(UDseq,UD));
        document.getElementById("ddtest").innerHTML = Math.round(scaledManhattanDist(DDseq,DD));

        if (DUChart.data.datasets.length > DUseq.length + 1) {
          removeLastFromChart(DUChart);
          removeLastFromChart(UDChart);
          removeLastFromChart(DDChart);
        }
        addDataToChart(DUChart, DU, 'rgba(0,0,255,0.5)', true);
        addDataToChart(UDChart, UD, 'rgba(0,0,255,0.5)', true);
        addDataToChart(DDChart, DD, 'rgba(0,0,255,0.5)', true);

			  DUtest.push(DU);
			  UDtest.push(UD);
			  DDtest.push(DD);

        if (typeof dunn == "undefined") {
          var l1 = Math.ceil(DU.length / 2);
          var l2 = Math.ceil(DU.length / 4);
          var options = { hiddenLayers : [l1,l2,l1] };

          dunn = new ML.FNN(options);
          dunn.activation = "exponential-elu"; //rectified linear unit (ReLU)
          dunn.train(DUseq, DUseq);

          udnn = new ML.FNN(options);
          udnn.activation = "exponential-elu";
          udnn.train(UDseq, UDseq);

          ddnn = new ML.FNN(options);
          ddnn.activation = "exponential-elu";
          ddnn.train(DDseq, DDseq);
        }

        DUNNChart.data.datasets = [];
        UDNNChart.data.datasets = [];
        DDNNChart.data.datasets = [];

        dupred = dunn.predict([DU]);
        udpred = udnn.predict([UD]);
        ddpred = ddnn.predict([DD]);

        addDataToChart(DUNNChart, dupred[0], 'rgba(0,255,0,0.5)', true);
        addDataToChart(UDNNChart, udpred[0], 'rgba(0,255,0,0.5)', true);
        addDataToChart(DDNNChart, ddpred[0], 'rgba(0,255,0,0.5)', true);

        addDataToChart(DUNNChart, DU, 'rgba(0,0,255,0.5)', true);
        addDataToChart(UDNNChart, UD, 'rgba(0,0,255,0.5)', true);
        addDataToChart(DDNNChart, DD, 'rgba(0,0,255,0.5)', true);

			  DU = [];
			  UD = [];
			  DD = [];

			  lastdown = lastup = -1;
        return;
      } else {
        document.getElementById("dutrain").innerHTML = Math.round(scaledManhattanDist(DUseq,DU));
        document.getElementById("udtrain").innerHTML = Math.round(scaledManhattanDist(UDseq,UD));
        document.getElementById("ddtrain").innerHTML = Math.round(scaledManhattanDist(DDseq,DD));
        document.getElementById("count").innerHTML = DUseq.length;
      }

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

			DU = [];
			UD = [];
			DD = [];

			lastdown = lastup = -1;
		}
		return;
	}

	if ( e.type ===  "keydown" ) {
    currText += e.key;
    txtctx.fillText(currText,0,txtCanvas.height/2);
		keydownevents.push(newevent);
		if (lastdown >= 0) {
			DD.push(e.timeStamp - lastdown);
		}
		if (lastup >= 0) {
			UD.push(e.timeStamp - lastup);
		}
		lastdown = e.timeStamp;
	} else {
		if (lastdown >= 0) {
			DU.push(e.timeStamp - lastdown);
		}
		lastup = e.timeStamp;
		keyupevents.push(newevent);
	};
}

document.onkeydown = captureKeyEvent;
document.onkeyup = captureKeyEvent;
