  //automatic runs when window load
  window.onload = function() {   

    // assign global variables pointing to the "definitive" (img) upper canvas
    canvaso = document.getElementById('imageView');
    contexto = canvaso.getContext("2d");

    // use current time to define the name of the results (log) file (for example Results_{ID}.txt)
    var today = new Date();
    var date_time = today.getFullYear()+'_'+(today.getMonth()+1)+'_'+today.getDate() + '_'+ today.getHours() + "_" + today.getMinutes() + "_" + today.getSeconds();    
    fileID = date_time;

    // initialize mouse cursor as pencil by default
    document.body.style.cursor = "url('/static/images/pencil"+document.getElementById('dsize').value+".png') 0 0, default";    
    // AJAX call to loadlist() in views.py to get list of images, gt, bboxes, categories 
    $.ajax({
      url: 'loadlist/',
      type: 'POST',
      tryCount : 0,
      retryLimit : 3,      
      success: function(resp){
          username = resp.username;

          var lines = resp.imgList;
          var linesGT = resp.gtList;
          var linesBB = resp.bboxList;
          var linesCls = resp.clsList;
          var linesIds = resp.idsList;
          // populate corresponding arrays with info loaded from the .txt files
          for (var j = 0, len = lines.length; j < len; j++) {
            imgArray[j] = 'https://drive.google.com/uc?id=' + lines[j];
            gtArray[j] = 'https://drive.google.com/uc?id=' + linesGT[j];
            bbArray[j] = 'https://drive.google.com/uc?id=' + linesBB[j];

            //convert from string to int array, numbers separated by comma
            var intCls = linesCls[j].split(",").map(Number); 

            // push each element into array 
            clsArray.push(intCls) 
          }
          
          // same for the list of categories
          var lines = resp.catsList;            
          for (var j = 0, len = lines.length; j < len; j++) {
            catArray[j] = lines[j];
          }

          // same for the randomly shuffled list of image IDs
          var lines = resp.idsList;        
          listIDs = new Array(imgArray.length);    
          for (var j = 0, len = lines.length; j < len; j++) {
            listIDs[j] = parseInt(lines[j]);
          }

          // set i with index of next image to be loaded
          i = parseInt(resp.nextId);
          // set variable containing current total score
          scoreTotal = parseInt(resp.scoreTotal)
          // average score per image
          avgScore = scoreTotal/(i+1)

          // get current ranking
          linesUsrs = resp.rankusers_;
          linesScores = resp.rankscores_;
          // how many draws user has done 
          numberOfImg = resp.rankimgs_;
          ranktotal = resp.ranktotal_;
          
          var str = '<b>Avg. per image (total / no. annotated images)</b><br>'     
          len = linesScores.length

          if (len > 5)
            len = 5;

          for (var j = 0; j < len; j++) {            
            str = str + linesUsrs[j] + ": " + parseInt(linesScores[j]) + "      (" + parseInt(ranktotal[j]) + "/" + parseInt(numberOfImg[j]) + ") <br> ";
          }
          document.getElementById("listAvgScoresId").innerHTML = str;

          init();
          lookupInit();
          loadImage();
      },
      error : function(xhr, textStatus, errorThrown ) {
          // if (textStatus == 'timeout') {
          this.tryCount++;
          if (this.tryCount <= this.retryLimit) {
              //try again
              $.ajax(this);
              return;
          }     
          else{
            alert("Server error, image will be reloaded. That will not affect your records, sorry about the inconvenience.")
            window.location.reload(); //will automatically refresh until the end
            return;
          }                 
      },
      timeout: 10000
    }); 

  }

  function toggleGTuser () {

    var GTmask = '/static/'+username+'/GTimage'+listIDs[i]+'.png'

    if(GTusr > 0){
      document.getElementById("maskImg").src = lastUsrMask
      GTusr = 0;
    }
    else{
      document.getElementById("maskImg").src = GTmask
      GTusr = 1;
    }
    
      // wait for the mask to load
      document.getElementById("maskImg").onload = function(){
      changeImgTran()    
      //without our calling, do nothing
      document.getElementById("maskImg").onload = function(){}
    }
  }

  function skip() {    
    if (skipCnt < 3){
      r = confirm("You have " + (3-skipCnt) + " skips available. Are you sure you want skip this image?")
      if (r == true){
        skipCnt = skipCnt + 1;
        alert("Skipping this image, you can skip " + (3-skipCnt)+ " more images");
        finish();
      }
    }
    else{
      alert("You've already skipped 3 images. Make sure you are trying hard, ok?");
      skipCnt = skipCnt + 1;
      finish();
    }

  }

//function handled with showing next image in the list
  function finish()
  {
    // check if there are further images available
    if (i<imgArray.length)
    {
      // get time interval that was used for this block
      end(); 

      // add score obtained for this image to the total of scores so far
      scoreTotal += img_score;
      // average score per image
      avgScore = scoreTotal/(i+2)      

      // update log file
      writeFile(); 

      // clear time process bar 
      // pause the bar
      clearInterval(intervalId);      

      // update the HTML txt component with the string of accuracies
      document.getElementById("sumId").innerHTML = "Total: " + scoreTotal;

      // hide traces
      TraceOnTempCanvas();

    }

    trace = [];

    img_score = 0;
    // document.getElementById("scoreId").textContent = "Total Score: " ;
    
  }  

  function createAgainList(){
      var cls_ = clsArray[listIDs[i]]; //number of existed elements in current image

      // remove rest elements from current array
      for (var ii = 2; ii < 22; ii++)
      {
        
        var element = "opt" + ii    // string
        //alert(element)

        var option_ = document.getElementById(element);
        var parent_ = document.getElementById("dcolor");
        
        if(option_ != null)
          parent_.removeChild(option_);        
      }
      
      // remove rest elements from current array
      //Create array of options to be added
      var array = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"];

      var mySelect = document.getElementById("dcolor");

      //Create and append the options
      for (var k = 0; k < cls_.length; k++) {
        var option = document.createElement("option");
        //set value as first, 1,2,3....21
        var j = cls_[k]+1;
        option.value = j;
        option.id = "opt" + j;
        //alert(option.id);
        option.text = array[cls_[k]]; //actually,0 is background; we need to change its value to 1
        mySelect.appendChild(option);
      }          
  }


  function nextImage(){
    
    // check if there are further images available
    if (i<9+skipCnt){
      document.getElementById("btnNxtImg").style.visibility = "hidden"
      document.getElementById("btnGTusr").style.visibility = "hidden"
      document.getElementById("btnFinish").style.visibility = "visible"
      document.getElementById("btnReload").style.visibility = "visible"
      document.getElementById("btnPause").style.visibility = "visible"
      document.getElementById("btnRefine").style.visibility = "visible"
      TraceOnTempCanvas();
      
      deleteBars();

      // load the next image (i+1)
      i +=1;      
      loadImage()

      // restart stopwatch and trace/refine counters
      pauseInterval = 0
      // start();
      trace_number=0;
      callCnt=0;

      trace = [];

      img_score = 0;
      document.getElementById("scoreId").textContent = "Score: " + img_score;
    }
    else{
      alert("Phase 1 completed! \n"+
        "You goal now is to get masks that fit all visible flowers as tightly as possible, without any score being shown to you. \n"+
        "WARNING: If your avg. accuracy for the last 4 images is not higher than 80%, you will have to redo these 4 images.\n"+
        "Please pay close attention at this quick video with examples (there is a pause/play button at the bottom left). ")
      window.location.href = '/freelabel/playFlowers/';
    }

  }

  //function handled with load image 
  function loadImage()
  {
      // clear array of accuracies
      accs = [];

      // delete scorebars
      // deleteBars();

      createAgainList();
      // reset background scorebar
      document.getElementById("bar0").style.width = "0%";

      // clear interval
      if(intervalId)
        clearInterval(intervalId)

      // clear text of total score
      document.getElementById("scoreId").textContent = 'Score: ' + img_score;
      document.getElementById("sumId").innerHTML = "Total: " + scoreTotal;

      // save current cursor (pencil/erase) and then update to loading symbol
      var currentCursor = document.body.style.cursor
      document.body.style.cursor = 'wait';

       // clear temporary (traces) upper canvas
      temp_context.clearRect(0, 0, temp_canvas.width, temp_canvas.height);

      // clear the text with accuracies shown right next to upper canvas
      document.getElementById("accId").innerHTML = '';

      // reset accuracy flag to < 90%
      toggleIcon(false);

      // clear URL of mask image on bottom canvas
      document.getElementById("maskImg").src = '#';

      // get image URL according to index i and the randomly permuted list       
      var pic = imgArray[listIDs[i]]

      // set image URL to image element on bottom canvas
      document.getElementById("initial").src = pic.replace();
      var img = document.getElementById("initial");

      // get selected transparency for image 
      var tran = document.getElementById("dtran").value

      // wait for the image to load
      img.onload = function(){

          // display image ID in case the user wants to provide some feedback
          document.getElementById("imgId").innerHTML = "ID: " + listIDs[i] + "("+ (i+1) +"/"+ (10+skipCnt) +")";

          // get the dimensions of current image to global variables
          currentHeight = img.clientHeight;
          currentWidth = img.clientWidth;
      
          // set image upper canvas dimensions accordingly
          canvaso.height = currentHeight;
          canvaso.width = currentWidth;

          // draw image on upper canvas with corresponding opacity
          contexto.clearRect(0,0,currentWidth,currentHeight);
          contexto.globalAlpha = tran;      
          contexto.drawImage(img, 0, 0, currentWidth, currentHeight);  
          contexto.globalAlpha = 1;

          // set tracews upper canvas dimensions accordingly
          temp_canvas.height = currentHeight;
          temp_canvas.width = currentWidth;

          // pass original image resolution to python
          var img_size = [img.naturalHeight, img.naturalWidth];
          
          // load bounding box list
          var bound = bbArray[listIDs[i]];

          // AJAX call to initanns() in views.py, which initializes the array 
          // that will contain the traces provided by the user for this image        
          $.ajax({
            url: '/freelabel/initanns/',
            type: 'POST',
            data: {"img_size": img_size,"BB": bound},
            tryCount : 0,
            retryLimit : 3,
            success: function(data) {           
              // resets the array of bbox for current image
              bb = []; 

              // process the list of bboxes
              var objects = 'List for objects: ' + '<br/>' + 'background';
              var linesbounding = data.bbList;                      

              for (var i = 0, len = linesbounding.length; i < len; i++) {                  
                //convert from string to string array include comma
                var intbounding = linesbounding[i].split(",").map(String);   //string object

                objects = objects + '</br>' + intbounding[4];
              
                document.getElementById("accId").innerHTML = objects;

                //convert from string to int array, numbers separated by comma
                var intbounding = linesbounding[i].split(",").map(Number); 

                // push each bbox into array bb of bounding boxes
                bb.push(intbounding) 

                // build rectangle with corresponding bbox's coordinates
                contexto.beginPath();
                contexto.rect(intbounding[0], intbounding[1], intbounding[2], intbounding[3]);
                contexto.lineWidth = 2; // thickness of 2 pixels

                colorArray[i] = getRandomColor(); // random color
                contexto.strokeStyle = colorArray[i]; // save this color
                // draw the bbox on canvas
                contexto.stroke();           
              }
              // create one percentage bar for each class present on GT
              createBars();

              // initialize scores array, adding 1 to include bkg
              scores = new Array(clsArray.length+1);

              document.body.style.cursor = currentCursor              

              // number of objects in image corresponds to number of bboxes
              var noObjs = linesbounding.length;              
              startTimebar(noObjs);
              start();
          },
          error : function(xhr, textStatus, errorThrown ) {
              this.tryCount++;
              if (this.tryCount <= this.retryLimit) {
                  //try again
                  $.ajax(this);
                  return;
              }     
              else{
                alert("Server error, image will be reloaded. That will not affect your records, sorry about the inconvenience.")
                window.location.reload(); //will automatically refresh until the end
                return;
              }                 
          },
          timeout: 5000
        });   
    }     
  }

  function writeFile(){
      
      //include maxtime, time bonus, score per class, total score
      $.ajax({
          url: '/freelabel/writeLog/',
          type: 'POST',
          data: {"next_i": i+1,
                 "scoreTotal": scoreTotal,
                 "id_image":listIDs[i], 
                 "time": seconds, 
                 "maxTime": maxTime,
                 "trace_number": trace_number, 
                 "refine_number": callCnt, 
                 "accuracies": accs, 
                 "scores": scores,
                 "timeBonus": timeBonus.toFixed(2),
                 "finalScore": img_score,
                 "fileID": fileID},
          tryCount : 0,
          retryLimit : 3,
          success: function(resp){    
            // replace the mask on bottom canvas with new segmentation mask
            var pic = '/static/'+username+'/GTimage'+listIDs[i]+'.png'
            document.getElementById("maskImg").src = pic.replace();                  
            
            // wait for the mask to load
            document.getElementById("maskImg").onload = function(){
              changeImgTran()

              hideButtons()      

              // get current ranking
              linesUsrs = resp.rankusers_;
              linesScores = resp.rankscores_;
              // how many draws user has done 
              numberOfImg = resp.rankimgs_;    //how many images have done drawing in total
              ranktotal = resp.ranktotal_;     //total score until now              
              var str = '<b>Avg. per image (score / no. annotated images)</b><br>'     

              len = linesScores.length

              if (len > 5)
                len = 5;

              for (var j = 0; j < len; j++) {            
                str = str + linesUsrs[j] + ": " + parseInt(linesScores[j]) + "      (" + parseInt(ranktotal[j]) + "/" + parseInt(numberOfImg[j]) + ") <br> ";
              }
              document.getElementById("listAvgScoresId").innerHTML = str;    

              //without our calling, do nothing
              document.getElementById("maskImg").onload = function(){}            

            }    
          },
          error : function(xhr, textStatus, errorThrown ) {
              // if (textStatus == 'timeout') {
              this.tryCount++;
              if (this.tryCount <= this.retryLimit) {
                  //try again
                  $.ajax(this);
                  return;
              }     
              else{
                alert("Server error, image will be reloaded. That will not affect your records, sorry about the inconvenience.")
                window.location.reload(); //will automatically refresh until the end
                return;
              }                 
          },
          timeout: 10000
      });  
    } 

