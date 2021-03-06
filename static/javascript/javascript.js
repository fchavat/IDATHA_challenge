// Este código fue extraído de la documentación de MDN.

(function() {
    // The width and height of the captured photo. We will set the
    // width to the value defined here, but the height will be
    // calculated based on the aspect ratio of the input stream.

    var width = 320;    // We will scale the photo width to this
    var height = 320;     // This will be computed based on the input stream

    // |streaming| indicates whether or not we're currently streaming
    // video from the camera. Obviously, we start at false.

    var streaming = false;

    // The various HTML elements we need to configure or control. These
    // will be set by the startup() function.

    var video = null;
    var canvas = null;
    var photo = null;
    var startbutton = null;

    function startup() {
      video = document.getElementById('video');
      canvas = document.getElementById('canvas');
      photo = document.getElementById('photo');
      startbutton = document.getElementById('submit-from-webcam');

      navigator.mediaDevices.getUserMedia({video: true, audio: false})
      .then(function(stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function(err) {
        console.log("An error occurred: " + err);
      });

      video.addEventListener('canplay', function(ev){
        if (!streaming) {
          height = video.videoHeight / (video.videoWidth/width);

          // Firefox currently has a bug where the height can't be read from
          // the video, so we will make assumptions if this happens.

          if (isNaN(height)) {
            height = width / (4/3);
          }

          video.setAttribute('width', width);
          video.setAttribute('height', height);
          canvas.setAttribute('width', width);
          canvas.setAttribute('height', height);
          streaming = true;
        }
      }, false);

      startbutton.addEventListener('click', function(ev){
        takepicture();
        $("#file-form").submit();
      }, false);
    }

    function takepicture() {
      var context = canvas.getContext('2d');
      if (width && height) {
        canvas.width = width;
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);

        var data = canvas.toDataURL('image/jpeg');
        var item_image = data.replace("data:image/jpeg;base64,", "");
        $("#file-input").val(item_image);
      }
    }

    // Set up our event listener to run the startup process
    // once loading is complete.
    window.addEventListener('load', startup, false);
})();

$("#submit-from-filepicker").click(function(ev){
  var fileReader = new FileReader();
  var input = $("#inputfile")[0];
  console.log(input);
  fileReader.readAsDataURL(inputfile.files[0]);
  fileReader.addEventListener("load", function () {
    var data = fileReader.result.replace("data:image/jpeg;base64,", "");
    $("#file-input").val(data);
    $("#file-form").submit();
  }, false);
})
$('#file-form').ajaxForm(function(data) {
  var res = $("<div class='col-12'>"+data+"</div>")
  $( ".results" ).prepend( res );
});
