(function() {
  let canvas = document.getElementById('canvas');
  let ctx = canvas.getContext('2d');

  let mouse = {x: 0, y: 0};
  let last_mouse = {x: 0, y: 0};

  /* Mouse Capturing Work */
  canvas.addEventListener('mousemove', function(e) {
      last_mouse.x = mouse.x;
      last_mouse.y = mouse.y;

      mouse.x = e.pageX - this.offsetLeft;
      mouse.y = e.pageY - this.offsetTop;
  }, false);


  /* Drawing on Paint App */
  ctx.lineWidth = 30;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  canvas.addEventListener('mousedown', function(e) {
      canvas.addEventListener('mousemove', onPaint, false);
  }, false);

  canvas.addEventListener('mouseup', function() {
      canvas.removeEventListener('mousemove', onPaint, false);
  }, false);

  let onPaint = function() {
      ctx.beginPath();
      ctx.moveTo(last_mouse.x, last_mouse.y);
      ctx.lineTo(mouse.x, mouse.y);
      ctx.closePath();
      ctx.stroke();
  };

}());

$("#clear").click(function(){
  let ctx = document.getElementById('canvas').getContext('2d');
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
});

// function scaleImageData(imageData, scale) {
//   let scaled = ctx.createImageData(imageData.width * scale, imageData.height * scale);

//   for (let row = 0; row < imageData.width; row += 1/scale) {
//     for (let col = 0; col < imageData.height; col += 1/scale) {

//     }
//   }

//   // for(let row = 0; row < imageData.height; row++) {
//   //   for(let col = 0; col < imageData.width; col++) {
//   //     let sourcePixel = [
//   //       imageData.data[(row * imageData.width + col) * 4 + 0],
//   //       imageData.data[(row * imageData.width + col) * 4 + 1],
//   //       imageData.data[(row * imageData.width + col) * 4 + 2],
//   //       imageData.data[(row * imageData.width + col) * 4 + 3]
//   //     ];
//   //     for(let y = 0; y < imageData.height; y += 1/scale) {
//   //       let destRow = row * scale + y;
//   //       for(let x = 0; x < imageData.width; x += 1/scale) {
//   //         let destCol = col * scale + x;
//   //         for(let i = 0; i < 4; i++) {
//   //           scaled.data[(destRow * scaled.width + destCol) * 4 + i] =
//   //             sourcePixel[i];
//   //         }
//   //       }
//   //     }
//   //   }
//   // }

//   return scaled;
// }

function captureCanvas() {
  let canvas = document.getElementById('canvas');
  let num_png = canvas.toDataURL();
  
  $.ajax({
    type: "POST",
    url: "/recognize",
    data: { 
       imgBase64: num_png
    }
  }).done(function(o) {
    console.log('saved'); 
    // If you want the file to be visible in the browser 
    // - please modify the callback in javascript. All you
    // need is to return the url to the file, you just saved 
    // and than put the image in your browser.
  });
}
