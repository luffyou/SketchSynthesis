console.log('start')
// $("#myVideo").hide(); // 摄像头画面隐藏
//调用用户媒体设备, 访问摄像头 videob不是变量而是关键字
getUserMedia({video : {width: 640, height: 480}}, success, error);
let mv = document.getElementById("myVideo");
//create a canvas to grab an image for upload
let imgCanvas = document.getElementById('canvas');
let imgCtx = imgCanvas.getContext("2d");
let imgResp = document.getElementById('resImg')
// imgCanvas.width = mv.videoWidth;
// imgCanvas.height = mv.videoHeight;
// let apiUrl = "{{ url_for('image') }}";
let apiUrl = "{{ url_for('img_trans') }}"; // 此处会申请一次img_trans函数，最好换成域名
let imgType = 'image/jpeg' // toDataURL不加image/jpeg,会默认成image/png
// v1 send blob
// mv.onclick = function() {
//     sendImagefromCanvasCommon();
//     // sendImagefromCanvasChrome();
// }

// v2 send dataUrl
// mv.addEventListener('click', function () {
//     imgCtx.drawImage(mv, 0, 0, mv.videoWidth, mv.videoHeight); //此句必须
//     let imgStr = imgCanvas.toDataURL(imgType);  //.substring(subIdx)
//     let tmStart = (new Date()).getTime();
//     $.ajax({
//         method: 'POST',
//         url: apiUrl,
//         data: imgStr,
//         // dataType: 'json', # 指定接收消息格式, 未指定会自行判断
//         contentType: false,
//         processData: false,
//         success: function(data, textStatus) { // ！！！
//             // alert(textStatus);
//             console.log(`recv: ${(new Date()).getTime()-tmStart} ms`)
//             imgResp.setAttribute('src',data);
//         },
//         　error: function (XMLHttpRequest, textStatus, errorThrown) {
//     　　　　  console.error(`${XMLHttpRequest.status},${XMLHttpRequest.readyState},${textStatus}`);
//     　　}
//     });
// },false)

// v3 long polling
imgCtx.drawImage(mv, 0, 0, mv.videoWidth, mv.videoHeight); 
imgStr = imgCanvas.toDataURL(imgType); 
$(function () {
    (function longPolling() {
        imgCtx.drawImage(mv, 0, 0, mv.videoWidth, mv.videoHeight); 
        imgStr = imgCanvas.toDataURL(imgType); 
        let tmStart = (new Date()).getTime(); 
        $.ajax({
            method: 'POST', // 否则默认get
            url: apiUrl,
            data: imgStr,
            timeout: 5000,
            contentType: false,
            processData: false,
            success: function (data, textStatus) {
                imgResp.setAttribute('src',data);
                console.log(`recv: ${(new Date()).getTime()-tmStart} ms`)
                longPolling(); // 递归调用
            },
            error: function (XMLHttpRequest, textStatus, errorThrown) {
                console.error(`${XMLHttpRequest.status},${XMLHttpRequest.readyState},${textStatus}`);
                longPolling(); // 递归调用
            }
        });
    })();
});


// function definition
//访问用户媒体设备的兼容方法
function getUserMedia(constraints, success, error) {
    if (navigator.mediaDevices.getUserMedia) {
        console.log('Latest standard API');
        navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
    } else if (navigator.webkitGetUserMedia) {
        console.log('Webkit core browser API')
        navigator.webkitGetUserMedia(constraints,success, error)
    } else if (navigator.mozGetUserMedia) {
        console.log('firfox browser API')
        navigator.mozGetUserMedia(constraints, success, error);
    } else if (navigator.getUserMedia) {
        console.log('Old version API')
        navigator.getUserMedia(constraints, success, error);
    } else {
        console.log('No adaptation media API');
    }
}
function success(stream) {
    console.log(stream);
    //兼容webkit核心浏览器
    // let CompatibleURL = window.URL || window.webkitURL;
    // mv.src = CompatibleURL.createObjectURL(stream);
    //将视频流设置为video元素的源
    mv.srcObject = stream;
    mv.play();
}
function error(error) {
    alert(`访问用户媒体设备失败${error.name}, ${error.message}`);
}


function sendImagefromCanvasCommon() {
    imgCtx.drawImage(mv, 0, 0, mv.videoWidth, mv.videoHeight); //此句必须
    let dataurl = imgCanvas.toDataURL(imgType);
    let blob = convertBase64UrlToBlob(dataurl);
    //使用ajax发送
    let formdata = new FormData();
    formdata.append("image", blob);
    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl, true);
    xhr.onload = function () {
        if (this.status === 200) {
            // console.log(this.response);
            // $("resImg").attr("src","data:image/png;base64,"+img);
            // imgResp.innerHTML = this.response; // <img src=....>
            imgResp.setAttribute('src',this.response);
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}
// 这是一个转换base64的一个方法
function convertBase64UrlToBlob(urlData){
    let bytes=window.atob(urlData.split(',')[1]);        //去掉url的头，并转换为byte
    //处理异常,将ascii码小于0的转换为大于0
    let ab = new ArrayBuffer(bytes.length);
    let ia = new Uint8Array(ab);
    for (let i = 0; i < bytes.length; i++) {
        ia[i] = bytes.charCodeAt(i);
    }
    return new Blob( [ab] , {type : imgType});
}

//Get the image from the canvas
function sendImagefromCanvasChrome() {
    // Make sure the canvas is set to the current mv size
    imgCtx.drawImage(mv, 0, 0, mv.videoWidth, mv.videoHeight);
    //Convert the canvas to blob and post the file
    imgCanvas.toBlob(postFile, imgType);
}
//Add file blob to a form and post
function postFile(file) {
    let formdata = new FormData();
    formdata.append("image", file);
    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl, true);
    xhr.onload = function () {
        if (this.status === 200) {
            // console.log(this.response);
            imgResp.setAttribute('src',this.response);
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}

console.log('end')