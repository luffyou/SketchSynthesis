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