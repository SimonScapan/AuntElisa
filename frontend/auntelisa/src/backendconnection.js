import $ from "jquery";

// backend is available on localhost port 5000
const SERVER = "http://localhost:5000";

// communication to backend with input as string from SpeechRecognition
export function communication(input) {
  console.log('input in communication : ' + input);
  let response;
  $.ajax({
    // the string is easily appended to the URL
    url: SERVER+"/backend/"+input,    
    dataType: "text",
    type: "GET",
    crossDomain: true,
    async: false,
    success: function(serverResponse) {
      console.log("Response: ", serverResponse);
      response = serverResponse;
    }
  });
  console.log('from bakend: ' + response);
  // response from backend is given back
  return response;
}