import $ from "./node_modules/jquery";
const SERVER = "http://localhost:5000";

export function setmessage(input) {
    console.log("setmessage",input)
    let response;
    $.ajax({
      url: SERVER+"/backend",    
      headers: {"Access-Control-Allow-Origin": "*"},
      dataType: "text",
      contentType: 'application/json',
      data: JSON.stringify(input),
      type: "SET",
      crossDomain: true,
      async: false,
      success: function(serverResponse) {
        console.log("Response: ", serverResponse);
        response = serverResponse;
      },
      error: function(serverResponse) {
        console.log("Response: ", serverResponse);
        response = serverResponse;
        debugger;
        throw new Error("Error during sending message");
      }
    });
    return response;
  }

export function getmessage() {
    let response;
    $.ajax({
      url: SERVER+"/backend",
      dataType: "text",
      type: "GET",
      async: false,
      success: function(serverResponse) {
        console.log("Response: ", serverResponse);
        response = serverResponse;
      }
    });
    return response;
}

