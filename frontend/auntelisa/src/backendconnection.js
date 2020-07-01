import $ from "jquery";
const SERVER = "http://localhost:5000";

export function login(user, password) {
  let response;
  $.ajax({
    url: SERVER,
    dataType: "text",
    type: "GET",
    crossDomain: true,
    async: false,
    success: function(serverResponse) {
      console.log("Response: ", serverResponse);
      response = serverResponse;
    }
  });
  return response;
}

export function communication(input) {
  console.log('input in communication : ' + input);
  let response;
  $.ajax({
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
  console.log(response);
  return response;
}