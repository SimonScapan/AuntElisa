import React from 'react';
import {Link} from "react-router-dom";
import IconButton from '@material-ui/core/IconButton';
import KeyboardVoiceIcon from '@material-ui/icons/KeyboardVoice';
import SvgIcon from '@material-ui/core/SvgIcon';
import Button from '@material-ui/core/Button';
import { communication } from '../backendconnection'


// home icon functionalities
function HomeIcon(props) {
    return (
      <SvgIcon {...props}>
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
      </SvgIcon>
    );
};

class Chatbot extends React.Component {
  // initialize state to handle user input text as string
  constructor() {
     super();
     this.state = {
        text: 'initial text'
     }
  }

  // function for computing speech-to-text
  hear = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    // set input language to english
    recognition.lang = 'en-US';
    // start listening
    recognition.start();
    // if there is no longer voice input, than we can access to the result
    recognition.onresult = (event) => {
        const speechToText = event.results[0][0].transcript;
        // set the users voice as string to state variable
        this.setState({ text: speechToText });
        console.log('input from speechrecognition: ' + this.state.text);
    }
  }

  // function for computing text-to-speech
  speak = (input) => {
      const synth = window.speechSynthesis;
  
      const speak = (text) => {
          // for this we use Mozillas SpeechSynthesiser
          var output = new SpeechSynthesisUtterance(text);
          // set the Language to english
          output.lang = 'en-GB';
          synth.speak(output);
      };
      return speak(input);
  }


  // get voice to state
  talk = () =>{
    // listen to user
    let inputtext = this.hear(); 
    // write output to state variable   
    this.setState({ text: inputtext });
  }

  // give output of chatbot
  response = () => {
    // get state variable
    let inputtext = this.state.text; 
    console.log(inputtext);
    // compute answer using chatbot
    inputtext = communication(inputtext);
    // reset state with new answer
    this.setState({ Text: inputtext });
    // give audio response to user
    this.speak(inputtext);
  }

  render() {
    return (
      <div className="Chatbot">

        {/* Home button*/}
        <div>
            <IconButton color="primary" aria-label="upload picture" component={Link} to={'/'}>
                <HomeIcon fontSize="large" />
            </IconButton>
        </div>

        {/* Talk button*/}
        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '20vh'}}>
          <Button 
            variant="contained"  
            startIcon={<KeyboardVoiceIcon />} 
            onClick={() => {this.talk()}}
          >
            Talk
          </Button>
        </div>

        {/* Response button*/}
        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '20vh'}}>
          <Button 
            variant="contained"  
            onClick={() => {this.response()}}
          >
            get response
          </Button>
        </div>

      </div>
    );
  }
}
export default Chatbot;