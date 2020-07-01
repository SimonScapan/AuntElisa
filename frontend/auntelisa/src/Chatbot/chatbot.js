import React from 'react';
import {Link} from "react-router-dom";
import IconButton from '@material-ui/core/IconButton';
import KeyboardVoiceIcon from '@material-ui/icons/KeyboardVoice';
import SvgIcon from '@material-ui/core/SvgIcon';
import Button from '@material-ui/core/Button';
import { communication } from '../backendconnection'

function HomeIcon(props) {
    return (
      <SvgIcon {...props}>
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
      </SvgIcon>
    );
};


class Chatbot extends React.Component {
  constructor() {
     super();
     this.state = {
        text: 'initial text'
     }
  }

  hear = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.start();
    recognition.onresult = (event) => {
        const speechToText = event.results[0][0].transcript;
        console.log(speechToText);
        this.setState({ text: speechToText });
        console.log(this.state.text);
    }
  }

  speak = (input) => {
      const synth = window.speechSynthesis;
  
      const speak = (text) => {
          var output = new SpeechSynthesisUtterance(text);
          output.lang = 'en-GB';
          synth.speak(output);
      };
      return speak(input);
  }

  wait = (s) => {
    var start = new Date().getTime();
    var end = start;
    while(end < start + (s * 1000) ) {
      end = new Date().getTime();
   }
 }

  talk = () =>{
    let inputtext = this.hear();    
    this.setState({ text: inputtext });
    console.log(inputtext);
    console.log(this.state.text);
  }

  response = () => {
    let inputtext = this.state.text; 
    console.log(inputtext);
    inputtext = communication(inputtext);
    this.setState({ Text: inputtext });
    this.speak(inputtext);
  }

  render() {
    return (
      <div className="Chatbot">
        <div>
          {/* Home button*/}
            <IconButton color="primary" aria-label="upload picture" component={Link} to={'/'}>
                <HomeIcon fontSize="large" />
            </IconButton>
        </div>

        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '20vh'}}>
          {/* Talk button*/}
          <Button 
            variant="contained"  
            startIcon={<KeyboardVoiceIcon />} 
            onClick={() => {this.talk()}}
          >
            Talk
          </Button>
        </div>
        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '20vh'}}>
          {/* Response button*/}
          <Button 
            variant="contained"  
            onClick={() => {this.response()}}
          >
            get response
          </Button>
        </div>
        <div>
          {/* input-text box*/}
          <h2>Text is: </h2>
          <h2>{this.state.Text}</h2>
        </div>
      </div>
    );
  }
}
export default Chatbot;