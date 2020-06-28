import React from 'react';
import {Link} from "react-router-dom";
import IconButton from '@material-ui/core/IconButton';
import KeyboardVoiceIcon from '@material-ui/icons/KeyboardVoice';
import SvgIcon from '@material-ui/core/SvgIcon';
import Button from '@material-ui/core/Button';

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
        inputText: 'input text',
        outputText: 'output text'
     }
  }

  hear = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.start();
    recognition.onresult = (event) => {
        const speechToText = event.results[0][0].transcript;
        console.log(speechToText);
        this.setState({ inputText: speechToText });
        console.log(this.state.inputText);
    }
  }

  speak = (input) => {
      const synth = window.speechSynthesis;
  
      const speak = (text) => {
          var foo = new SpeechSynthesisUtterance(text);
          foo.lang = 'en-GB';
          synth.speak(foo);
      };
      return speak(input);
  }

  changeOutput = () => {
    this.setState({ outputText: this.state.inputText });
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
        <div>
          {/* Microfone button*/}
          <Button 
            variant="contained" 
            color="secondary" 
            startIcon={<KeyboardVoiceIcon />} 
            onClick={() => {this.hear()}}
          >
            Talk
          </Button>
        </div>
        <div>
          {/* input-text box*/}
          <h2>Your input is: </h2>
          <h2>{this.state.inputText}</h2>
        </div>
        <div>
          {/* speach bubble*/}
          <Button 
            variant="contained" 
            size="medium" 
            color="primary" 
            onClick={() => {this.changeOutput()}}
          >
              compute input
          </Button>
        </div>
        <div>
          {/* output-text box*/}
          <h2>Elisa told you: </h2>
          <h2>{this.state.outputText}</h2>
        </div>
        <div>
          <Button 
            variant="contained" 
            size="medium" 
            color="primary" 
            onClick={() => {this.speak(this.state.outputText)}}
          >
              Say "Welcome to Aunt Elisa"
          </Button>
        </div>
      </div>
    );
  }
}
export default Chatbot;