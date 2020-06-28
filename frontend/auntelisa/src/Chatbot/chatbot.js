import React from 'react';
import {Link} from "react-router-dom";
import IconButton from '@material-ui/core/IconButton';
import SvgIcon from '@material-ui/core/SvgIcon';
import Button from '@material-ui/core/Button';
import Speak from './speak'

function HomeIcon(props) {
    return (
      <SvgIcon {...props}>
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
      </SvgIcon>
    );
};

function Chatbot() {
    return (
      <div className="Chatbot">
        <div>
            <IconButton color="primary" aria-label="upload picture" component={Link} to={'/'}>
                <HomeIcon fontSize="large" />
            </IconButton>
        </div>
        <div>
          <Button variant="contained" size="medium" color="primary" onClick={() => {Speak('Welcome to Aunt Elisa')}}>
              Say "Welcome to Aunt Elisa"
          </Button>
        </div>
      </div>
    );
}
export default Chatbot;