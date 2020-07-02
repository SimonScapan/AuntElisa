import React from 'react';
import Button from '@material-ui/core/Button';
import './landingpage.css';
import {Link} from "react-router-dom";
import logo from './logo.mp4';

function landingpage() {
    return (
      <div className="landingpage">

        {/* div with animated logo of AuntElisa */}
        <div style={{display: 'flex',  justifyContent:'center', height: '80vh'}}>
          <video className='videoTag' autoPlay loop muted>
            <source src={logo} type='video/mp4' />
          </video>
        </div>

        {/* div with button to get to chatbot page */}
        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '20vh'}}>
          <header className="Header">
            <Button variant="contained" size="large" color="primary" component={Link} to={'/chatbot'}>
              Start Aunt Elisa
            </Button>
          </header>
        </div>
        
      </div>
    );
  }
export default landingpage;