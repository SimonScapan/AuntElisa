import React from 'react';
import Button from '@material-ui/core/Button';
import './landingpage.css';
import {Link} from "react-router-dom";


function landingpage() {
    return (
      <div className="Landingpage" style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '60vh'}}>
        <header className="Header">
          <Button variant="contained" size="large" color="primary" component={Link} to={'/chatbot'}>
            Welcome to Aunt Elisa
          </Button>
        </header>
      </div>
    );
  }
export default landingpage;