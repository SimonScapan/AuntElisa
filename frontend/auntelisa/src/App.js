import React from 'react';

import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import chatbot from './Chatbot/chatbot'
import landingpage from './Landingpage/landingpage'

function App(){
  return(
    <Router>

      {/* two routes are needed: main-/landingpage and chatbot */}
      <Switch>
        <Route path='/' component={landingpage} exact />
        <Route path='/chatbot' component={chatbot} exact />
      </Switch>

    </Router>
  )
}
export default App;