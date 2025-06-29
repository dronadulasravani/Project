import React  from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import Header from './components/Header';
import Home from './components/Home';
import About from './components/About';
import Prediction from './components/Prediction';
import Ourteam from './components/Ourteam';
import Chatbot from './components/Chatbot';
import Mobilenavbar from './components/Mobilenavbar';
import Footer from './components/Footer';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <>
    <Header/>
    <Home />
    <About />
    <Prediction />
    <Chatbot />
    <Ourteam />
    <Footer />
    <Mobilenavbar />
  </>
);

