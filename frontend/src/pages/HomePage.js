import React, { useState, useEffect } from 'react';
import { Form, Button, Container } from 'react-bootstrap';
import ClassifyBtn from '../components/ClassifyButton';
import WordsAnimation from '../components/FadeInWords';
import UserInputArea from '../components/Input';
import UserOutputArea from '../components/Output';
import NavBar from '../components/NavBar'
import '../styles/fadein.css';
import '../styles/home.css';
import '../styles/columns.css';
import '../styles/result.css';
import axios from "axios";


const HomePage = () => {
  const [postText, setPostText] = useState('');
  const [results, setResults] = useState([]);

  // handle user input
  const handleInputChange = (value) => {
    setPostText(value);
  };

  const handleSubmit = async () => {
    try {
      console.log('postText=', postText);
      const response = await axios.post('http://127.0.0.1:5000/generate_results', {
        userInput: postText, // this is the reslut posted to the result based on user text input
      });

      console.log('Response Data:', response.data);
      setResults([response.data.result]);
    } catch (error) {
      console.error('Error:', error.message);
    }
  };

  // handle window size
  const [windowDimensions, setWindowDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setWindowDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    // Attach the event listener for window resize
    window.addEventListener('resize', handleResize);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div style={{width: `${windowDimensions.width}px`, height: `${windowDimensions.height}px`, position: 'relative', background: '#F2F0F0'}}>
      <div>
        <NavBar/>
      </div>
      <div className="col-container">
        <div className="column">
          {/* Content for the home column */}
          <div>
        
            <div className='fadein'>
              <WordsAnimation />
            </div>
          
            <div className='title'>Your AI Classifier</div>

            {/* taking in a user's input */}
            <div >
              <UserInputArea value={postText} onChange={handleInputChange}/>
            </div>

            <ClassifyBtn className='classify-button' onClick={handleSubmit}/>
          </div>
        </div>
        
        <div className="column">
          <div className='result'>Results</div>
          {/* Content for the result column */}
          <UserOutputArea outputValue={results}/>          
          </div>
        </div>
    </div>
  );
}
export default HomePage;
