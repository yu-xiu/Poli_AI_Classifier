import React, { useEffect, useState } from 'react';
import '../styles/fadein.css';

const FadeInComponent = () => {
    const [fadeIn, setFadeIn] = useState(false);

    useEffect(() => {
        // Trigger the fade-in effect after the component mounts
        setFadeIn(true);
    }, []);

    const text = "How can we help you today?".split(' ');
    
    return (
        
        <div className={`fade-in-question-container ${fadeIn ? 'fade-in' : ''}`}>
            {text.map((word, index) => (
                <span key={index} className="word">{word} </span>
            ))}
        </div>
        
    );
};
  
  export default FadeInComponent;