import React, { useState, useEffect } from 'react';
import { Button } from 'react-bootstrap';

const MyButton = ({ onClick }) => {
  const [buttonStyle, setButtonStyle] = useState({
    width: 100,
    height: 53,
    left: 400, 
    bottom: 0, 
    position: 'relative',
    background: '#418BF9',
    borderRadius: 15,
    color: 'white',
    border: 'none',
    marginTop: 10,
    boxShadow: '0 4px 20px rgba(76, 175, 80, 0.2)',
    transition: 'background 0.3s ease',
    fontSize: 22,
    fontWeight: 'bold',
    
  });

  useEffect(() => {
    const handleResize = () => {
      setButtonStyle({
        ...buttonStyle,
        left: window.innerWidth - (2*window.innerWidth / 3) + 35, 
        bottom: window.innerHeight -(3*window.innerHeight/ 4) - 190, 
      });
    };

    // Attach the event listener for window resize
    window.addEventListener('resize', handleResize);

    // Call handleResize initially
    handleResize();

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []); // Empty dependency array ensures that useEffect runs only once on mount
  
  const handleMouseOver = () => {
    setButtonStyle((prevStyle) => ({
      ...prevStyle,
      background: '#45a049',
    }));
  };

  const handleMouseOut = () => {
    setButtonStyle((prevStyle) => ({
      ...prevStyle,
      background: '#418BF9',
    }));
  };

  return (
    <Button
      style={buttonStyle}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      onClick={onClick}
    >
      Classify
    </Button>
  );
};

export default MyButton;