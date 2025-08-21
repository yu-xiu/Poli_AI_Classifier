import React, { useState, useEffect } from 'react';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
function InputArea({value, onChange}) {
    const handleInputChange = (event) => {
        // Call the provided onChange prop to update the parent component's state
        onChange(event.target.value);
    };
    
    const [inputStyles, setInputStyles] = useState({
        fontSize: 22,
        marginTop: 70,
        marginLeft: 15,
        borderRadius: 20,
        backgroundColor: '#E0E0E0',
        border: 'none',
        boxShadow: '0 0 10px #969494',
        paddingTop: '20px',
        paddingLeft: '20px',
    });
    const [textareaSize, setTextareaSize] = useState({
    rows: 10, // initial number of rows
    cols: 40, // initial number of cols
  });

  useEffect(() => {
    const handleResize = () => {
      // Update styles based on window width
      setInputStyles({
        ...inputStyles,
        fontSize: window.innerWidth > 768 ? 22 : 16,
        marginTop: window.innerWidth > 768 ? 70 : 50,
        // Add more style adjustments based on window width if needed
      });

      // Update textarea size based on window width
      setTextareaSize({
        rows: window.innerWidth > 768 ? 10 : 5,
        cols: window.innerWidth > 768 ? 40 : 20,
      });
    };

    // Attach the event listener for window resize
    window.addEventListener('resize', handleResize);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [inputStyles]);
    return(
        <InputGroup size="lg">
            <Form.Group >
                <Form.Control 
                    as="textarea" 
                    type="text"
                    rows={textareaSize.rows} 
                    cols={textareaSize.cols}
                    placeholder='Type your political post here'
                    onChange={handleInputChange}
                    style={inputStyles}
                    value={value}
                    id="myTextArea" 
                    name="myTextArea"
                />
            </Form.Group>
        </InputGroup>
    )
}
export default InputArea;