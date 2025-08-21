import React, {useState, useEffect} from 'react';
import '../styles/result.css';
import '../styles/fadein.css';

function OutputArea({outputValue}) {
    const [fadeIn, setFadeIn] = useState(false);

    useEffect(() => {
        // Trigger the fade-in effect after the component mounts
        setFadeIn(true);
    }, []);
    const text = "BERT BIAS".split(' ');
    const text_message = "BERT MESSAGE".split(' ');

    console.log("OUTPUT!!!!");
    return (
        <div className="container">
            <div className="result-box">
                {/* <h2>BERT BIAS</h2> */}
                <div className={`fade-in-result-container ${fadeIn ? 'fade-in-result' : ''}`}>
                    {text.map((word, index) => (
                        <span key={index} className="word">{word} </span>
                    ))}
                </div>
            <div className="result-content">
                {outputValue.length > 0 && (
                <ul>
                    <li className='list-item'>{outputValue[0][0]}</li>
                </ul>
                )}
            </div>
            </div>

            <div className="result-box">
                {/* <h2>BERT MESSAGE</h2> */}
                <div className={`fade-in-result-container ${fadeIn ? 'fade-in-result' : ''}`}>
                    {text_message.map((word, index) => (
                        <span key={index} className="word">{word} </span>
                    ))}
                </div>
                <div className="result-content">
                    {outputValue.length > 0 && (
                        <ul>
                            <li className='list-item'>{outputValue[0][1]}</li>
                        </ul>
                    )}
                </div>
            </div>
        </div>
    );
};
export default OutputArea;