import React from 'react';

const ResultPage = () => {
  return (
    <div style={{width: '100%', height: '100%', position: 'relative'}}>
        <div style={{width: 848, height: 464, left: 0, top: 0, position: 'absolute', background: '#F2F0F0'}} />
        <div style={{width: 846, height: 45, left: 2, top: 0, position: 'absolute', background: '#E6E6E6'}} />
        <div style={{width: 16, height: 15, left: 37, top: 23, position: 'absolute', background: '#418BF9', borderRadius: 9999}} />
        <div style={{width: 7, height: 7, left: 75, top: 13, position: 'absolute', background: '#418BF9', borderRadius: 9999}} />
        <div style={{left: 16, top: 9, position: 'absolute', color: 'black', fontSize: 30, fontFamily: 'Inter', fontWeight: '400', wordWrap: 'break-word'}}>P</div>
        <div style={{left: 57, top: 9, position: 'absolute', color: 'black', fontSize: 30, fontFamily: 'Inter', fontWeight: '400', wordWrap: 'break-word'}}>L</div>
        <div style={{width: 10, height: 15, left: 76, top: 16, position: 'absolute', color: 'black', fontSize: 22, fontFamily: 'Inter', fontWeight: '400', wordWrap: 'break-word'}}>I</div>
        <div style={{width: 321, height: 163, left: 92, top: 108, position: 'absolute', background: '#FBF5F5', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)', borderRadius: 29, border: '1px #E6E6E6 solid'}} />
        <div style={{width: 321, height: 163, left: 434, top: 280, position: 'absolute', background: '#FBF5F5', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)', borderRadius: 29, border: '1px #EAE7E7 solid'}} />
        <div style={{width: 321, height: 163, left: 92, top: 280, position: 'absolute', background: '#FBF5F5', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)', borderRadius: 29, border: '1px #EAE7E7 solid'}} />
        <div style={{width: 321, height: 163, left: 434, top: 108, position: 'absolute', background: '#FBF5F5', boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)', borderRadius: 29, border: '1px #EAE7E7 solid'}} />
        <div style={{left: 361, top: 55, position: 'absolute', color: 'black', fontSize: 35, fontFamily: 'Inter', fontWeight: '700', wordWrap: 'break-word'}}>Results</div>
        <div style={{width: 297, height: 0, left: 101, top: 140, position: 'absolute', border: '1px #EAE7E7 solid'}}></div>
        <div style={{width: 297, height: 0, left: 448, top: 140, position: 'absolute', border: '1px #EAE7E7 solid'}}></div>
        <div style={{width: 297, height: 0, left: 448, top: 315, position: 'absolute', border: '1px #E6E6E6 solid'}}></div>
        <div style={{width: 297, height: 0, left: 101, top: 315, position: 'absolute', border: '1px #E6E6E6 solid'}}></div>
        <div style={{width: 90, height: 16, left: 114, top: 120, position: 'absolute', color: '#646262', fontSize: 15, fontFamily: 'Inter', fontWeight: '700', wordWrap: 'break-word'}}>GPT2 Model</div>
        <div style={{width: 90, height: 16, left: 458, top: 297, position: 'absolute', color: '#646262', fontSize: 15, fontFamily: 'Inter', fontWeight: '700', wordWrap: 'break-word'}}>TF-IDF MNB</div>
        <div style={{width: 98, height: 19, left: 114, top: 291, position: 'absolute', color: '#646262', fontSize: 15, fontFamily: 'Inter', fontWeight: '700', wordWrap: 'break-word'}}>LSTM Model</div>
        <div style={{width: 124, height: 16, left: 458, top: 120, position: 'absolute', color: '#646262', fontSize: 15, fontFamily: 'Inter', fontWeight: '700', wordWrap: 'break-word'}}>BERT Model</div>
    </div>
  );
}

export default ResultPage;