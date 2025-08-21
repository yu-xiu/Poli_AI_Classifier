import React from 'react';
import { Nav, Image } from 'react-bootstrap';
import { Link } from 'react-router-dom';

function Navigation() {
  return (
    <Nav variant="tabs" defaultActiveKey="/home" style={{ backgroundColor: '#f0f0f0', padding: '10px', background: '#E0EDF5', display: 'flex', justifyContent: 'space-between' }}>
      <Nav.Item>
        <Nav.Link href="/home" style={{ color: '#333', fontWeight: 'bold', fontSize: '36px' }}>
        <Image src='/poli_logo.png' alt="Home" style={{ width: '110px', height: '50px', marginRight: '20px', marginTop: '8px' }} />        </Nav.Link>
      </Nav.Item>
    </Nav>
  );
}
export default Navigation;