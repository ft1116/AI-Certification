import React from 'react';
import LeasingDashboard from './LeasingDashboard';

interface MapTestProps {
  onBack?: () => void;
}

const MapTest: React.FC<MapTestProps> = ({ onBack }) => {
  return <LeasingDashboard onBack={onBack} />;
};

export default MapTest;
