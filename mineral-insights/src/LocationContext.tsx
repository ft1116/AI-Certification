import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface LocationData {
  type: 'city' | 'state' | 'county' | 'region' | 'coordinates' | 'str' | 'permits';
  name?: string;
  coordinates?: [number, number];
  section?: string;
  township?: string;
  range?: string;
  zoom?: number;
  data?: any; // For permit data
  summary?: any; // For permit summary
}

interface LocationContextType {
  location: LocationData | null;
  setLocation: (location: LocationData | null) => void;
}

const LocationContext = createContext<LocationContextType | undefined>(undefined);

export const useLocation = () => {
  const context = useContext(LocationContext);
  if (context === undefined) {
    throw new Error('useLocation must be used within a LocationProvider');
  }
  return context;
};

interface LocationProviderProps {
  children: ReactNode;
}

export const LocationProvider: React.FC<LocationProviderProps> = ({ children }) => {
  const [location, setLocation] = useState<LocationData | null>(null);

  return (
    <LocationContext.Provider value={{ location, setLocation }}>
      {children}
    </LocationContext.Provider>
  );
};
