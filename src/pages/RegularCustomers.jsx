// src/pages/RegularCustomers.js
import React, { useState, useEffect } from 'react';
import { customerService } from '../services/api';

const RegularCustomers = () => {
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRegularCustomers = async () => {
      try {
        setLoading(true);
        const data = await customerService.getRegularCustomers();
        setCustomers(data);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching regular customers:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchRegularCustomers();
  }, []);

  if (loading) {
    return <div className="p-6">Loading regular customers...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Regular Customers</h1>
      <div className="space-y-4">
        {customers.length > 0 ? (
          customers.map((customer) => (
            <div
              key={customer.id}
              className="p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200"
            >
              <div className="flex items-center">
                <img
                  src={customer.photo}
                  alt={customer.name}
                  className="w-12 h-12 rounded-full mr-4 object-cover"
                />
                <div>
                  <h2 className="font-semibold text-lg">{customer.name}</h2>
                  <p className="text-gray-600">Visits: {customer.visits}</p>
                  <p className="text-gray-600">Loyalty Points: {customer.loyalty_points}</p>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center text-gray-500">
            No regular customers found
          </div>
        )}
      </div>
    </div>
  );
};

export default RegularCustomers;