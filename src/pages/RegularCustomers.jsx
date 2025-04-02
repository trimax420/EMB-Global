// src/pages/RegularCustomers.js
import React from "react";

const RegularCustomers = () => {
  const customerList = [
    {
      id: 1,
      name: "Alice Brown",
      photo: "https://via.placeholder.com/100",
      visits: 25,
      loyaltyPoints: 500,
    },
    {
      id: 2,
      name: "Bob Green",
      photo: "https://via.placeholder.com/100",
      visits: 18,
      loyaltyPoints: 350,
    },
    {
      id: 3,
      name: "Charlie White",
      photo: "https://via.placeholder.com/100",
      visits: 12,
      loyaltyPoints: 200,
    },
  ];

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Regular Customers</h1>
      <div className="space-y-4">
        {customerList.map((customer) => (
          <div
            key={customer.id}
            className="p-4 bg-white rounded-lg shadow-md"
          >
            <div className="flex items-center">
              <img
                src={customer.photo}
                alt={customer.name}
                className="w-12 h-12 rounded-full mr-4"
              />
              <div>
                <h2 className="font-semibold">{customer.name}</h2>
                <p>Visits: {customer.visits}</p>
                <p>Loyalty Points: {customer.loyaltyPoints}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RegularCustomers;