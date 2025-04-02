// import React, { useState } from 'react';

// const BillingActivityPage = () => {
//   // Dummy billing data
//   const dummyBillingData = [
//     {
//       id: 1,
//       transactionId: 'TXN12345',
//       customerId: 'CUST101',
//       cashier: 'John Doe',
//       timestamp: '2023-10-10T10:15:00',
//       products: [
//         { name: 'Apple', quantity: 3, price: 50 },
//         { name: 'Banana', quantity: 2, price: 30 },
//         { name: 'Milk', quantity: 1, price: 70 }
//       ],
//       totalAmount: 260,
//       paymentMethod: 'Cash',
//       status: 'Properly Billed',
//       suspiciousActivity: false,
//       skippedItems: []
//     },
//     {
//       id: 2,
//       transactionId: 'TXN12346',
//       customerId: 'CUST102',
//       cashier: 'Jane Smith',
//       timestamp: '2023-10-10T11:30:00',
//       products: [
//         { name: 'Bread', quantity: 2, price: 40 },
//         { name: 'Cheese', quantity: 0, price: 120 }, // Skipped item
//         { name: 'Butter', quantity: 1, price: 90 }
//       ],
//       totalAmount: 130,
//       paymentMethod: 'Card',
//       status: 'Skipped Items',
//       suspiciousActivity: false,
//       skippedItems: ['Cheese']
//     },
//     {
//       id: 3,
//       transactionId: 'TXN12347',
//       customerId: 'CUST103',
//       cashier: 'Alice Johnson',
//       timestamp: '2023-10-10T12:45:00',
//       products: [
//         { name: 'Rice', quantity: 5, price: 80 },
//         { name: 'Oil', quantity: 1, price: 150 },
//         { name: 'Sugar', quantity: 2, price: 50 }
//       ],
//       totalAmount: 630,
//       paymentMethod: 'UPI',
//       status: 'Suspicious Activity',
//       suspiciousActivity: true,
//       skippedItems: []
//     },
//     {
//       id: 4,
//       transactionId: 'TXN12348',
//       customerId: 'CUST104',
//       cashier: 'Michael Brown',
//       timestamp: '2023-10-10T14:00:00',
//       products: [
//         { name: 'Chocolate', quantity: 10, price: 20 }, // Unusually high quantity
//         { name: 'Chips', quantity: 3, price: 30 }
//       ],
//       totalAmount: 290,
//       paymentMethod: 'Card',
//       status: 'Suspicious Activity',
//       suspiciousActivity: true,
//       skippedItems: []
//     },
//     {
//       id: 5,
//       transactionId: 'TXN12349',
//       customerId: 'CUST105',
//       cashier: 'Emily Davis',
//       timestamp: '2023-10-10T15:15:00',
//       products: [
//         { name: 'Eggs', quantity: 1, price: 60 },
//         { name: 'Juice', quantity: 2, price: 40 }
//       ],
//       totalAmount: 140,
//       paymentMethod: 'Cash',
//       status: 'Properly Billed',
//       suspiciousActivity: false,
//       skippedItems: []
//     }
//   ];

//   const [filter, setFilter] = useState('all'); // Filters: 'all', 'suspicious', 'skipped'

//   // Filter billing data based on the selected filter
//   const filteredBillingData = dummyBillingData.filter((entry) => {
//     if (filter === 'suspicious') return entry.suspiciousActivity;
//     if (filter === 'skipped') return entry.skippedItems.length > 0;
//     return true;
//   });

//   return (
//     <div className="p-6">
//       {/* Header Section */}
//       <div className="flex justify-between items-center mb-6">
//         <h1 className="text-2xl font-bold text-gray-800">Billing Activity</h1>
//       </div>

//       {/* Filters Section */}
//       <div className="mb-6">
//         <label htmlFor="filter" className="mr-4 font-medium text-gray-700">
//           Filter By:
//         </label>
//         <select
//           value={filter}
//           onChange={(e) => setFilter(e.target.value)}
//           className="px-4 py-2 border border-gray-300 rounded"
//         >
//           <option value="all">All</option>
//           <option value="suspicious">Suspicious Activity</option>
//           <option value="skipped">Skipped Items</option>
//         </select>
//       </div>

//       {/* Billing Data Table */}
//       <div className="overflow-x-auto">
//       <thead>
//         <tr className="bg-gray-100">
//             <th className="py-2 px-4 border-b">Transaction ID</th>
//             <th className="py-2 px-4 border-b">Customer ID</th>
//             <th className="py-2 px-4 border-b">Cashier</th>
//             <th className="py-2 px-4 border-b">Timestamp</th>
//             <th className="py-2 px-4 border-b">Products</th>
//             <th className="py-2 px-4 border-b">Total Amount</th>
//             <th className="py-2 px-4 border-b">Payment Method</th>
//             <th className="py-2 px-4 border-b">Status</th>
//             <th className="py-2 px-4 border-b">Skipped Items</th>
//         </tr>
//         </thead>
//         <tbody>
//         {filteredBillingData.map((entry) => (
//             <tr key={entry.id} className="hover:bg-gray-50">
//             <td className="py-2 px-4 border-b">{entry.transactionId}</td>
//             <td className="py-2 px-4 border-b">{entry.customerId}</td>
//             <td className="py-2 px-4 border-b">{entry.cashier}</td>
//             <td className="py-2 px-4 border-b">{new Date(entry.timestamp).toLocaleString()}</td>
//             <td className="py-2 px-4 border-b">
//                 {entry.products.map((product) => `${product.name} (${product.quantity})`).join(', ')}
//             </td>
//             <td className="py-2 px-4 border-b">${entry.totalAmount}</td>
//             <td className="py-2 px-4 border-b">{entry.paymentMethod}</td>
//             <td className="py-2 px-4 border-b">
//                 <span
//                 className={`inline-block px-2 py-1 rounded text-white ${
//                     entry.status === 'Properly Billed'
//                     ? 'bg-green-500'
//                     : entry.status === 'Suspicious Activity'
//                     ? 'bg-yellow-500'
//                     : 'bg-red-500'
//                 }`}
//                 >
//                 {entry.status}
//                 </span>
//             </td>
//             <td className="py-2 px-4 border-b">
//                 {entry.skippedItems.length > 0 ? entry.skippedItems.join(', ') : '-'}
//             </td>
//             </tr>
//         ))}
//         </tbody>
//       </div>
//     </div>
//   );
// };

// export default BillingActivityPage;

import React, { useState } from 'react';

const BillingActivityPage = () => {
  // Dummy billing data with suspicious activity details and video URLs
  const dummyBillingData = [
    {
      id: 1,
      transactionId: 'TXN12345',
      customerId: 'CUST101',
      cashier: 'John Doe',
      timestamp: '2023-10-10T10:15:00',
      products: [
        { name: 'Apple', quantity: 3, price: 50 },
        { name: 'Banana', quantity: 2, price: 30 },
        { name: 'Milk', quantity: 1, price: 70 }
      ],
      totalAmount: 260,
      paymentMethod: 'Cash',
      status: 'Properly Billed',
      suspiciousActivity: false,
      skippedItems: [],
      suspiciousReason: null,
      image: 'https://via.placeholder.com/100',
      videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ' // Replace with actual video URL
    },
    {
      id: 2,
      transactionId: 'TXN12346',
      customerId: 'CUST102',
      cashier: 'Jane Smith',
      timestamp: '2023-10-10T11:30:00',
      products: [
        { name: 'Bread', quantity: 2, price: 40 },
        { name: 'Cheese', quantity: 0, price: 120 }, // Skipped item
        { name: 'Butter', quantity: 1, price: 90 }
      ],
      totalAmount: 130,
      paymentMethod: 'Card',
      status: 'Skipped Items',
      suspiciousActivity: false,
      skippedItems: ['Cheese'],
      suspiciousReason: null,
      image: 'https://via.placeholder.com/100',
      videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
    },
    {
      id: 3,
      transactionId: 'TXN12347',
      customerId: 'CUST103',
      cashier: 'Alice Johnson',
      timestamp: '2023-10-10T12:45:00',
      products: [
        { name: 'Rice', quantity: 5, price: 80 },
        { name: 'Oil', quantity: 1, price: 150 },
        { name: 'Sugar', quantity: 2, price: 50 }
      ],
      totalAmount: 630,
      paymentMethod: 'UPI',
      status: 'Suspicious Activity',
      suspiciousActivity: true,
      skippedItems: [],
      suspiciousReason: 'Mismatched price detected for Rice.',
      image: 'https://via.placeholder.com/100',
      videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
    },
    {
      id: 4,
      transactionId: 'TXN12348',
      customerId: 'CUST104',
      cashier: 'Michael Brown',
      timestamp: '2023-10-10T14:00:00',
      products: [
        { name: 'Chocolate', quantity: 10, price: 20 }, // Unusually high quantity
        { name: 'Chips', quantity: 3, price: 30 }
      ],
      totalAmount: 290,
      paymentMethod: 'Card',
      status: 'Suspicious Activity',
      suspiciousActivity: true,
      skippedItems: [],
      suspiciousReason: 'Unusually high quantity of Chocolate purchased.',
      image: 'https://via.placeholder.com/100',
      videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
    }
  ];

  const [filter, setFilter] = useState('all'); // Filters: 'all', 'suspicious', 'skipped'
  const [selectedVideo, setSelectedVideo] = useState(null);

  // Filter billing data based on the selected filter
  const filteredBillingData = dummyBillingData.filter((entry) => {
    if (filter === 'suspicious') return entry.suspiciousActivity;
    if (filter === 'skipped') return entry.skippedItems.length > 0;
    return true;
  });

  // Close video modal
  const closeVideoModal = () => {
    setSelectedVideo(null);
  };

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Billing Activity</h1>
      </div>

      {/* Filters Section */}
      <div className="mb-6">
        <label htmlFor="filter" className="mr-4 font-medium text-gray-700">
          Filter By:
        </label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded"
        >
          <option value="all">All</option>
          <option value="suspicious">Suspicious Activity</option>
          <option value="skipped">Skipped Items</option>
        </select>
      </div>

      {/* Billing Data Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b">Transaction ID</th>
              <th className="py-2 px-4 border-b">Customer ID</th>
              <th className="py-2 px-4 border-b">Products</th>
              <th className="py-2 px-4 border-b">Total Amount</th>
              <th className="py-2 px-4 border-b">Status</th>
              <th className="py-2 px-4 border-b">Skipped Items</th>
              <th className="py-2 px-4 border-b">Suspicious Reason</th>
              <th className="py-2 px-4 border-b">Image</th>
            </tr>
          </thead>
          <tbody>
            {filteredBillingData.length > 0 ? (
              filteredBillingData.map((entry) => (
                <tr key={entry.id} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">{entry.transactionId}</td>
                  <td className="py-2 px-4 border-b">{entry.customerId}</td>
                  <td className="py-2 px-4 border-b">
                    {entry.products.map((product) => `${product.name} (${product.quantity})`).join(', ')}
                  </td>
                  <td className="py-2 px-4 border-b">${entry.totalAmount}</td>
                  <td className="py-2 px-4 border-b">
                    <span
                      className={`inline-block px-2 py-1 rounded text-white ${
                        entry.status === 'Properly Billed'
                          ? 'bg-green-500'
                          : entry.status === 'Suspicious Activity'
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                    >
                      {entry.status}
                    </span>
                  </td>
                  <td className="py-2 px-4 border-b">
                    {entry.skippedItems.length > 0 ? entry.skippedItems.join(', ') : '-'}
                  </td>
                  <td className="py-2 px-4 border-b">{entry.suspiciousReason || '-'}</td>
                  <td className="py-2 px-4 border-b">
                    <img
                      src={entry.image}
                      alt="Transaction"
                      className="w-12 h-12 object-cover cursor-pointer rounded"
                      onClick={() => setSelectedVideo(entry.videoUrl)}
                    />
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="8" className="py-4 text-center">
                  No data available
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Video Modal */}
      {selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg w-3/4 max-w-2xl relative">
            <button
              onClick={closeVideoModal}
              className="absolute top-2 right-2 text-gray-600 hover:text-gray-800"
            >
              &times;
            </button>
            <iframe
              width="100%"
              height="400"
              src={selectedVideo}
              title="Transaction Video"
              frameBorder="0"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      )}
    </div>
  );
};

export default BillingActivityPage;