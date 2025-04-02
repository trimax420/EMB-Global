import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  ChevronDown, 
  UserPlus, 
  UserX, 
  Clock, 
  Calendar, 
  Camera, 
  AlertTriangle,
  Tag,
  Edit,
  Trash2,
  MoreVertical,
  CheckCircle,
  XCircle,
  Eye,
  Download,
  FileText,
  ArrowUpRight,
  Flag,
  Users,
  ExternalLink,
  Shield,
  Bell,
  Plus,
  X,
  Zap,
  Map,
  Calendar as CalendarIcon,
  User,
  BarChart4
} from 'lucide-react';

// Mock data with improved details
const initialSuspects = [
  {
    id: 1,
    name: 'Unknown Subject #38',
    status: 'Active',
    lastSeen: '2025-04-01T09:12:00',
    location: 'Main Entrance',
    incidents: 2,
    description: 'Male, approx. 30-40 years old, 5\'10", wearing dark clothing. Suspected of shoplifting from electronics department.',
    tags: ['Theft', 'Electronics'],
    risk: 'Medium',
    image: null,
    detections: [
      { date: '2025-04-01T09:12:00', location: 'Main Entrance', camera: 'CAM-04', confidence: 89 },
      { date: '2025-03-30T15:37:00', location: 'Electronics Department', camera: 'CAM-12', confidence: 94 }
    ],
    notes: 'Subject appears to be surveying security measures and camera locations. Has been observed multiple times in electronics department handling display phones.'
  },
  {
    id: 2,
    name: 'Unknown Subject #42',
    status: 'Active',
    lastSeen: '2025-04-01T08:47:00',
    location: 'Parking Lot North',
    incidents: 1,
    description: 'Female, approx. 25-35 years old, 5\'6", with blonde hair. Observed loitering in parking lot and approaching customers.',
    tags: ['Loitering', 'Harassment'],
    risk: 'Low',
    image: null,
    detections: [
      { date: '2025-04-01T08:47:00', location: 'Parking Lot North', camera: 'CAM-22', confidence: 87 },
      { date: '2025-03-29T10:15:00', location: 'Parking Lot North', camera: 'CAM-23', confidence: 91 }
    ],
    notes: 'Subject has been approaching customers in parking lot asking for money. No aggressive behavior reported but causing customer discomfort.'
  },
  {
    id: 3,
    name: 'John Doe',
    status: 'Inactive',
    lastSeen: '2025-03-30T14:22:00',
    location: 'Jewelry Section',
    incidents: 3,
    description: 'Male, mid-40s, 6\'0", wearing business casual attire. Previously caught attempting to access restricted areas.',
    tags: ['Unauthorized Access', 'Suspicious Activity'],
    risk: 'High',
    image: null,
    detections: [
      { date: '2025-03-30T14:22:00', location: 'Jewelry Section', camera: 'CAM-09', confidence: 97 },
      { date: '2025-03-28T11:33:00', location: 'Staff Corridor', camera: 'CAM-31', confidence: 96 },
      { date: '2025-03-25T16:05:00', location: 'Security Office Entrance', camera: 'CAM-02', confidence: 98 }
    ],
    notes: 'Subject presents himself as a vendor/contractor but has no proper credentials. Has attempted to access staff areas multiple times. Local authorities have been notified.'
  },
  {
    id: 4,
    name: 'Unknown Subject #39',
    status: 'Active',
    lastSeen: '2025-03-31T16:05:00',
    location: 'Electronics Department',
    incidents: 1,
    description: 'Male teenager, approx. 16-18 years old, wearing red hoodie and jeans. Observed handling display items suspiciously.',
    tags: ['Suspicious Activity', 'Electronics'],
    risk: 'Low',
    image: null,
    detections: [
      { date: '2025-03-31T16:05:00', location: 'Electronics Department', camera: 'CAM-12', confidence: 88 }
    ],
    notes: 'Subject was observed removing security devices from display phones. When approached by staff, claimed to be "just looking." No theft occurred but behavior was suspicious.'
  },
  {
    id: 5,
    name: 'Jane Smith',
    status: 'Active',
    lastSeen: '2025-04-01T11:30:00',
    location: 'Cosmetics Section',
    incidents: 4,
    description: 'Female, 40s, 5\'4", dark hair. Known shoplifter with multiple prior incidents.',
    tags: ['Repeat Offender', 'Theft', 'Cosmetics'],
    risk: 'High',
    image: null,
    detections: [
      { date: '2025-04-01T11:30:00', location: 'Cosmetics Section', camera: 'CAM-07', confidence: 96 },
      { date: '2025-03-27T14:12:00', location: 'Cosmetics Section', camera: 'CAM-07', confidence: 97 },
      { date: '2025-03-20T10:45:00', location: 'Pharmacy Section', camera: 'CAM-08', confidence: 95 }
    ],
    notes: 'Known repeat offender with history of shoplifting cosmetics and pharmaceutical items. Has been banned from premises but continues to return. Police have been involved previously.'
  }
];

// Activity log data
const activityLogData = [
  { id: 1, action: 'Suspect Added', user: 'Security Officer Jones', timestamp: '2025-04-01T10:15:00', details: 'Added Unknown Subject #38 to watchlist' },
  { id: 2, action: 'Status Change', user: 'Security Officer Smith', timestamp: '2025-04-01T09:30:00', details: 'Changed John Doe from Active to Inactive' },
  { id: 3, action: 'New Detection', user: 'AI System', timestamp: '2025-04-01T09:12:00', details: 'Unknown Subject #38 detected at Main Entrance' },
  { id: 4, action: 'Notes Updated', user: 'Security Officer Williams', timestamp: '2025-03-31T16:30:00', details: 'Updated notes for Unknown Subject #39' },
  { id: 5, action: 'Suspect Added', user: 'Security Officer Lee', timestamp: '2025-03-31T15:45:00', details: 'Added Jane Smith to watchlist' }
];

const SuspectList = () => {
  const [suspects, setSuspects] = useState(initialSuspects);
  const [activeTab, setActiveTab] = useState('watchlist');
  const [selectedSuspect, setSelectedSuspect] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeView, setActiveView] = useState('details'); // 'details', 'history', 'footage'
  const [showActivityLog, setShowActivityLog] = useState(false);
  const [showTagFilter, setShowTagFilter] = useState(false);
  const [selectedTags, setSelectedTags] = useState([]);
  const [riskFilter, setRiskFilter] = useState('all');
  const [showDelete, setShowDelete] = useState(false);
  const [newTag, setNewTag] = useState('');
  const [tagSuggestions] = useState(['Theft', 'Loitering', 'Unauthorized Access', 'Harassment', 'Electronics', 'Repeat Offender', 'Suspicious Activity', 'Vandalism']);
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    status: 'Active',
    location: '',
    incidents: 0,
    description: '',
    tags: [],
    risk: 'Medium',
    notes: ''
  });

  // Get all unique tags from suspects
  const allTags = Array.from(new Set(suspects.flatMap(suspect => suspect.tags)));

  // Filter suspects based on current filters
  const filteredSuspects = suspects
    .filter(suspect => 
      (activeTab === 'watchlist' && suspect.status === 'Active') ||
      (activeTab === 'archive' && suspect.status === 'Inactive')
    )
    .filter(suspect => 
      searchQuery === '' || 
      (suspect.name && suspect.name.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (suspect.description && suspect.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (suspect.tags && suspect.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())))
    )
    .filter(suspect => 
      selectedTags.length === 0 || 
      selectedTags.every(tag => suspect.tags.includes(tag))
    )
    .filter(suspect => 
      riskFilter === 'all' || 
      suspect.risk === riskFilter
    );

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Format date into relative time
  const formatRelativeTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) {
      return 'Just now';
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60);
      return `${minutes} ${minutes === 1 ? 'minute' : 'minutes'} ago`;
    } else if (diffInSeconds < 86400) {
      const hours = Math.floor(diffInSeconds / 3600);
      return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
    } else {
      const days = Math.floor(diffInSeconds / 86400);
      return `${days} ${days === 1 ? 'day' : 'days'} ago`;
    }
  };

  // Get color based on risk level
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High': return 'bg-red-100 text-red-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'Low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Get risk level icon based on risk level
  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'High':
        return <AlertTriangle size={14} className="text-red-500" />;
      case 'Medium':
        return <AlertTriangle size={14} className="text-yellow-500" />;
      case 'Low':
        return <AlertTriangle size={14} className="text-green-500" />;
      default:
        return <AlertTriangle size={14} className="text-gray-500" />;
    }
  };

  // Handle opening the form for adding/editing
  const handleOpenForm = (suspect = null) => {
    if (suspect) {
      setFormData({
        id: suspect.id,
        name: suspect.name,
        status: suspect.status,
        location: suspect.location,
        incidents: suspect.incidents,
        description: suspect.description,
        tags: [...suspect.tags],
        risk: suspect.risk,
        notes: suspect.notes || ''
      });
    } else {
      setFormData({
        name: '',
        status: 'Active',
        location: '',
        incidents: 0,
        description: '',
        tags: [],
        risk: 'Medium',
        notes: ''
      });
    }
    setSelectedSuspect(suspect);
    setShowForm(true);
  };

  // Handle form submission
  const handleSubmitForm = (e) => {
    e.preventDefault();
    
    const now = new Date().toISOString();
    
    if (selectedSuspect) {
      // Update existing suspect
      setSuspects(suspects.map(suspect => 
        suspect.id === selectedSuspect.id 
          ? { 
              ...suspect, 
              ...formData, 
              lastSeen: suspect.lastSeen // Preserve lastSeen
            } 
          : suspect
      ));
    } else {
      // Add new suspect
      const newSuspect = {
        ...formData,
        id: suspects.length + 1,
        lastSeen: now,
        detections: [{
          date: now,
          location: formData.location,
          camera: 'Unknown',
          confidence: 90
        }]
      };
      setSuspects([...suspects, newSuspect]);
    }
    
    setShowForm(false);
  };

  // Handle adding a tag
  const handleAddTag = () => {
    if (newTag && !formData.tags.includes(newTag)) {
      setFormData({
        ...formData,
        tags: [...formData.tags, newTag]
      });
      setNewTag('');
    }
  };

  // Handle removing a tag
  const handleRemoveTag = (tag) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter(t => t !== tag)
    });
  };

  // Handle toggling a tag filter
  const handleToggleTagFilter = (tag) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  // Handle suspect deletion
  const handleDeleteSuspect = () => {
    if (selectedSuspect) {
      setSuspects(suspects.filter(suspect => suspect.id !== selectedSuspect.id));
      setSelectedSuspect(null);
      setShowDelete(false);
    }
  };

  // Handle changing suspect status
  const handleToggleStatus = (suspect) => {
    setSuspects(suspects.map(s => 
      s.id === suspect.id 
        ? { ...s, status: s.status === 'Active' ? 'Inactive' : 'Active' } 
        : s
    ));
    
    if (selectedSuspect && selectedSuspect.id === suspect.id) {
      setSelectedSuspect({
        ...selectedSuspect,
        status: selectedSuspect.status === 'Active' ? 'Inactive' : 'Active'
      });
    }
  };

  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  return (
    <div className="p-4 md:p-6 bg-gray-50 min-h-screen flex flex-col">
      {/* Header with Breadcrumb */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <div>
          <div className="flex items-center text-sm text-gray-500 mb-1">
            <span className="hover:text-blue-600 cursor-pointer">Dashboard</span>
            <ChevronDown size={16} className="mx-1 transform rotate-270" />
            <span className="text-gray-900">Suspect Management</span>
          </div>
          <h1 className="text-2xl font-bold text-gray-800">Suspect Management</h1>
        </div>
        
        <div className="flex flex-wrap gap-2">
          <button 
            className="bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-600 hover:bg-gray-50 flex items-center space-x-1"
            onClick={() => setShowActivityLog(true)}
          >
            <Clock size={16} />
            <span>Activity Log</span>
          </button>
          <button 
            className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-blue-600 hover:bg-blue-100 flex items-center space-x-1"
            onClick={() => {
              setSelectedSuspect(null);
              handleOpenForm();
            }}
          >
            <UserPlus size={16} />
            <span>Add Suspect</span>
          </button>
        </div>
      </div>
      
      {/* Search & Filter Bar */}
      <div className="bg-white rounded-xl shadow-sm p-4 mb-6">
        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-grow">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search size={16} className="text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search by name, description, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border rounded-lg focus:ring-blue-500 focus:border-blue-500 w-full"
            />
          </div>
          
          <div className="flex gap-2 flex-wrap">
            <button 
              className={`px-3 py-2 rounded-lg border ${showTagFilter ? 'bg-blue-50 border-blue-200 text-blue-600' : 'border-gray-300 text-gray-600 hover:bg-gray-50'} flex items-center gap-1`}
              onClick={() => setShowTagFilter(!showTagFilter)}
            >
              <Tag size={16} />
              <span>Tags</span>
              {selectedTags.length > 0 && (
                <span className="bg-blue-100 text-blue-800 rounded-full w-5 h-5 flex items-center justify-center text-xs">
                  {selectedTags.length}
                </span>
              )}
            </button>
            
            <select
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-gray-600"
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
            >
              <option value="all">All Risk Levels</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
          </div>
        </div>
        
        {/* Tag filter chips */}
        {showTagFilter && (
          <div className="mt-3 flex flex-wrap gap-2">
            {allTags.map(tag => (
              <button
                key={tag}
                className={`px-3 py-1 rounded-full text-sm flex items-center ${
                  selectedTags.includes(tag) 
                    ? 'bg-blue-100 text-blue-800' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => handleToggleTagFilter(tag)}
              >
                {selectedTags.includes(tag) ? (
                  <CheckCircle size={14} className="mr-1" />
                ) : (
                  <Tag size={14} className="mr-1" />
                )}
                {tag}
              </button>
            ))}
          </div>
        )}
        
        {/* Applied filter indicator */}
        {(searchQuery || selectedTags.length > 0 || riskFilter !== 'all') && (
          <div className="mt-3 flex items-center text-sm text-gray-500">
            <Filter size={14} className="mr-1" />
            <span>
              Filters applied: 
              {searchQuery && <span className="font-medium"> Search</span>}
              {selectedTags.length > 0 && <span className="font-medium"> Tags ({selectedTags.length})</span>}
              {riskFilter !== 'all' && <span className="font-medium"> Risk ({riskFilter})</span>}
            </span>
            <button 
              className="ml-2 text-blue-600 hover:text-blue-800"
              onClick={() => {
                setSearchQuery('');
                setSelectedTags([]);
                setRiskFilter('all');
              }}
            >
              Clear all
            </button>
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row gap-6">
        {/* Suspects List */}
        <div className="w-full lg:w-3/5 bg-white rounded-xl shadow-sm overflow-hidden flex flex-col">
          <div className="border-b border-gray-200">
            <div className="flex">
              <button
                className={`px-4 py-3 text-sm font-medium ${
                  activeTab === 'watchlist' 
                    ? 'border-b-2 border-blue-500 text-blue-600 bg-blue-50' 
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
                onClick={() => setActiveTab('watchlist')}
              >
                <div className="flex items-center">
                  <Shield size={16} className="mr-2" />
                  Active Watchlist
                  <span className="ml-2 bg-blue-100 text-blue-800 rounded-full px-2 py-0.5 text-xs">
                    {suspects.filter(s => s.status === 'Active').length}
                  </span>
                </div>
              </button>
              <button
                className={`px-4 py-3 text-sm font-medium ${
                  activeTab === 'archive' 
                    ? 'border-b-2 border-blue-500 text-blue-600 bg-blue-50' 
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
                onClick={() => setActiveTab('archive')}
              >
                <div className="flex items-center">
                  <Archive size={16} className="mr-2" />
                  Archive
                  <span className="ml-2 bg-gray-100 text-gray-600 rounded-full px-2 py-0.5 text-xs">
                    {suspects.filter(s => s.status === 'Inactive').length}
                  </span>
                </div>
              </button>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {filteredSuspects.length > 0 ? (
              <ul className="divide-y divide-gray-100">
                {filteredSuspects.map(suspect => (
                  <li 
                    key={suspect.id}
                    className={`hover:bg-blue-50 cursor-pointer ${selectedSuspect?.id === suspect.id ? 'bg-blue-50' : ''}`}
                    onClick={() => {
                      setSelectedSuspect(suspect);
                      setShowForm(false);
                      setActiveView('details');
                    }}
                  >
                    <div className="flex p-4">
                      <div className="bg-gray-100 w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0">
                        {suspect.image ? (
                          <img 
                            src={suspect.image} 
                            alt={suspect.name} 
                            className="w-full h-full object-cover rounded-lg" 
                          />
                        ) : (
                          <UserX size={24} className="text-gray-400" />
                        )}
                      </div>
                      <div className="ml-4 flex-1">
                        <div className="flex justify-between">
                          <h3 className="font-medium text-gray-900 flex items-center">
                            {suspect.name}
                            {suspect.status === 'Active' ? (
                              <span className="ml-2 w-2 h-2 bg-green-500 rounded-full" title="Active"></span>
                            ) : (
                              <span className="ml-2 w-2 h-2 bg-gray-400 rounded-full" title="Inactive"></span>
                            )}
                          </h3>
                          <div className="flex items-center">
                            <span className={`px-2 py-1 rounded-full text-xs flex items-center ${getRiskColor(suspect.risk)}`}>
                              {getRiskIcon(suspect.risk)}
                              <span className="ml-1">{suspect.risk} Risk</span>
                            </span>
                          </div>
                        </div>
                        <div className="flex justify-between">
                          <p className="text-sm text-gray-500 mt-1">
                            <span className="inline-flex items-center">
                              <Map size={14} className="mr-1 text-gray-400" />
                              {suspect.location}
                            </span>
                            <span className="mx-2">•</span>
                            <span className="inline-flex items-center">
                              <Clock size={14} className="mr-1 text-gray-400" />
                              {formatRelativeTime(suspect.lastSeen)}
                            </span>
                          </p>
                          <p className="text-xs text-orange-600 bg-orange-50 px-2 py-0.5 rounded-full">
                            {suspect.incidents} {suspect.incidents === 1 ? 'incident' : 'incidents'}
                          </p>
                        </div>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-1">{suspect.description}</p>
                        <div className="flex flex-wrap items-center mt-2 gap-1">
                          {suspect.tags.slice(0, 3).map((tag, index) => (
                            <span key={index} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                              {tag}
                            </span>
                          ))}
                          {suspect.tags.length > 3 && (
                            <span className="text-xs text-gray-500">
                              +{suspect.tags.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex flex-col items-center justify-center h-full p-8">
                <div className="bg-gray-100 p-4 rounded-full">
                  <UserX size={36} className="text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-700 mt-4">No Suspects Found</h3>
                <p className="text-gray-500 text-center mt-2 max-w-sm">
                  {searchQuery || selectedTags.length > 0 || riskFilter !== 'all'
                    ? 'No suspects match your current filters. Try adjusting your search criteria.'
                    : activeTab === 'watchlist'
                    ? 'There are no active suspects on the watchlist.'
                    : 'There are no suspects in the archive.'}
                </p>
                {(searchQuery || selectedTags.length > 0 || riskFilter !== 'all') && (
                  <button 
                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    onClick={() => {
                      setSearchQuery('');
                      setSelectedTags([]);
                      setRiskFilter('all');
                    }}
                  >
                    Clear Filters
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Right Panel - Details or Form */}
        <div className="w-full lg:w-2/5 bg-white rounded-xl shadow-sm overflow-hidden flex flex-col">
          {showForm ? (
            <>
              <div className="px-6 py-4 border-b border-gray-200 font-medium flex justify-between items-center bg-gray-50">
                <span className="text-lg">{selectedSuspect ? 'Edit Suspect' : 'Add New Suspect'}</span>
                <button 
                  className="p-1 hover:bg-gray-200 rounded-full"
                  onClick={() => setShowForm(false)}
                >
                  <X size={20} className="text-gray-500" />
                </button>
              </div>
              
              <div className="p-6 overflow-y-auto flex-1">
                <form onSubmit={handleSubmitForm} className="space-y-5">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Name/Identifier
                    </label>
                    <input 
                      type="text" 
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      placeholder="e.g., Unknown Subject #44"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Description
                    </label>
                    <textarea 
                      name="description"
                      value={formData.description}
                      onChange={handleInputChange}
                      className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-24"
                      placeholder="Physical description, clothing, etc."
                      required
                    ></textarea>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Risk Level
                      </label>
                      <select 
                        name="risk"
                        value={formData.risk}
                        onChange={handleInputChange}
                        className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="Low">Low</option>
                        <option value="Medium">Medium</option>
                        <option value="High">High</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Status
                      </label>
                      <select 
                        name="status"
                        value={formData.status}
                        onChange={handleInputChange}
                        className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="Active">Active</option>
                        <option value="Inactive">Inactive</option>
                      </select>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Last Seen Location
                    </label>
                    <input 
                      type="text" 
                      name="location"
                      value={formData.location}
                      onChange={handleInputChange}
                      className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      placeholder="e.g., Electronics Department"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Incident Count
                    </label>
                    <input 
                      type="number" 
                      name="incidents"
                      value={formData.incidents}
                      onChange={handleInputChange}
                      min="0"
                      className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Tags
                    </label>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {formData.tags.map((tag, index) => (
                        <span key={index} className="bg-blue-100 text-blue-800 px-2.5 py-1 rounded-full text-sm flex items-center gap-1">
                          {tag}
                          <button
                            type="button"
                            className="text-blue-600 hover:text-blue-800"
                            onClick={() => handleRemoveTag(tag)}
                          >
                            <X size={14} />
                          </button>
                        </span>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <div className="relative flex-grow">
                        <input
                          type="text"
                          value={newTag}
                          onChange={(e) => setNewTag(e.target.value)}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              handleAddTag();
                            }
                          }}
                          placeholder="Add a tag..."
                          className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                        {newTag && (
                          <div className="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10 max-h-32 overflow-y-auto">
                            {tagSuggestions
                              .filter(tag => 
                                tag.toLowerCase().includes(newTag.toLowerCase()) && 
                                !formData.tags.includes(tag)
                              )
                              .map((tag, index) => (
                                <div 
                                  key={index}
                                  className="px-3 py-2 hover:bg-blue-50 cursor-pointer"
                                  onClick={() => {
                                    setNewTag(tag);
                                    handleAddTag();
                                  }}
                                >
                                  {tag}
                                </div>
                              ))}
                          </div>
                        )}
                      </div>
                      <button
                        type="button"
                        className="bg-blue-600 text-white px-4 py-2.5 rounded-lg hover:bg-blue-700"
                        onClick={handleAddTag}
                      >
                        Add
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Security Notes
                    </label>
                    <textarea 
                      name="notes"
                      value={formData.notes}
                      onChange={handleInputChange}
                      className="w-full p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-24"
                      placeholder="Additional notes or observations..."
                    ></textarea>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Photo (Optional)
                    </label>
                    <div className="flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 bg-gray-50">
                      <div className="space-y-1 text-center">
                        <Camera className="mx-auto text-gray-400" size={30} />
                        <div className="text-sm text-gray-600">
                          <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
                            <span>Upload a file</span>
                            <input id="file-upload" name="file-upload" type="file" className="sr-only" />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex gap-3 pt-4">
                    <button 
                      type="submit" 
                      className="bg-blue-600 text-white px-5 py-2.5 rounded-lg hover:bg-blue-700 flex-1 flex items-center justify-center gap-2"
                    >
                      {selectedSuspect ? <Edit size={18} /> : <UserPlus size={18} />}
                      {selectedSuspect ? 'Update Suspect' : 'Add Suspect'}
                    </button>
                    <button 
                      type="button" 
                      className="bg-white border border-gray-300 text-gray-700 px-5 py-2.5 rounded-lg hover:bg-gray-50 flex items-center justify-center gap-2"
                      onClick={() => setShowForm(false)}
                    >
                      <X size={18} />
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            </>
          ) : selectedSuspect ? (
            <>
              <div className="border-b border-gray-200 bg-white">
                <div className="flex overflow-x-auto">
                  <button
                    className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                      activeView === 'details' 
                        ? 'border-b-2 border-blue-500 text-blue-600' 
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                    onClick={() => setActiveView('details')}
                  >
                    <User size={16} className="inline mr-1" />
                    Details
                  </button>
                  <button
                    className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                      activeView === 'history' 
                        ? 'border-b-2 border-blue-500 text-blue-600' 
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                    onClick={() => setActiveView('history')}
                  >
                    <Clock size={16} className="inline mr-1" />
                    History
                  </button>
                  <button
                    className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                      activeView === 'stats' 
                        ? 'border-b-2 border-blue-500 text-blue-600' 
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                    onClick={() => setActiveView('stats')}
                  >
                    <BarChart4 size={16} className="inline mr-1" />
                    Statistics
                  </button>
                </div>
              </div>
              
              <div className="p-6 overflow-y-auto flex-1">
                {activeView === 'details' && (
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex gap-4 items-center">
                        <div className="bg-gray-100 w-16 h-16 rounded-lg flex items-center justify-center">
                          {selectedSuspect.image ? (
                            <img 
                              src={selectedSuspect.image} 
                              alt={selectedSuspect.name} 
                              className="w-full h-full object-cover rounded-lg" 
                            />
                          ) : (
                            <UserX size={30} className="text-gray-400" />
                          )}
                        </div>
                        <div>
                          <h2 className="text-xl font-semibold">{selectedSuspect.name}</h2>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`px-2 py-1 rounded-full text-xs flex items-center ${getRiskColor(selectedSuspect.risk)}`}>
                              {getRiskIcon(selectedSuspect.risk)}
                              <span className="ml-1">{selectedSuspect.risk} Risk</span>
                            </span>
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              selectedSuspect.status === 'Active' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-gray-100 text-gray-800'
                            }`}>
                              {selectedSuspect.status}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <button 
                          className="p-2 hover:bg-gray-100 rounded-lg text-gray-500 hover:text-gray-700"
                          onClick={() => handleOpenForm(selectedSuspect)}
                          title="Edit"
                        >
                          <Edit size={18} />
                        </button>
                        <button 
                          className="p-2 hover:bg-red-50 rounded-lg text-gray-500 hover:text-red-600"
                          onClick={() => setShowDelete(true)}
                          title="Delete"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </div>
                    
                    <div className="mt-6 space-y-5">
                      <div>
                        <h3 className="text-sm font-medium text-gray-500">Description</h3>
                        <p className="mt-1 text-gray-800">{selectedSuspect.description}</p>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h3 className="text-sm font-medium text-gray-500">Last Seen</h3>
                          <p className="mt-1 text-gray-800 flex items-center">
                            <Calendar size={14} className="mr-2 text-gray-400" />
                            {formatDate(selectedSuspect.lastSeen)}
                          </p>
                        </div>
                        <div>
                          <h3 className="text-sm font-medium text-gray-500">Location</h3>
                          <p className="mt-1 text-gray-800 flex items-center">
                            <Map size={14} className="mr-2 text-gray-400" />
                            {selectedSuspect.location}
                          </p>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium text-gray-500">Incident Count</h3>
                        <div className="mt-1 flex items-center">
                          <div className={`px-3 py-1.5 rounded-lg ${
                            selectedSuspect.incidents > 2
                              ? 'bg-red-100 text-red-800'
                              : selectedSuspect.incidents > 0
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            <span className="text-lg font-semibold">{selectedSuspect.incidents}</span>
                            <span className="text-sm ml-1">
                              {selectedSuspect.incidents === 1 ? 'incident' : 'incidents'}
                            </span>
                          </div>
                          {selectedSuspect.incidents > 2 && (
                            <span className="ml-2 text-sm text-red-600">High incident rate</span>
                          )}
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium text-gray-500">Tags</h3>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {selectedSuspect.tags.map((tag, index) => (
                            <span key={index} className="flex items-center bg-gray-100 text-gray-700 px-3 py-1.5 rounded-full text-sm">
                              <Tag size={14} className="mr-1 text-gray-500" />
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      {selectedSuspect.notes && (
                        <div>
                          <h3 className="text-sm font-medium text-gray-500">Security Notes</h3>
                          <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-gray-800">
                            {selectedSuspect.notes}
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className="mt-8 pt-6 border-t border-gray-200">
                      <h3 className="text-sm font-medium text-gray-500 mb-3">Actions</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <button className="flex items-center justify-center gap-2 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                          <Bell size={16} />
                          Set Alert
                        </button>
                        <button 
                          className="flex items-center justify-center gap-2 py-2.5 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                          onClick={() => handleToggleStatus(selectedSuspect)}
                        >
                          {selectedSuspect.status === 'Active' ? (
                            <>
                              <Archive size={16} />
                              Move to Archive
                            </>
                          ) : (
                            <>
                              <Shield size={16} />
                              Move to Active
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                
                {activeView === 'history' && (
                  <div>
                    <h3 className="text-lg font-medium mb-4">Detection History</h3>
                    {selectedSuspect.detections && selectedSuspect.detections.length > 0 ? (
                      <div className="border border-gray-200 rounded-lg overflow-hidden">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Location</th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Camera</th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {selectedSuspect.detections.map((detection, index) => (
                              <tr key={index} className="hover:bg-gray-50">
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-800">
                                  {formatDate(detection.date)}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-800">
                                  {detection.location}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-800">
                                  {detection.camera}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap">
                                  <div className="flex items-center">
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                      <div 
                                        className={`h-2.5 rounded-full ${
                                          detection.confidence >= 90 
                                            ? 'bg-green-500' 
                                            : detection.confidence >= 75 
                                            ? 'bg-yellow-500' 
                                            : 'bg-red-500'
                                        }`} 
                                        style={{ width: `${detection.confidence}%` }}
                                      ></div>
                                    </div>
                                    <span className="ml-2 text-sm text-gray-700">{detection.confidence}%</span>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="text-center py-8 bg-gray-50 rounded-lg">
                        <Clock size={36} className="mx-auto text-gray-400 mb-2" />
                        <p className="text-gray-500">No detection history available</p>
                      </div>
                    )}
                  </div>
                )}
                
                {activeView === 'stats' && (
                  <div>
                    <h3 className="text-lg font-medium mb-4">Suspect Statistics</h3>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="bg-blue-50 border border-blue-100 rounded-lg p-4">
                        <p className="text-sm text-blue-700">Detection Rate</p>
                        <p className="text-2xl font-bold text-blue-800 mt-1">
                          {selectedSuspect.detections?.length || 0}
                        </p>
                        <p className="text-xs text-blue-600 mt-1">
                          Last 30 days
                        </p>
                      </div>
                      <div className="bg-orange-50 border border-orange-100 rounded-lg p-4">
                        <p className="text-sm text-orange-700">Average Confidence</p>
                        <p className="text-2xl font-bold text-orange-800 mt-1">
                          {selectedSuspect.detections?.length 
                            ? Math.round(selectedSuspect.detections.reduce((sum, det) => sum + det.confidence, 0) / selectedSuspect.detections.length)
                            : 0}%
                        </p>
                        <p className="text-xs text-orange-600 mt-1">
                          AI Detection Confidence
                        </p>
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Location Frequency</h4>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        {selectedSuspect.detections?.length ? (
                          <div className="space-y-3">
                            {Array.from(new Set(selectedSuspect.detections.map(d => d.location)))
                              .map(location => {
                                const count = selectedSuspect.detections.filter(d => d.location === location).length;
                                const percentage = Math.round((count / selectedSuspect.detections.length) * 100);
                                
                                return (
                                  <div key={location}>
                                    <div className="flex justify-between mb-1">
                                      <span className="text-sm text-gray-700">{location}</span>
                                      <span className="text-sm text-gray-500">{count} detections</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                      <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${percentage}%` }}></div>
                                    </div>
                                  </div>
                                );
                              })
                            }
                          </div>
                        ) : (
                          <p className="text-center py-4 text-gray-500">No location data available</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full p-8 bg-gray-50">
              <div className="bg-white p-4 rounded-full shadow-sm mb-4">
                <UserX size={36} className="text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-700">No Suspect Selected</h3>
              <p className="text-gray-500 text-center mt-2 max-w-xs">
                Select a suspect from the list to view details or add a new suspect
              </p>
              <button 
                className="mt-6 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                onClick={() => handleOpenForm()}
              >
                <UserPlus size={16} />
                Add New Suspect
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Activity Log Modal */}
      {showActivityLog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
              <h2 className="text-lg font-semibold">Activity Log</h2>
              <button 
                className="p-1 hover:bg-gray-100 rounded-full"
                onClick={() => setShowActivityLog(false)}
              >
                <X size={20} className="text-gray-500" />
              </button>
            </div>
            <div className="overflow-y-auto p-4 flex-1">
              <div className="space-y-4">
                {activityLogData.map((entry, index) => (
                  <div key={index} className="flex gap-3">
                    <div className="mt-1 w-9 h-9 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                      {entry.action === 'Suspect Added' && <UserPlus size={16} className="text-blue-600" />}
                      {entry.action === 'Status Change' && <RefreshCw size={16} className="text-purple-600" />}
                      {entry.action === 'New Detection' && <Eye size={16} className="text-green-600" />}
                      {entry.action === 'Notes Updated' && <Edit size={16} className="text-orange-600" />}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">{entry.action}</p>
                      <p className="text-sm text-gray-500">{entry.details}</p>
                      <div className="flex items-center mt-1 text-xs text-gray-500">
                        <Clock size={12} className="mr-1" />
                        {formatDate(entry.timestamp)}
                        <span className="mx-1">•</span>
                        <User size={12} className="mr-1" />
                        {entry.user}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-4 border-t border-gray-200">
              <button 
                className="w-full py-2 bg-gray-100 text-gray-800 rounded-lg hover:bg-gray-200"
                onClick={() => setShowActivityLog(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Delete Confirmation Modal */}
      {showDelete && selectedSuspect && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-6">
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-red-100 mx-auto mb-4">
              <AlertTriangle size={24} className="text-red-600" />
            </div>
            <h2 className="text-lg font-semibold text-center mb-2">Delete Suspect</h2>
            <p className="text-gray-600 text-center mb-6">
              Are you sure you want to delete <span className="font-medium">{selectedSuspect.name}</span>? This action cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                className="flex-1 py-2.5 bg-gray-100 text-gray-800 rounded-lg hover:bg-gray-200"
                onClick={() => setShowDelete(false)}
              >
                Cancel
              </button>
              <button
                className="flex-1 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700"
                onClick={handleDeleteSuspect}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Component for Archive icon
const Archive = ({ size, className }) => {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect width="20" height="5" x="2" y="3" rx="1" />
      <path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" />
      <path d="M10 12h4" />
    </svg>
  );
};

// Component for RefreshCw icon
const RefreshCw = ({ size, className }) => {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
      <path d="M3 21v-5h5" />
    </svg>
  );
};

export default SuspectList;