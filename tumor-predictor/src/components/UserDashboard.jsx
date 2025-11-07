import React, { useState, useEffect } from 'react'
import { User, Activity, TrendingUp, AlertCircle, Settings } from 'lucide-react'

const UserDashboard = ({ currentUser, userDashboard, onNavigateToProfile }) => {
  if (!currentUser) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Loading User Profile...</h3>
        <p className="text-gray-600">Please wait while we load your dashboard data.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* User Profile Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-lg p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <User className="text-white" size={32} />
            </div>
            <div>
              <h2 className="text-2xl font-bold">{currentUser.full_name}</h2>
              <p className="text-blue-100">{currentUser.specialization} • {currentUser.hospital_affiliation}</p>
              <p className="text-sm text-blue-200">License: {currentUser.license_number}</p>
            </div>
          </div>
          <button
            onClick={() => {
              if (onNavigateToProfile) {
                onNavigateToProfile();
              } else {
                console.log('Profile button clicked');
                alert('Profile button is now active! This would typically navigate to the profile section.');
              }
            }}
            className="px-6 py-3 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-all duration-200 flex items-center space-x-2 text-white font-medium shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
          >
            <User className="text-white" size={18} />
            <span>View Profile</span>
          </button>
        </div>
      </div>

      {/* User Statistics removed as requested */}

      {/* Treatment Effectiveness Chart */}
      {userDashboard?.statistics?.treatment_effectiveness && Object.keys(userDashboard.statistics.treatment_effectiveness).length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Treatment Effectiveness</h3>
          <div className="space-y-3">
            {Object.entries(userDashboard.statistics.treatment_effectiveness).map(([treatment, stats]) => (
              <div key={treatment} className="bg-gray-50 p-4 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium text-lg">{treatment}</span>
                  <span className="text-sm text-gray-600">{stats.effectiveness}% effective</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all duration-300" 
                    style={{ width: `${stats.effectiveness}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Total: {stats.total}</span>
                  <span>Excellent: {stats.excellent} | Good: {stats.good}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      {userDashboard?.recent_activity && userDashboard.recent_activity.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Activity</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {userDashboard.recent_activity.map((activity, idx) => (
              <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Activity className="text-blue-600" size={16} />
                  <span className="text-gray-700 capitalize">{activity.action_type}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">{activity.patient_id}</div>
                  <div className="text-xs text-gray-400">{new Date(activity.created_at).toLocaleDateString()}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default UserDashboard
