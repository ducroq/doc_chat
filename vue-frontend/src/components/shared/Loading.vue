<template>
    <div class="loading-container">
      <div class="loading-spinner"></div>
      <span v-if="text" class="loading-text">{{ text }}</span>
    </div>
  </template>
  
  <script setup>
  defineProps({
    text: {
      type: String,
      default: 'Loading...'
    }
  });
  </script>
  
  <style scoped>
  .loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  
  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-radius: 50%;
    border-top: 4px solid #4a6cf7;
    animation: spin 1s linear infinite;
  }
  
  .loading-text {
    margin-top: 8px;
    color: #666;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  </style>