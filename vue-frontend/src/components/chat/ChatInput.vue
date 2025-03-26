<template>
    <div class="chat-input-container">
      <input
        type="text"
        v-model="message"
        @keyup.enter="sendMessage"
        placeholder="Ask a question about your documents..."
        :disabled="disabled"
      />
      <button
        class="send-button"
        @click="sendMessage"
        :disabled="!message.trim() || disabled"
      >
        <span>Send</span>
      </button>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  
  const props = defineProps({
    disabled: {
      type: Boolean,
      default: false
    }
  });
  
  const emit = defineEmits(['send']);
  const message = ref('');
  
  function sendMessage() {
    if (!message.value.trim() || props.disabled) return;
    
    emit('send', message.value);
    message.value = '';
  }
  </script>
  
  <style scoped>
  .chat-input-container {
    display: flex;
    gap: 8px;
    width: 100%;
  }
  
  input {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
  }
  
  input:focus {
    outline: none;
    border-color: #4a6cf7;
  }
  
  input:disabled {
    background-color: #f5f5f5;
    cursor: not-allowed;
  }
  
  .send-button {
    padding: 0 20px;
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .send-button:hover:not(:disabled) {
    background-color: #3a5cd7;
  }
  
  .send-button:disabled {
    background-color: #a0aec0;
    cursor: not-allowed;
  }
  </style>