<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <h3>Document Chat</h3>
    </div>
    
    <div class="sidebar-content">
      <div class="system-status">
        <h4>System Status</h4>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(chatStore.systemStatus.api)]"></span>
          <span>API Service: {{ chatStore.systemStatus.api }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(chatStore.systemStatus.weaviate)]"></span>
          <span>Vector Database: {{ chatStore.systemStatus.weaviate }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(chatStore.systemStatus.mistral_api)]"></span>
          <span>LLM Service: {{ chatStore.systemStatus.mistral_api }}</span>
        </div>
      </div>
      
      <div class="sidebar-actions">
        <button class="action-button" @click="newConversation">
          🔄 New Conversation
        </button>
        <button class="action-button logout" @click="logout">
          🚪 Logout
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore } from '../../stores/chat';
import authService from '../../services/authService';

const router = useRouter();
const chatStore = useChatStore();

onMounted(() => {
  chatStore.checkSystemStatus();
});

function getStatusClass(status) {
  if (status === 'connected' || status === 'configured') {
    return 'status-success';
  } else if (status === 'error' || status === 'disconnected') {
    return 'status-error';
  }
  return 'status-unknown';
}

function newConversation() {
  chatStore.clearChat();
}

function logout() {
  authService.logout();
  router.push('/login');
}
</script>

<style scoped>
.sidebar {
  width: 300px;
  height: 100%;
  background-color: #f5f7fb;
  border-right: 1px solid #eaeaea;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #eaeaea;
}

.sidebar-header h3 {
  margin: 0;
}

.sidebar-content {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.system-status {
  margin-bottom: 24px;
}

.system-status h4 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
}

.status-item {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
}

.status-success {
  background-color: #48bb78;
}

.status-error {
  background-color: #e53e3e;
}

.status-unknown {
  background-color: #a0aec0;
}

.sidebar-actions {
  margin-top: auto;
}

.action-button {
  width: 100%;
  padding: 10px;
  margin-bottom: 8px;
  border: none;
  border-radius: 4px;
  background-color: #4a6cf7;
  color: white;
  cursor: pointer;
  font-size: 14px;
  text-align: left;
}

.action-button:hover {
  background-color: #3a5cd7;
}

.action-button.logout {
  background-color: #e53e3e;
}

.action-button.logout:hover {
  background-color: #c53030;
}
</style>