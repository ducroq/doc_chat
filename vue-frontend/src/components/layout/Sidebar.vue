<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <h3>Document Chat</h3>
    </div>
    
    <div class="sidebar-content">
      <div class="system-status">
        <h4>System Status</h4>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.api)]"></span>
          <span>API Service: {{ systemStatus.api }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.weaviate)]"></span>
          <span>Vector Database: {{ systemStatus.weaviate }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.mistral_api)]"></span>
          <span>LLM Service: {{ systemStatus.mistral_api }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', loggingEnabled ? 'status-warning' : 'status-success']"></span>
          <span>Chat Logging: {{ loggingEnabled ? 'Enabled' : 'Disabled' }}</span>
          <span v-if="loggingEnabled" class="logging-warning" @click="showPrivacyInfo">⚠️</span>
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
    <div v-if="showPrivacyModal" class="privacy-modal">
      <div class="privacy-content">
        <h4>Chat Logging Information</h4>
        <p>This system is currently logging chat interactions for research purposes.</p>
        <p>All logs are anonymized and automatically deleted after 30 days.</p>
        <p>For more information, please see the <a href="/privacy" target="_blank">Privacy Notice</a>.</p>
        <button @click="showPrivacyModal = false">Close</button>
      </div>
    </div>    
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore } from '../../stores/chat';
import authService from '../../services/authService';

const router = useRouter();
const chatStore = useChatStore();
const systemStatus = ref({
  api: 'unknown',
  weaviate: 'unknown',
  mistral_api: 'unknown'
});

const loggingEnabled = ref(false);
const showPrivacyModal = ref(false);

onMounted(async () => {
  // Set up config check
  let configCheckInterval = null;
  
  function checkConfig() {
    if (window.APP_CONFIG && typeof window.APP_CONFIG.enableChatLogging !== 'undefined') {
      console.log("Found APP_CONFIG", window.APP_CONFIG);
      loggingEnabled.value = window.APP_CONFIG.enableChatLogging === true || 
                            window.APP_CONFIG.enableChatLogging === "true";
      
      // Clear interval once config is found
      if (configCheckInterval) {
        clearInterval(configCheckInterval);
        configCheckInterval = null;
      }
    }
  }
  
  // Check immediately
  checkConfig();
  
  // Set up interval to check periodically
  configCheckInterval = setInterval(checkConfig, 100);
  
  // Clean up after 2 seconds max
  setTimeout(() => {
    if (configCheckInterval) {
      clearInterval(configCheckInterval);
      configCheckInterval = null;
    }
  }, 2000);
  
  // Fetch system status (independent of config check)
  try {
    const status = await chatStore.checkSystemStatus();
    if (status) {
      systemStatus.value = status;
    }
  } catch (error) {
    console.warn('Could not fetch system status:', error);
  }
});

function showPrivacyInfo() {
  showPrivacyModal.value = true;
}

function getStatusClass(status) {
  if (status === 'connected' || status === 'configured' || status === 'running') {
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

.status-warning {
  background-color: #f6ad55;
}

.logging-warning {
  margin-left: 8px;
  cursor: pointer;
}

/* Modal Styles */
.privacy-modal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 100;
}

.privacy-content {
  background-color: white;
  padding: 16px;
  border-radius: 8px;
  max-width: 80%;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.privacy-content h4 {
  margin-top: 0;
  margin-bottom: 12px;
}

.privacy-content button {
  margin-top: 12px;
  padding: 8px 16px;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.privacy-content button:hover {
  background-color: #3a5cd7;
}
</style>