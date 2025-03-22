<template>
  <div class="chat-layout">
    <Sidebar />
    
    <div class="chat-container">
      <div class="chat-header">
        <h1>🇪🇺 Document Chat</h1>
      </div>
      
      <div class="chat-messages" ref="messagesContainer">
        <p v-if="!chatStore.messages.length" class="empty-state">
          Ask a question about your documents...
        </p>
        
        <ChatMessage
          v-for="message in chatStore.messages"
          :key="message.id"
          :message="message"
        />
        
        <div v-if="chatStore.isLoading" class="loading-indicator">
          <Loading text="Thinking..." />
        </div>
      </div>
      
      <div class="chat-input">
        <ChatInput @send="sendMessage" :disabled="chatStore.isLoading" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useChatStore } from '../stores/chat';
import ChatMessage from '../components/chat/ChatMessage.vue';
import ChatInput from '../components/chat/ChatInput.vue';
import Sidebar from '../components/layout/Sidebar.vue';
import Loading from '../components/shared/Loading.vue';

const chatStore = useChatStore();
const messagesContainer = ref(null);

onMounted(() => {
  // Check system status on mount
  chatStore.checkSystemStatus();
  
  // Scroll to bottom on mount
  scrollToBottom();
});

// Watch for new messages and scroll to bottom
watch(
  () => chatStore.messages.length,
  () => {
    scrollToBottom();
  }
);

function scrollToBottom() {
  setTimeout(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    }
  }, 100);
}

async function sendMessage(content) {
  if (!content.trim()) return;
  
  try {
    await chatStore.sendMessage(content);
    // Message added to store in action
  } catch (error) {
    console.error('Failed to send message:', error);
    // Error handled in store action
  }
}
</script>

<style scoped>
.chat-layout {
  display: flex;
  height: 100vh;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.chat-header {
  padding: 16px;
  border-bottom: 1px solid #eaeaea;
}

.chat-header h1 {
  margin: 0;
  font-size: 24px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.chat-input {
  border-top: 1px solid #eaeaea;
  padding: 16px;
}

.empty-state {
  text-align: center;
  color: #888;
  margin-top: 40px;
}

.loading-indicator {
  display: flex;
  justify-content: center;
  margin: 16px 0;
}
</style>