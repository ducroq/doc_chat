<template>
  <div class="chat-layout">
    <Sidebar />
    
    <div class="chat-container">
      <div class="chat-header">
        <h1>🇪🇺 Document Chat</h1>
        <button class="print-button" @click="printChat" title="Print Chat">
          🖨️ Print
        </button>
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

function formatCitation(source) {
  // Extract filename and chunkId
  const filename = source.filename || 'Unknown';
  const chunkId = source.chunkId || 'Unknown';
  
  // If we have metadata, use it to create a richer citation
  if (source.metadata) {
    const metadata = source.metadata;
    let citation = metadata.title ? `${metadata.title}` : filename;
    
    if (metadata.itemType) {
      citation += ` [${metadata.itemType}]`;
    }
    
    // Add authors if available
    if (metadata.creators && metadata.creators.length > 0) {
      const authors = metadata.creators
        .filter(c => c.creatorType === 'author')
        .map(a => `${a.lastName}, ${a.firstName}`);
      
      if (authors.length > 0) {
        citation += ` by ${authors.join(', ')}`;
      }
    }
    
    // Add date if available
    if (metadata.date) {
      citation += ` (${metadata.date})`;
    }
    
    // Add page if available
    if (source.page) {
      citation += `, Page ${source.page}`;
    }
    
    return citation;
  }
  
  // Simple citation without metadata
  return `${filename} (Chunk ${chunkId})`;
}

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

function printChat() {
  // Create a printable version of the chat
  const printContent = document.createElement('div');
  printContent.innerHTML = `
    <h1>Document Chat Export</h1>
    <p>Generated on: ${new Date().toLocaleString()}</p>
    <hr>
    ${chatStore.messages.map(msg => {
      const role = msg.role === 'user' ? 'You' : 'Assistant';
      let html = `<p><strong>${role}:</strong> ${msg.content}</p>`;
      
      if (msg.sources && msg.sources.length > 0) {
        html += '<p><strong>Sources:</strong></p><ul>';
        msg.sources.forEach(source => {
          // Use the same formatting as on screen
          html += `<li>${formatCitation(source)}</li>`;
        });
        html += '</ul>';
      }
      
      return html + '<hr>';
    }).join('')}
  `;
  
  // Create a new window for printing
  const printWindow = window.open('', '_blank');
  printWindow.document.write(`
    <html>
      <head>
        <title>Chat Export</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          hr { border: 0; border-top: 1px solid #eee; margin: 20px 0; }
          ul { margin-top: 10px; padding-left: 20px; }
          li { margin-bottom: 5px; }
        </style>
      </head>
      <body>${printContent.innerHTML}</body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus(); // Focus on the new window
  setTimeout(() => {
    printWindow.print();
  }, 500); // Short delay to ensure content is fully loaded
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

.print-button {
  padding: 8px 12px;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.print-button:hover {
  background-color: #3a5cd7;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
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