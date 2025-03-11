document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    
    // Initialize config
    const config = window.config || { apiUrl: '/api' };
    const API_URL = config.apiUrl;
    
    function addMessage(content, isUser = false, sources = []) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        // Main content
        const contentElement = document.createElement('div');
        contentElement.className = 'content';
        contentElement.innerHTML = content.replace(/\n/g, '<br>');
        messageElement.appendChild(contentElement);
        
        // Add sources if provided
        if (sources && sources.length > 0) {
            const sourcesElement = document.createElement('div');
            sourcesElement.className = 'sources';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.textContent = 'Sources:';
            sourcesTitle.style.fontWeight = 'bold';
            sourcesElement.appendChild(sourcesTitle);
            
            sources.forEach(source => {
                const sourceText = document.createElement('div');
                let text = `• ${source.filename} (Chunk ${source.chunkId})`;
                
                // Add metadata if available
                if (source.metadata) {
                    const metadata = source.metadata;
                    if (metadata.title) {
                        text += ` - ${metadata.title}`;
                    }
                    if (metadata.itemType) {
                        text += ` [${metadata.itemType}]`;
                    }
                    
                    // Handle creators/authors
                    if (metadata.creators && metadata.creators.length > 0) {
                        const authors = metadata.creators
                            .filter(creator => creator.creatorType === 'author')
                            .map(creator => `${creator.lastName || ''}, ${creator.firstName || ''}`.trim())
                            .filter(name => name !== ', ');
                        
                        if (authors.length > 0) {
                            text += ` by ${authors.slice(0, 2).join(', ')}`;
                            if (authors.length > 2) {
                                text += ' et al.';
                            }
                        }
                    }
                    
                    // Add date if available
                    if (metadata.date) {
                        text += ` (${metadata.date})`;
                    }
                }
                
                sourceText.textContent = text;
                sourcesElement.appendChild(sourceText);
            });
            
            messageElement.appendChild(sourcesElement);
        }
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator message assistant-message';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        console.log("Sending message:", message);
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input
        messageInput.value = "";
        
        // Show typing indicator
        addTypingIndicator();
        
        try {
            // Send to API with longer timeout
            console.log("Sending request to:", `${API_URL}/chat`);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60-second timeout
            
            const response = await fetch(`${API_URL}/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ question: message }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            // Remove typing indicator
            removeTypingIndicator();
            
            console.log("Response status:", response.status);
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Response data:", data);
            
            // Add assistant response to chat
            addMessage(data.answer, false, data.sources);
            
        } catch (error) {
            console.error("Error sending message:", error);
            removeTypingIndicator();
            
            // Add error message
            const errorElement = document.createElement("div");
            errorElement.className = "error-message";
            errorElement.textContent = `Error: ${error.message}. Please try again.`;
            chatMessages.appendChild(errorElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }    
    // Set up event listeners
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add initial welcome message
    addMessage("Welcome to Document Chat! Ask a question about your documents.");
});