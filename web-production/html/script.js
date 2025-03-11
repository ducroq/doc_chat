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
    
    function addErrorMessage(message) {
        const errorElement = document.createElement("div");
        errorElement.className = "error-message";
        errorElement.textContent = message;
        chatMessages.appendChild(errorElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        console.log("Sending message:", message);
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input
        messageInput.value = "";
        
        // Show typing indicator
        addTypingIndicator();
        
        // Using regular promises instead of async/await to avoid the message channel error
        console.log("Sending request to:", `${API_URL}/chat`);
        
        // Set up the fetch request
        fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question: message })
        })
        .then(response => {
            console.log("Response status:", response.status);
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status} ${response.statusText}`);
            }
            
            return response.json();
        })
        .then(data => {
            console.log("Response data:", data);
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add assistant response to chat
            if (data && data.answer) {
                addMessage(data.answer, false, data.sources || []);
            } else {
                throw new Error("Received an empty or invalid response");
            }
        })
        .catch(error => {
            console.error("Error sending message:", error);
            removeTypingIndicator();
            
            // Add different error messages based on the error type
            let errorMessage = "An error occurred. Please try again.";
            
            if (error.message.includes("timeout")) {
                errorMessage = "The request took too long to complete. The server might be busy processing your question.";
            } else if (error.message.includes("NetworkError")) {
                errorMessage = "Network error. Please check your connection and try again.";
            } else if (error.message.includes("SyntaxError") || error.message.includes("parse")) {
                errorMessage = "Received an invalid response from the server. The system might be temporarily overloaded.";
            } else if (error.message.includes("404")) {
                errorMessage = "API endpoint not found. Please contact the administrator.";
            } else if (error.message.includes("503")) {
                errorMessage = "The service is temporarily unavailable. Weaviate might still be initializing.";
            }
            
            addErrorMessage(`${errorMessage} (${error.message})`);
        });
    }
    
    // Health check for API
    function checkApiHealth() {
        try {
            fetch(`${API_URL}/status`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json"
                }
            })
            .then(response => {
                if (response.ok) {
                    console.log("API health check: OK");
                    const statusIndicator = document.getElementById('status-indicator');
                    if (statusIndicator) {
                        statusIndicator.className = 'status-indicator online';
                        statusIndicator.textContent = 'API Connected';
                    }
                    return true;
                } else {
                    console.warn("API health check: Failed", response.status);
                    const statusIndicator = document.getElementById('status-indicator');
                    if (statusIndicator) {
                        statusIndicator.className = 'status-indicator offline';
                        statusIndicator.textContent = 'API Offline';
                    }
                    return false;
                }
            })
            .catch(error => {
                console.error("API health check error:", error);
                const statusIndicator = document.getElementById('status-indicator');
                if (statusIndicator) {
                    statusIndicator.className = 'status-indicator offline';
                    statusIndicator.textContent = 'API Offline';
                }
                return false;
            });
        } catch (error) {
            console.error("API health check error:", error);
            return false;
        }
    }
    
    // Set up event listeners
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Do a health check initially
    checkApiHealth();
    
    // Periodically check API health
    setInterval(checkApiHealth, 30000); // Check every 30 seconds
    
    // Add initial welcome message
    addMessage("Welcome to Document Chat! Ask a question about your documents.");
});