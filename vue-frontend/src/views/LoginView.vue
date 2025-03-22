<template>
    <div class="login-container">
      <div class="login-card">
        <h1>🇪🇺 Document Chat</h1>
        <h2>Login</h2>
        
        <form @submit.prevent="login">
          <div class="form-group">
            <label for="username">Username</label>
            <input 
              type="text" 
              id="username" 
              v-model="username" 
              required
              placeholder="Enter username"
            />
          </div>
          
          <div class="form-group">
            <label for="password">Password</label>
            <input 
              type="password" 
              id="password" 
              v-model="password" 
              required
              placeholder="Enter password"
            />
          </div>
          
          <p v-if="error" class="error-message">{{ error }}</p>
          
          <button type="submit" :disabled="loading">
            {{ loading ? 'Logging in...' : 'Login' }}
          </button>
        </form>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  import { useRouter } from 'vue-router';
  
  const router = useRouter();
  const username = ref('');
  const password = ref('');
  const error = ref('');
  const loading = ref(false);
  
  // For development, just use a simple mock login
  function login() {
    loading.value = true;
    error.value = '';
    
    // Simulate API call
    setTimeout(() => {
      // In production, this would be a real API call
      if (username.value === 'admin' && password.value === 'password') {
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('username', username.value);
        router.push('/');
      } else {
        error.value = 'Invalid username or password';
      }
      loading.value = false;
    }, 1000);
  }
  </script>
  
  <style scoped>
  .login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f5f7fb;
  }
  
  .login-card {
    width: 100%;
    max-width: 400px;
    padding: 2rem;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  h1 {
    text-align: center;
    margin-bottom: 0.5rem;
  }
  
  h2 {
    text-align: center;
    margin-bottom: 2rem;
    color: #555;
  }
  
  .form-group {
    margin-bottom: 1.5rem;
  }
  
  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }
  
  input {
    width: 100%;
    padding: 0.75rem;
    font-size: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
  }
  
  button {
    width: 100%;
    padding: 0.75rem;
    font-size: 1rem;
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  button:hover {
    background-color: #3a5cd7;
  }
  
  button:disabled {
    background-color: #a0aec0;
    cursor: not-allowed;
  }
  
  .error-message {
    color: #e53e3e;
    margin-bottom: 1rem;
  }
  </style>