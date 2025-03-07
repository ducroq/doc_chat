# User Guide

This guide explains how to use the EU-Compliant Document Chat system to interact with your documents.

## Introduction

The Document Chat system allows you to ask questions about text documents in natural language. The system will search through your documents and provide relevant answers with source citations.

## Accessing the System

Access the web interface through your browser:

- **Development environment**: http://localhost:8501
- **Production environment**: http://your-domain.com (or IP address)

The interface has a clean, chat-based design that's intuitive to use.

## Asking Questions

1. **Type your question** in the input field at the bottom of the screen
2. **Press Enter** or click the send button
3. The system will process your question and provide an answer

### Example Questions

Good questions are specific and relate to content in the documents:

- "What are the key features of the RAG system?"
- "How does the document processor handle text chunking?"
- "What authentication methods are supported?"
- "Explain the token budget management"

## Understanding Responses

The system provides:

1. **Answer**: Generated text that addresses your question
2. **Sources**: References to the specific documents and chunks used to generate the answer

Example response:

```
The document processor chunks text into segments of approximately 1000 characters with 200 character overlap. This chunking strategy balances the need for chunks that are small enough for precise retrieval but large enough to provide adequate context.

Sources:
• processor.py (Chunk 3)
• docs/document_processing.md (Chunk 2)
```

### Source Citations

Source citations help you:
- Verify the information is correct
- Find more context in the original documents
- Understand where the answer is coming from

## Conversation History

Your conversation history is preserved during your session. You can:

- Scroll up to view previous questions and answers
- Ask follow-up questions that reference previous answers
- Start a new session by refreshing the page

## System Status

The sidebar shows the status of system components:

- ✅ API Service: Connected
- ✅ Vector Database: Connected
- ✅ LLM Service: Configured

If you see any ❌ error indicators, the system may not be fully operational.

## Best Practices

### For Better Results

1. **Be specific**: Ask clear, focused questions
2. **Use natural language**: You don't need special syntax
3. **Check sources**: Review the cited documents for more context
4. **Ask follow-ups**: If an answer is incomplete, ask for more details

### When You Don't Get Good Results

If the system doesn't provide a good answer:

1. **Rephrase your question**: Try asking in a different way
2. **Be more specific**: Add more details to your question
3. **Verify content exists**: Make sure the information is in your documents
4. **Check system status**: Ensure all components are working

## Limitations

The system has some limitations to be aware of:

- **Only knows what's in your documents**: It doesn't have general knowledge
- **Text files only**: Currently only processes .txt files
- **May not connect concepts**: Sometimes misses connections between different documents
- **Response rate limits**: There are limits on how many questions can be asked per minute
- **Token budget**: There's a daily limit on total usage

## Troubleshooting

### Common Issues

1. **Slow responses**: During peak times or with complex questions, responses may take longer
2. **Error messages**: If you see an error, try again after a short wait
3. **Incorrect answers**: Always verify with the source documents
4. **"I don't know" responses**: The information may not be in the documents

### Reporting Problems

If you encounter persistent issues:

1. Note the specific question that caused the problem
2. Capture any error messages
3. Contact your system administrator

## Privacy & Data Protection

The EU-Compliant Document Chat system is designed with privacy in mind:

- All data processing occurs within the system boundaries
- No user queries or document data are stored long-term
- The system uses EU-based service providers (Weaviate, Mistral AI)
- Complies with GDPR and other EU data protection regulations

Your conversations are not used to train AI models and are not shared with third parties.