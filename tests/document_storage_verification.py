import requests
import json
import time
from typing import Dict, List, Any

API_URL = "http://localhost:8000"  # Update if using a different address

def check_system_status() -> Dict[str, str]:
    """Check if all system components are running correctly"""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def count_indexed_documents() -> int:
    """Count how many unique documents are indexed in the system"""
    # This is a demonstration - you'd need to add an endpoint 
    # to your API that provides this information
    try:
        response = requests.get(f"{API_URL}/documents/count")
        response.raise_for_status()
        return response.json().get("count", 0)
    except Exception:
        return -1  # Indicates error or endpoint doesn't exist

def test_vector_search(query: str) -> Dict[str, Any]:
    """Test vector search functionality with a specific query"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"question": query},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_semantic_relationships(queries: List[str]) -> Dict[str, List]:
    """
    Test if semantically related queries return overlapping results,
    which indicates good vector embedding quality
    """
    results = {}
    all_filenames = set()
    
    for query in queries:
        search_result = test_vector_search(query)
        if "error" in search_result:
            results[query] = {"error": search_result["error"]}
            continue
            
        filenames = [obj.get("filename") for obj in search_result.get("results", [])]
        results[query] = filenames
        all_filenames.update(filenames)
    
    # Calculate overlap between results
    overlap_results = {}
    for i, query1 in enumerate(queries):
        overlaps = {}
        for j, query2 in enumerate(queries):
            if i != j:
                set1 = set(results[query1])
                set2 = set(results[query2])
                if set1 and set2:  # Ensure non-empty sets
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                    overlaps[query2] = f"{overlap:.2f}"
        overlap_results[query1] = overlaps
    
    return {
        "individual_results": results,
        "unique_documents": list(all_filenames),
        "semantic_overlaps": overlap_results
    }

def test_rag_quality(query: str) -> Dict[str, Any]:
    """Test RAG generation quality with a specific query"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_verification_tests():
    """Run a comprehensive set of verification tests"""
    print("🔍 Running verification tests for EU-compliant RAG system...")
    
    # Step 1: Check if all components are running
    print("\n1️⃣  Checking system status...")
    status = check_system_status()
    print(json.dumps(status, indent=2))
    
    if "api" not in status or status["api"] != "running":
        print("❌ API is not running. Verification cannot continue.")
        return
    
    # Step 2: Check document storage
    print("\n2️⃣  Testing vector search...")
    test_queries = [
        "What is gdpr?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = test_vector_search(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
            
        print(f"Found {len(result.get('results', []))} relevant chunks")
        for i, chunk in enumerate(result.get("results", [])[:2]):  # Show first 2 results
            print(f"Result {i+1}: {chunk.get('filename')} (Chunk {chunk.get('chunkId')})")
            # Truncate content for display
            content = chunk.get("content", "")
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"  {content}")
    
    # Step 3: Test semantic relationships
    print("\n3️⃣  Testing semantic relationships...")
    semantic_queries = [
        "EU data regulations",
        "GDPR compliance",
        "European privacy laws",
        "Document processing", # Unrelated query for contrast
    ]
    
    semantic_results = test_semantic_relationships(semantic_queries)
    print("\nSemantic overlaps between queries (0.00 to 1.00):")
    for query, overlaps in semantic_results["semantic_overlaps"].items():
        print(f"Query: '{query}'")
        for other_query, score in overlaps.items():
            print(f"  - Overlap with '{other_query}': {score}")
    
    # Step 4: Test RAG generation
    print("\n4️⃣  Testing RAG generation...")
    rag_query = "What is the GDPR?"
    rag_result = test_rag_quality(rag_query)
    
    if "error" in rag_result:
        print(f"❌ Error: {rag_result['error']}")
    else:
        print(f"Query: '{rag_query}'")
        print("\nGenerated answer:")
        print(rag_result.get("answer", "No answer generated"))
        print("\nSources:")
        for source in rag_result.get("sources", []):
            print(f"- {source.get('filename')} (Chunk {source.get('chunkId')})")
    
    print("\n✅ Verification tests completed!")

if __name__ == "__main__":
    run_verification_tests()