import weaviate
import os
from dotenv import load_dotenv

# Load environment variables (if you have a .env file)
load_dotenv()

def check_weaviate_storage():
    """Check Weaviate directly to see if documents and embeddings are stored correctly"""
    print("🔍 Checking Weaviate storage directly...")
    
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(port=8080)
        
        # Check if the client is ready
        if not client.is_ready():
            print("❌ Weaviate is not ready.")
            return
        
        print("✅ Connected to Weaviate successfully.")
        
        # Get collection info
        if not client.collections.exists("DocumentChunk"):
            print("❌ DocumentChunk collection does not exist.")
            return
        
        collection = client.collections.get("DocumentChunk")
        
        # Get collection stats
        print("\n1️⃣ Collection information:")
        try:
            objects_count = collection.aggregate.over_all().with_meta_count().do()
            print(f"Total stored chunks: {objects_count.total_count}")
        except Exception as e:
            print(f"Error getting aggregation: {str(e)}")
        
        # Get unique filenames
        print("\n2️⃣ Unique documents:")
        try:
            result = collection.query.fetch_objects(
                return_properties=["filename"],
                limit=1000  # Adjust as needed
            )
            
            unique_filenames = set()
            for obj in result.objects:
                unique_filenames.add(obj.properties["filename"])
            
            print(f"Number of unique documents: {len(unique_filenames)}")
            for filename in unique_filenames:
                print(f"- {filename}")
        except Exception as e:
            print(f"Error getting unique filenames: {str(e)}")
        
        # Check embedding vectors - get a sample
        print("\n3️⃣ Checking sample vectors:")
        try:
            # We'll get a sample chunk to check its vector
            sample = collection.query.fetch_objects(
                return_properties=["content", "filename", "chunkId"],
                include_vector=True,
                limit=1
            )
            
            if sample.objects:
                obj = sample.objects[0]
                vector = obj.vector
                
                print(f"Sample from: {obj.properties['filename']} (Chunk {obj.properties['chunkId']})")
                
                if vector:
                    vector_length = len(vector)
                    print(f"Vector exists with dimension: {vector_length}")
                    print(f"Vector sample (first 5 elements): {vector[:5]}")
                else:
                    print("❌ No vector found for this object.")
            else:
                print("No objects found in the collection.")
        except Exception as e:
            print(f"Error checking vectors: {str(e)}")
        
        # Test a simple nearest search
        print("\n4️⃣ Testing nearest neighbor search:")
        try:
            search_term = "RAG system architecture"
            print(f"Searching for: '{search_term}'")
            
            results = collection.query.near_text(
                query=search_term,
                limit=3,
                return_properties=["content", "filename", "chunkId"]
            )
            
            print(f"Found {len(results.objects)} results")
            for i, obj in enumerate(results.objects):
                print(f"\nResult {i+1}: {obj.properties['filename']} (Chunk {obj.properties['chunkId']})")
                # Truncate content for display
                content = obj.properties['content']
                if len(content) > 100:
                    content = content[:97] + "..."
                print(f"  {content}")
                
        except Exception as e:
            print(f"Error performing search: {str(e)}")
        
        print("\n✅ Weaviate check completed!")
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    check_weaviate_storage()