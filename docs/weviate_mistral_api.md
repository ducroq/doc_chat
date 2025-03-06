Mistral Generative AI with Weaviate
Weaviate's integration with Mistral's APIs allows you to access their models' capabilities directly from Weaviate.

Configure a Weaviate collection to use a generative AI model with Mistral. Weaviate will perform retrieval augmented generation (RAG) using the specified model and your Mistral API key.

More specifically, Weaviate will perform a search, retrieve the most relevant objects, and then pass them to the Mistral generative model to generate outputs.

RAG integration illustration

Requirements
Weaviate configuration
Your Weaviate instance must be configured with the Mistral generative AI integration (generative-mistral) module.

For Weaviate Cloud (WCD) users
For self-hosted users
API credentials
You must provide a valid Mistral API key to Weaviate for this integration. Go to Mistral to sign up and obtain an API key.

Provide the API key to Weaviate using one of the following methods:

Set the MISTRAL_APIKEY environment variable that is available to Weaviate.
Provide the API key at runtime, as shown in the examples below.
Python API v4
JS/TS API v3
import weaviate
from weaviate.classes.init import Auth
import os

# Recommended: save sensitive data as environment variables
mistral_key = os.getenv("MISTRAL_APIKEY")
headers = {
    "X-Mistral-Api-Key": mistral_key,
}

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                       # `weaviate_url`: your Weaviate URL
    auth_credentials=Auth.api_key(weaviate_key),      # `weaviate_key`: your Weaviate API key
    headers=headers
)

# Work with Weaviate

client.close()

py docs  API docs
Configure collection
Generative model integration mutability
A collection's generative model integration configuration is mutable from v1.25.23, v1.26.8 and v1.27.1. See this section for details on how to update the collection configuration.

Configure a Weaviate index as follows to use a Mistral generative model:

Python API v4
JS/TS API v3
from weaviate.classes.config import Configure

client.collections.create(
    "DemoCollection",
    generative_config=Configure.Generative.mistral()
    # Additional parameters not shown
)

py docs  API docs
Select a model
You can specify one of the available models for Weaviate to use, as shown in the following configuration example:

Python API v4
JS/TS API v3
from weaviate.classes.config import Configure

client.collections.create(
    "DemoCollection",
    generative_config=Configure.Generative.mistral(
        model="mistral-large-latest"
    )
    # Additional parameters not shown
)

py docs  API docs
You can specify one of the available models for Weaviate to use. The default model is used if no model is specified.

Generative parameters
Configure the following generative parameters to customize the model behavior.

Python API v4
JS/TS API v3
from weaviate.classes.config import Configure

client.collections.create(
    "DemoCollection",
    generative_config=Configure.Generative.mistral(
        # # These parameters are optional
        # model="mistral-large",
        # temperature=0.7,
        # max_tokens=500,
    )
)

py docs  API docs
For further details on model parameters, see the Mistral API documentation.

Runtime parameters
You can provide the API key as well as some optional parameters at runtime through additional headers in the request. The following headers are available:

X-Mistral-Api-Key: The Mistral API key.
X-Mistral-Baseurl: The base URL to use (e.g. a proxy) instead of the default Mistral URL.
Any additional headers provided at runtime will override the existing Weaviate configuration.

Provide the headers as shown in the API credentials examples above.

Retrieval augmented generation
After configuring the generative AI integration, perform RAG operations, either with the single prompt or grouped task method.

Single prompt
Single prompt RAG integration generates individual outputs per search result

To generate text for each object in the search results, use the single prompt method.

The example below generates outputs for each of the n search results, where n is specified by the limit parameter.

When creating a single prompt query, use braces {} to interpolate the object properties you want Weaviate to pass on to the language model. For example, to pass on the object's title property, include {title} in the query.

Python API v4
JS/TS API v3
collection = client.collections.get("DemoCollection")

response = collection.generate.near_text(
    query="A holiday film",  # The model provider integration will automatically vectorize the query
    single_prompt="Translate this into French: {title}",
    limit=2
)

for obj in response.objects:
    print(obj.properties["title"])
    print(f"Generated output: {obj.generated}")  # Note that the generated output is per object


py docs  API docs
Grouped task
Grouped task RAG integration generates one output for the set of search results

To generate one text for the entire set of search results, use the grouped task method.

In other words, when you have n search results, the generative model generates one output for the entire group.

Python API v4
JS/TS API v3
collection = client.collections.get("DemoCollection")

response = collection.generate.near_text(
    query="A holiday film",  # The model provider integration will automatically vectorize the query
    grouped_task="Write a fun tweet to promote readers to check out these films.",
    limit=2
)

print(f"Generated output: {response.generated}")  # Note that the generated output is per query
for obj in response.objects:
    print(obj.properties["title"])


py docs  API docs
References
Available models
open-mistral-7b (aka mistral-tiny-2312) (default)
open-mixtral-8x7b (aka mistral-small-2312)
mistral-tiny
mistral-small
mistral-small-latest (aka mistral-small-2402)
mistral-medium
mistral-medium-latest (aka mistral-medium-2312)
mistral-large-latest (aka mistral-large-2402)
Further resources


Mistral Embeddings with Weaviate
Weaviate's integration with Mistral's APIs allows you to access their models' capabilities directly from Weaviate.

Configure a Weaviate vector index to use an Mistral embedding model, and Weaviate will generate embeddings for various operations using the specified model and your Mistral API key. This feature is called the vectorizer.

At import time, Weaviate generates text object embeddings and saves them into the index. For vector and hybrid search operations, Weaviate converts text queries into embeddings.

Embedding integration illustration

Requirements
Weaviate configuration
Your Weaviate instance must be configured with the Mistral vectorizer integration (text2vec-mistral) module.

For Weaviate Cloud (WCD) users
For self-hosted users
API credentials
You must provide a valid Mistral API key to Weaviate for this integration. Go to Mistral to sign up and obtain an API key.

Provide the API key to Weaviate using one of the following methods:

Set the MISTRAL_APIKEY environment variable that is available to Weaviate.
Provide the API key at runtime, as shown in the examples below.
Python API v4
JS/TS API v3
Go
import weaviate
from weaviate.classes.init import Auth
import os

# Recommended: save sensitive data as environment variables
mistral_key = os.getenv("MISTRAL_APIKEY")
headers = {
    "X-Mistral-Api-Key": mistral_key,
}

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                       # `weaviate_url`: your Weaviate URL
    auth_credentials=Auth.api_key(weaviate_key),      # `weaviate_key`: your Weaviate API key
    headers=headers
)

# Work with Weaviate

client.close()

py docs  API docs
Configure the vectorizer
Configure a Weaviate index as follows to use an Mistral embedding model:

Python API v4
JS/TS API v3
Go
from weaviate.classes.config import Configure

client.collections.create(
    "DemoCollection",
    vectorizer_config=[
        Configure.NamedVectors.text2vec_mistral(
            name="title_vector",
            source_properties=["title"],
        )
    ],
    # Additional parameters not shown
)

py docs  API docs
Select a model
You can specify one of the available models for the vectorizer to use, as shown in the following configuration examples.

Python API v4
JS/TS API v3
Go
from weaviate.classes.config import Configure

client.collections.create(
    "DemoCollection",
    vectorizer_config=[
        Configure.NamedVectors.text2vec_mistral(
            name="title_vector",
            source_properties=["title"],
            model="mistral-embed"
        )
    ],
    # Additional parameters not shown
)

py docs  API docs
The default model is used if no model is specified.

Vectorization behavior
Runtime parameters
You can provide the API key as well as some optional parameters at runtime through additional headers in the request. The following headers are available:

X-Mistral-Api-Key: The Mistral API key.
X-Mistral-Baseurl: The base URL to use (e.g. a proxy) instead of the default Mistral URL.
Any additional headers provided at runtime will override the existing Weaviate configuration.

Provide the headers as shown in the API credentials examples above.

Data import
After configuring the vectorizer, import data into Weaviate. Weaviate generates embeddings for text objects using the specified model.

Python API v4
JS/TS API v3
Go
source_objects = [
    {"title": "The Shawshank Redemption", "description": "A wrongfully imprisoned man forms an inspiring friendship while finding hope and redemption in the darkest of places."},
    {"title": "The Godfather", "description": "A powerful mafia family struggles to balance loyalty, power, and betrayal in this iconic crime saga."},
    {"title": "The Dark Knight", "description": "Batman faces his greatest challenge as he battles the chaos unleashed by the Joker in Gotham City."},
    {"title": "Jingle All the Way", "description": "A desperate father goes to hilarious lengths to secure the season's hottest toy for his son on Christmas Eve."},
    {"title": "A Christmas Carol", "description": "A miserly old man is transformed after being visited by three ghosts on Christmas Eve in this timeless tale of redemption."}
]

collection = client.collections.get("DemoCollection")

with collection.batch.dynamic() as batch:
    for src_obj in source_objects:
        # The model provider integration will automatically vectorize the object
        batch.add_object(
            properties={
                "title": src_obj["title"],
                "description": src_obj["description"],
            },
            # vector=vector  # Optionally provide a pre-obtained vector
        )
        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break

failed_objects = collection.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
    print(f"First failed object: {failed_objects[0]}")


py docs  API docs
Re-use existing vectors
If you already have a compatible model vector available, you can provide it directly to Weaviate. This can be useful if you have already generated embeddings using the same model and want to use them in Weaviate, such as when migrating data from another system.

Searches
Once the vectorizer is configured, Weaviate will perform vector and hybrid search operations using the specified Mistral model.

Embedding integration at search illustration

Vector (near text) search
When you perform a vector search, Weaviate converts the text query into an embedding using the specified model and returns the most similar objects from the database.

The query below returns the n most similar objects from the database, set by limit.

Python API v4
JS/TS API v3
Go
collection = client.collections.get("DemoCollection")

response = collection.query.near_text(
    query="A holiday film",  # The model provider integration will automatically vectorize the query
    limit=2
)

for obj in response.objects:
    print(obj.properties["title"])


py docs  API docs
Hybrid search
What is a hybrid search?
A hybrid search performs a vector search and a keyword (BM25) search, before combining the results to return the best matching objects from the database.

When you perform a hybrid search, Weaviate converts the text query into an embedding using the specified model and returns the best scoring objects from the database.

The query below returns the n best scoring objects from the database, set by limit.

Python API v4
JS/TS API v3
Go
collection = client.collections.get("DemoCollection")

response = collection.query.hybrid(
    query="A holiday film",  # The model provider integration will automatically vectorize the query
    limit=2
)

for obj in response.objects:
    print(obj.properties["title"])


py docs  API docs
References
Available models
As of September 2024, the only available model is mistral-embed.

Further resources
Other integrations
Mistral generative models + Weaviate.
Code examples
Once the integrations are configured at the collection, the data management and search operations in Weaviate work identically to any other collection. See the following model-agnostic examples:

The how-to: manage data guides show how to perform data operations (i.e. create, update, delete).
The how-to: search guides show how to perform search operations (i.e. vector, keyword, hybrid) as well as retrieval augmented generation.