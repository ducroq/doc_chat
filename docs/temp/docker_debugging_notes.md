

```bash
docker-compose config    
```

```bash
docker-compose ps  
```

Stop all running containers
```bash
docker-compose down
```

List all Docker volumes to find the Weaviate data volume
```bash
docker volume ls
```

Remove the Weaviate data volume
```bash
docker volume rm doc_chat_weaviate_data
```


Rebuild and restart a container, e.g. the processor:
```bash
docker-compose stop processor
docker-compose build processor
docker-compose up -d processor
```

Check the logs to see if it's processing the existing files (--follow gives real-time logs):

```bash
docker logs doc_chat-processor-1 --follow
```

Check if the processor environment variables are correct:
```bash
docker inspect doc_chat-processor-1 | Select-String "DATA_FOLDER"
```

List the contents of the data directory as seen by the container:
```bash
docker exec doc_chat-processor-1 ls -la /data
```

Print the current working directory in the container
```bash
docker exec doc_chat-processor-1 pwd
```

Check the environment variables to see what DATA_FOLDER is set to
```bash
docker exec doc_chat-processor-1 printenv | findstr DATA_FOLDER
```

Check if the processor container can read a specific file
```bash
docker exec doc_chat-processor-1 cat /data/gdpr_info.txt
```


