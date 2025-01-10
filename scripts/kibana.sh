# kibana
docker rm -f kibana 
docker run -d --name Kibana \
  --network elastic \
  -p 127.0.0.1:5601:5601 \
  -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 \
  docker.elastic.co/kibana/kibana:8.15.0