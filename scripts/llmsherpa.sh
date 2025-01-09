docker rm -f llmsherpa
docker run --name llmsherpa -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest 