docker system prune --volumes
docker container stop $(docker container ls -aq)
docker container prune
docker image prune
